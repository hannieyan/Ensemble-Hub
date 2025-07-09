"""
Unified Ensemble Framework for Ensemble-Hub

This module provides a unified interface for ensemble methods that can combine:
1. Model Selection (pre-inference): Choose which models to use
2. Output Aggregation (during/post-inference): Combine model outputs

The framework supports flexible combinations:
- Model selection only: Use selected models independently
- Output aggregation only: Use all models with aggregation
- Both: Select models then aggregate their outputs
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import ray
import torch
from pydantic import BaseModel, Field

from .model_selection.learned import MetaLearningSelector
# Model Selection imports
from .model_selection.statistical import ZScoreSelector, AllModelsSelector, JudgmentSelector
from .output_aggregation.sentence_level.progressive_selector import ProgressiveSelector
from .output_aggregation.sentence_level.random_selector import RandomSentenceSelector
# Output Aggregation imports
from .output_aggregation.sentence_level.reward_based import RewardBasedSelector
from .output_aggregation.sentence_level.loop_selector import LoopSelector
from .output_aggregation.token_level.distribution import DistributionAggregator, WeightedAverageAggregator
from .output_aggregation.token_level.gac import GaCTokenAggregator
from ..generators.hf_engine import get_remote_hf_generator_class

logger = logging.getLogger(__name__)


class EnsembleConfig(BaseModel):
    """Core ensemble configuration parameters"""
    
    # Core ensemble method configuration
    model_selection_method: str = Field(default="all", description="Model selection: zscore, all, model_judgment")
    model_selection_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for model selection method")
    output_aggregation_method: str = Field(default="loop", description="Output aggregation method: reward_based, random, loop, progressive, etc.")
    output_aggregation_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for output aggregation method (includes reward_specs, score_threshold, etc.)")

    max_rounds: int = Field(default=500, description="Maximum generation rounds")
    
    # Global generation parameters
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    stop_strings: List[str] = Field(default_factory=list, description="Default stop strings for generation")
    seed: Optional[int] = Field(default=None, description="Random seed")
    
    # Debug options
    show_output_details: bool = Field(default=False, description="Show detailed output in logs")
    show_input_details: bool = Field(default=False, description="Show raw request in logs")
    enable_thinking: bool = Field(default=False, description="Enable thinking mode for supported models")
    save_results: bool = Field(default=False, description="Save results to saves/logs directory")
    
    # Server configuration (not part of ensemble logic, but needed somewhere)
    model_specs: List[Dict[str, Any]] = Field(default_factory=list, description="Model specifications")


class EnsembleFramework:
    """
    Unified ensemble framework that combines model selection and output aggregation.
    """
    
    # Registry of available methods
    MODEL_SELECTORS = {
        "zscore": ZScoreSelector,
        "all": AllModelsSelector,
        "learned": MetaLearningSelector,
        "model_judgment": JudgmentSelector,  # Alias for backward compatibility
    }

    # Unified registry with metadata
    OUTPUT_AGGREGATORS = {
        # Sentence-level aggregators
        "reward_based": (RewardBasedSelector, "sentence", ["exclude_self_scoring", "max_repeat", "name", "reward_specs", "score_threshold"]),
        "random": (RandomSentenceSelector, "sentence", ["max_repeat", "name"]),
        "loop": (LoopSelector, "sentence", ["max_repeat", "name"]),
        "progressive": (ProgressiveSelector, "sentence", ["outline_max_tokens", "outline_prompt_template", "final_prompt_template", "template_language", "name"]),
        
        # Token-level aggregators
        "gac": (GaCTokenAggregator, "token", None),
        "distribution": (DistributionAggregator, "token", None),
        "weighted_average": (WeightedAverageAggregator, "token", None),
        
        # Response-level aggregators (not implemented yet)
        # "majority_voting": (MajorityVoting, "response", None),
        # "weighted_voting": (WeightedVoting, "response", None),
        # "generative_fusion": (GenerativeFusion, "response", None),
    }
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.model_selector = None
        self.output_aggregator = None
        
        # Initialize components based on configuration
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize model selector and output aggregator based on config."""
        
        # Initialize model selector
        selector_class = self.MODEL_SELECTORS[self.config.model_selection_method]
        self.model_selector = selector_class(**self.config.model_selection_params)

        # Initialize output aggregator
        aggregator_data = self.OUTPUT_AGGREGATORS[self.config.output_aggregation_method]
        aggregator_class, level, allowed_params = aggregator_data
        
        # Filter constructor parameters based on metadata
        constructor_params = {}
        if allowed_params:
            for key in allowed_params:
                if key in self.config.output_aggregation_params:
                    constructor_params[key] = self.config.output_aggregation_params[key]
        else:
            # If no specific params defined, pass all
            constructor_params = self.config.output_aggregation_params

        self.output_aggregator = aggregator_class(**constructor_params)
        self.aggregation_level = level
        logger.info(f"Initialized output aggregator: {self.config.output_aggregation_method} ({level}-level)")



    def ensemble(
        self,
        examples: List,
        model_specs: List[Dict[str, Any]],
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        is_chat: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run the complete ensemble pipeline with batch support.
        
        Args:
            examples: List of input examples, each with instruction, input, output
            model_specs: List of model specifications
            model_stats: Model statistics for selection
            is_chat: Whether the input examples are in chat format (list of dicts)
            **kwargs: Additional parameters
            
        Returns:
            List of Dict with output, selected_models, and method info
        """
        
        logger.info(f"Running ensemble with config: selection={self.config.model_selection_method}, aggregation={self.config.output_aggregation_method}, batch_size={len(examples)}")

        # Load all generators first
        generators = {}
        
        for spec in model_specs:
            # Try to get existing actor first
            try:
                generator = ray.get_actor(spec["path"])
                logger.info(f"✅ Reusing existing actor: {spec['path']}")
            except ValueError:
                # Actor doesn't exist, create new one
                # Default GPU allocation: 0.5 for shared GPU usage, 0 for CPU-only
                default_gpus = 1 if torch.cuda.is_available() else 0
                actor = get_remote_hf_generator_class(spec.get("num_gpus", default_gpus))
                generator = actor.options(name=spec["path"], lifetime="detached").remote(
                    model_path=spec["path"],
                    max_memory=spec.get("max_memory", None),
                    dtype=torch.bfloat16,
                    quantization=spec.get("quantization", "none"),
                    enable_thinking=spec.get("enable_thinking", False),
                )
                logger.info(f"✅ Created new actor: {spec['path']}")
            generators[spec["path"]] = generator

        # Model selection stage
        logger.info(f"🎯 Stage 1: Model Selection ({self.config.model_selection_method})")
        selected_specs = self.model_selector.select_models(
            example=examples,
            model_specs=model_specs,
            model_stats=model_stats,
            generators=generators,  # Pass Ray generators to selector
            **kwargs
        )

        # Create selected generators dict from selected specs
        selected_generators = [
            generators[spec['path']] for spec in selected_specs if spec['path'] in generators
        ]
        logger.info(f"Selected {len(selected_generators)} models: {list(spec['path'] for spec in selected_specs)}")

        # Stage 2: Output Generation/Aggregation
        logger.info(f"🔗 Stage 2: Output Aggregation ({self.config.output_aggregation_method} - {self.aggregation_level})")

        # Run aggregation - output_aggregator should handle batch
        outputs = self.output_aggregator.aggregate_generation(
            generators=selected_generators,
            examples=examples,
            is_chat=is_chat,
            **kwargs
        )
        
        # Get model attribution data if available
        attribution_data = None
        if hasattr(self.output_aggregator, 'get_attribution_data'):
            attribution_data = self.output_aggregator.get_attribution_data()
            # logger.debug(f"Retrieved attribution data with keys: {list(attribution_data.keys())}")
        
        # Create results for each example
        selected_paths = [s['path'] for s in selected_specs]
        method_name = f"{self.config.model_selection_method}+{self.config.output_aggregation_method}"
        
        results = []
        for i, output in enumerate(outputs):
            result = {
                "output": output,
                "selected_models": selected_paths,
                "method": method_name,
                "config": {
                    "model_selection": self.config.model_selection_method,
                    "output_aggregation": f"{self.config.output_aggregation_method}_{self.aggregation_level}",
                }
            }
            
            # Add attribution data if available
            if attribution_data and i < len(attribution_data):
                result["attribution"] = attribution_data[i]
                logger.debug(f"Added attribution data to result {i}")
            
            results.append(result)
            
        return results



def get_default_model_stats() -> Dict[str, Dict[str, float]]:
    """
    Get default model statistics. In production, this should load from a file.
    """
    return {
        "Qwen/Qwen2.5-0.5B-Instruct": {
            "ppl_mean": 9.795982360839844,
            "ppl_std": 22.284496307373047,
            "conf_mean": 0.6799513101577759,
            "conf_std": 0.08082679659128189,
            "weight": 0.2,
            "size": 0.5
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
            "ppl_mean": 9.795982360839844,
            "ppl_std": 22.284496307373047,
            "conf_mean": 0.6799513101577759,
            "conf_std": 0.08082679659128189,
            "weight": 0.2,
            "size": 1.5
        },
        "Qwen/Qwen3-4B": {
            "ppl_mean": 6.160105228424072,
            "ppl_std": 6.118084907531738,
            "conf_mean": 0.8231604099273682,
            "conf_std": 0.07646501809358597,
            "weight": 1.0,
            "size": 4.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
            "ppl_mean": 16.57339096069336,
            "ppl_std": 50.37682342529297,
            "conf_mean": 0.6976740956306458,
            "conf_std": 0.10360505431890488,
            "weight": 0.5,
            "size": 7.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
            "ppl_mean": 8.22177505493164,
            "ppl_std": 14.440741539001465,
            "conf_mean": 0.7438507676124573,
            "conf_std": 0.0863514393568039,
            "weight": 1.0,
            "size": 14.0
        },
        "Qwen/Qwen2.5-Math-7B-Instruct": {
            'ppl_mean': 4.232998847961426,
            'ppl_std': 3.664811611175537,
            'conf_mean': 0.7785097360610962,
            'conf_std': 0.09053431451320648,
            "weight": 1.0,
            "size": 7.0
        },
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
            "ppl_mean": 4.0472869873046875,
            "ppl_std": 3.9851391315460205,
            "conf_mean": 0.7702987194061279,
            "conf_std": 0.0831739529967308,
            "weight": 1.0,
            "size": 32.0
        }
    }