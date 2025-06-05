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
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Model Selection imports
from .model_selection.statistical import ZScoreSelector, AllModelsSelector, RandomSelector
from .model_selection.learned import LLMBlenderSelector, MetaLearningSelector

# Output Aggregation imports
from .output_aggregation.sentence_level.reward_based import RewardBasedSelector
from .output_aggregation.sentence_level.random_selector import RandomSentenceSelector
from .output_aggregation.sentence_level.round_robin import RoundRobinSelector
from .output_aggregation.sentence_level.progressive_selector import ProgressiveSelector
from .output_aggregation.token_level.distribution import DistributionAggregator, WeightedAverageAggregator
from .output_aggregation.token_level.gac import GaCTokenAggregator
from .output_aggregation.response_level.base import BaseResponseAggregator

from ensemblehub.generators import GeneratorPool
from ensemblehub.scorers.base import ScorerPool

logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    # Model Selection Configuration
    use_model_selection: bool = True
    model_selection_method: str = "zscore"  # zscore, all, random, llm_blender, meta_learning
    model_selection_params: Dict[str, Any] = None
    
    # Output Aggregation Configuration  
    use_output_aggregation: bool = True
    aggregation_method: str = "loop"  # reward_based, random, round_robin, gac, distribution
    aggregation_level: str = "sentence"  # sentence, token, response
    aggregation_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_selection_params is None:
            self.model_selection_params = {}
        if self.aggregation_params is None:
            self.aggregation_params = {}


class EnsembleFramework:
    """
    Unified ensemble framework that combines model selection and output aggregation.
    """
    
    # Registry of available methods
    MODEL_SELECTORS = {
        "zscore": ZScoreSelector,
        "all": AllModelsSelector, 
        "random": RandomSelector,
        "llm_blender": LLMBlenderSelector,
        "meta_learning": MetaLearningSelector,
    }
    
    SENTENCE_AGGREGATORS = {
        "reward_based": RewardBasedSelector,
        "random": RandomSentenceSelector,
        "loop": RoundRobinSelector,
        "progressive": ProgressiveSelector,
    }
    
    TOKEN_AGGREGATORS = {
        "gac": GaCTokenAggregator,
        "distribution": DistributionAggregator,
        "weighted_average": WeightedAverageAggregator,
    }
    
    RESPONSE_AGGREGATORS = {
        # TODO: Implement when response-level methods are ready
        # "majority_voting": MajorityVoting,
        # "weighted_voting": WeightedVoting,
        # "generative_fusion": GenerativeFusion,
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
        if self.config.use_model_selection:
            selector_class = self.MODEL_SELECTORS.get(self.config.model_selection_method)
            if selector_class is None:
                raise ValueError(f"Unknown model selection method: {self.config.model_selection_method}")
            
            self.model_selector = selector_class(**self.config.model_selection_params)
        
        # Initialize output aggregator
        if self.config.use_output_aggregation:
            if self.config.aggregation_level == "sentence":
                aggregator_class = self.SENTENCE_AGGREGATORS.get(self.config.aggregation_method)
            elif self.config.aggregation_level == "token":
                aggregator_class = self.TOKEN_AGGREGATORS.get(self.config.aggregation_method)
            elif self.config.aggregation_level == "response":
                aggregator_class = self.RESPONSE_AGGREGATORS.get(self.config.aggregation_method)
            else:
                raise ValueError(f"Unknown aggregation level: {self.config.aggregation_level}")
            
            if aggregator_class is None:
                raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method} for level {self.config.aggregation_level}")
            
            # Filter constructor parameters - some params are for the method calls, not constructor
            constructor_params = {}
            if aggregator_class.__name__ == "RewardBasedSelector":
                # RewardBasedSelector only accepts specific constructor params
                for key in ["exclude_self_scoring", "max_repeat", "name"]:
                    if key in self.config.aggregation_params:
                        constructor_params[key] = self.config.aggregation_params[key]
            elif aggregator_class.__name__ == "ProgressiveSelector":
                # ProgressiveSelector only accepts specific constructor params
                for key in ["switch_mode", "length_thresholds", "special_tokens", "max_repeat", "name"]:
                    if key in self.config.aggregation_params:
                        constructor_params[key] = self.config.aggregation_params[key]
            elif aggregator_class.__name__ in ["RandomSentenceSelector", "RoundRobinSelector"]:
                # These only accept max_repeat and name
                for key in ["max_repeat", "name"]:
                    if key in self.config.aggregation_params:
                        constructor_params[key] = self.config.aggregation_params[key]
            else:
                constructor_params = self.config.aggregation_params
                
            self.output_aggregator = aggregator_class(**constructor_params)
            logger.info(f"Initialized output aggregator: {self.config.aggregation_method} ({self.config.aggregation_level}-level)")
    
    def run_ensemble(
        self,
        examples: List,
        model_specs: List[Dict[str, Any]],
        generators,
        scorers,
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        is_chat: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run the complete ensemble pipeline with batch support.
        
        Args:
            examples: List of input examples, each with instruction, input, output
            model_specs: List of model specifications
            generators: Generator instances or pool
            scorers: Scorer instances or pool
            model_stats: Model statistics for selection
            **kwargs: Additional parameters
            
        Returns:
            List of Dict with output, selected_models, and method info
        """
        
        logger.info(f"Running ensemble with config: selection={self.config.use_model_selection}, aggregation={self.config.use_output_aggregation}, batch_size={len(examples)}")
        
        # For now, process each example separately
        # TODO: In future, optimize model selection for batch processing
        results = []
        
        for idx, single_example in enumerate(examples):
            # Stage 1: Model Selection (per example for now)
            selected_specs = model_specs  # Default: use all models
            if self.config.use_model_selection and self.model_selector:
                selected_specs = self.model_selector.select_models(
                    example=single_example,
                    model_specs=model_specs,
                    model_stats=model_stats,
                    **kwargs
                )
        
        # Stage 2: Output Generation/Aggregation
        if self.config.use_output_aggregation and self.output_aggregator and (len(selected_specs) > 1):
            logger.info(f"🔗 Stage 2: Output Aggregation ({self.config.aggregation_method} - {self.config.aggregation_level})")
            
            # Get generators for selected models
            if hasattr(generators, 'get_generator'):
                # Generator pool
                selected_generators = []
                for spec in selected_specs:
                    gen = generators.get_generator(
                        spec["path"], 
                        spec.get("engine", "hf"), 
                        spec.get("device"), 
                        spec.get("quantization", "none"), 
                        spec.get("enable_thinking", False)
                    )
                    selected_generators.append(gen)
            else:
                # Assume generators is already a list
                selected_generators = generators[:len(selected_specs)]
            
            # Run aggregation - output_aggregator should handle batch
            outputs = self.output_aggregator.aggregate_generation(
                generators=selected_generators,
                scorers=scorers,
                examples=examples,
                is_chat=is_chat,
                **kwargs
            )
            
        elif len(selected_specs) == 1:
            # Simple fallback: use first selected model
            if hasattr(generators, 'get_generator'):
                first_gen = generators.get_generator(
                    selected_specs[0]["path"], 
                    selected_specs[0].get("engine", "hf"), 
                    selected_specs[0].get("device"), 
                    selected_specs[0].get("quantization", "none"), 
                    selected_specs[0].get("enable_thinking", False)
                )
            else:
                first_gen = generators[0]
            
            # Determine max_tokens based on generator type and model capabilities
            default_max_tokens = 256  # fallback default
            
            # Check generator type and get max tokens accordingly
            if hasattr(first_gen, '__class__'):
                gen_class_name = first_gen.__class__.__name__
                logger.info(f"Generator type: {gen_class_name}")
                
                if gen_class_name == "HFGenerator" and hasattr(first_gen, 'tokenizer'):
                    # For HuggingFace models, try to get model_max_length
                    if hasattr(first_gen.tokenizer, 'model_max_length'):
                        model_max_length = first_gen.tokenizer.model_max_length
                        if model_max_length and model_max_length < 1000000:  # Sanity check
                            # Use 75% of model max length for generation
                            default_max_tokens = int(model_max_length * 0.75)
                            logger.info(f"Single model: Using max_tokens={default_max_tokens} (75% of HF model max length {model_max_length})")
                        else:
                            default_max_tokens = 32768
                            logger.info(f"Single model: Using safe max_tokens={default_max_tokens} (HF model max length unreasonable)")
                    else:
                        default_max_tokens = 32768
                        logger.info(f"Single model: Using default max_tokens={default_max_tokens} (HF model max length not available)")
                
                elif gen_class_name == "VLLMGenerator":
                    # For vLLM models, they typically support longer contexts
                    # vLLM internally handles max tokens well, so we can use a large default
                    default_max_tokens = 32768
                    logger.info(f"Single model: Using max_tokens={default_max_tokens} for vLLM generator")
                
                else:
                    # Unknown generator type, use a reasonable default
                    default_max_tokens = 16384
                    logger.info(f"Single model: Using default max_tokens={default_max_tokens} for {gen_class_name}")
            else:
                # Can't determine generator type
                default_max_tokens = 16384
                logger.info(f"Single model: Using fallback max_tokens={default_max_tokens}")
            
            # Prepare generation parameters
            gen_kwargs = {
                "max_tokens": kwargs.get("max_tokens", default_max_tokens),
                "temperature": kwargs.get("temperature", 0.95),
                "top_p": kwargs.get("top_p", 0.7),
            }
            if "seed" in kwargs:
                gen_kwargs["seed"] = kwargs["seed"]
            if "stop_strings" in kwargs:
                gen_kwargs["stop_strings"] = kwargs["stop_strings"]
            
            # Pass the batch to the generator - generator should handle batch
            batch_results = first_gen.generate(inputs=examples, is_chat=is_chat, **gen_kwargs)
            outputs = [res.text if hasattr(res, 'text') else res for res in batch_results]
        else:
            # No models selected
            logger.error("No models selected for generation")
            outputs = [""] * len(examples)
        
        # Get model attribution data if available
        attribution_data = None
        if self.config.use_output_aggregation and hasattr(self.output_aggregator, 'get_attribution_data'):
            try:
                attribution_data = self.output_aggregator.get_attribution_data()
                if attribution_data:
                    logger.debug(f"Retrieved attribution data with keys: {list(attribution_data.keys())}")
            except Exception as e:
                logger.warning(f"Failed to get attribution data: {e}")
        
        # Create results for each example
        selected_paths = [s['path'] for s in selected_specs]
        method_name = f"{self.config.model_selection_method if self.config.use_model_selection else 'no_selection'}+{self.config.aggregation_method if self.config.use_output_aggregation else 'no_aggregation'}"
        
        results = []
        for output in outputs:
            result = {
                "output": output,
                "selected_models": selected_paths,
                "method": method_name,
                "config": {
                    "model_selection": self.config.model_selection_method if self.config.use_model_selection else None,
                    "output_aggregation": f"{self.config.aggregation_method}_{self.config.aggregation_level}" if self.config.use_output_aggregation else None,
                }
            }
            
            # Add attribution data if available
            if attribution_data:
                result["attribution"] = attribution_data
                logger.debug(f"Added attribution data to result")
            
            results.append(result)
            
        return results


def run_ensemble(
    examples: List,
    model_specs: List[Dict] = None,
    reward_spec: List[Dict] = None,
    ensemble_method: str = "loop",
    model_selection_method: str = "zscore",
    max_rounds: int = 500,
    score_threshold: float = -2.0,
    progressive_mode: str = "length",
    length_thresholds: List[int] = None,
    special_tokens: List[str] = None,
    is_chat: bool = False,
    **kwargs
) -> List:
    """
    New unified ensemble function using the refactored architecture.
    Now uses the EnsembleFramework for better organization.
    Supports both single example and batch processing.

    Args:
        example: Single input example with "instruction", "input", "output" (deprecated)
        examples: List of input examples (preferred for batch processing)
        model_specs: List of model specifications
        reward_spec: List of reward model specifications
        ensemble_method: Output aggregation method ("reward_based", "random", "round_robin")
        model_selection_method: Model selection method ("zscore", "all", "random")
        max_rounds: Maximum generation rounds
        score_threshold: Score threshold for early stopping

    Returns:
        Dict with "output" and "selected_models" for single example
        List of Dicts for batch processing
    """

    # Initialize pools
    model_pool = GeneratorPool()
    scorers = ScorerPool()

    # Load model statistics
    model_stats = get_default_model_stats()

    # Load external scorers
    for spec in reward_spec:
        try:
            scorers.get_scorer(spec)
            logger.info(f"✅ Loaded scorer: {spec.get('path', 'unknown')}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load scorer {spec.get('path', 'unknown')}: {e}")


    aggregation_method = ensemble_method

    # Handle progressive-specific parameters (only constructor params, not runtime params)
    aggregation_params = {}
    if ensemble_method == "progressive":
        aggregation_params.update({
            "switch_mode": progressive_mode,
            "length_thresholds": length_thresholds or [1000, 2000, 3000],
            "special_tokens": special_tokens or [r"<\think>"]
        })

    # Note: max_rounds and score_threshold are runtime parameters, not constructor parameters

    # Create ensemble framework
    config = EnsembleConfig(
        use_model_selection=True,
        model_selection_method=model_selection_method,
        model_selection_params={"model_count": -1} if model_selection_method == "zscore" else {},
        use_output_aggregation=True,
        aggregation_method=aggregation_method,
        aggregation_level="sentence",
        aggregation_params=aggregation_params
    )

    framework = EnsembleFramework(config)

    # Run ensemble
    results = framework.run_ensemble(
        examples=examples,
        model_specs=model_specs,
        generators=model_pool,
        scorers=scorers,
        model_stats=model_stats,
        max_rounds=max_rounds,
        score_threshold=score_threshold,
        is_chat=is_chat,
        **kwargs
    )

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