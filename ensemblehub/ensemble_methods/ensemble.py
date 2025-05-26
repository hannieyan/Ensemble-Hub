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
    aggregation_method: str = "reward_based"  # reward_based, random, round_robin, gac, distribution
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
        "round_robin": RoundRobinSelector,
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
            logger.info(f"Initialized model selector: {self.config.model_selection_method}")
        
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
        example: Dict[str, Any],
        model_specs: List[Dict[str, Any]],
        generators,
        scorers,
        model_stats: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete ensemble pipeline.
        
        Args:
            example: Input example with instruction, input, output
            model_specs: List of model specifications
            generators: Generator instances or pool
            scorers: Scorer instances or pool
            model_stats: Model statistics for selection
            **kwargs: Additional parameters
            
        Returns:
            Dict with output, selected_models, and method info
        """
        logger.info(f"Running ensemble with config: selection={self.config.use_model_selection}, aggregation={self.config.use_output_aggregation}")
        
        # Stage 1: Model Selection
        selected_specs = model_specs  # Default: use all models
        if self.config.use_model_selection and self.model_selector:
            logger.info(f"🔍 Stage 1: Model Selection ({self.config.model_selection_method})")
            selected_specs = self.model_selector.select_models(
                example=example,
                model_specs=model_specs,
                model_stats=model_stats,
                **kwargs
            )
            logger.info(f"✅ Selected {len(selected_specs)} models: {[s['path'] for s in selected_specs]}")
        else:
            logger.info("⏭️  Skipping model selection, using all models")
        
        # Stage 2: Output Generation/Aggregation
        if self.config.use_output_aggregation and self.output_aggregator:
            logger.info(f"🔗 Stage 2: Output Aggregation ({self.config.aggregation_method} - {self.config.aggregation_level})")
            
            # Get generators for selected models
            if hasattr(generators, 'get_generator'):
                # Generator pool
                selected_generators = []
                for spec in selected_specs:
                    gen = generators.get_generator(spec["path"], spec.get("engine", "hf"), spec.get("device"))
                    selected_generators.append(gen)
            else:
                # Assume generators is already a list
                selected_generators = generators[:len(selected_specs)]
            
            # Run aggregation
            output = self.output_aggregator.aggregate_generation(
                generators=selected_generators,
                scorers=scorers,
                example=example,
                **kwargs
            )
            
            logger.info("✅ Output aggregation completed")
        else:
            logger.info("⏭️  Skipping output aggregation, using first model")
            # Simple fallback: use first selected model
            if hasattr(generators, 'get_generator'):
                first_gen = generators.get_generator(selected_specs[0]["path"], selected_specs[0].get("engine", "hf"), selected_specs[0].get("device"))
            else:
                first_gen = generators[0]
            
            # Generate simple output
            from ...conversation import ConversationTemplate
            question = example.get("instruction", "") + " " + example.get("input", "")
            conversation = ConversationTemplate("You are a helpful assistant.", question)
            prompt = conversation.render()
            result = first_gen.generate({"prompt": prompt}, max_tokens=kwargs.get("max_tokens", 256))
            output = result.text
        
        # Get model attribution data if available
        attribution_data = None
        if self.config.use_output_aggregation and hasattr(self.output_aggregator, 'get_attribution_data'):
            try:
                attribution_data = self.output_aggregator.get_attribution_data()
                if attribution_data:
                    logger.debug(f"Retrieved attribution data with keys: {list(attribution_data.keys())}")
            except Exception as e:
                logger.warning(f"Failed to get attribution data: {e}")
        
        # Return results
        selected_paths = [s['path'] for s in selected_specs]
        method_name = f"{self.config.model_selection_method if self.config.use_model_selection else 'no_selection'}+{self.config.aggregation_method if self.config.use_output_aggregation else 'no_aggregation'}"
        
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
            
        return result
    
    @classmethod
    def create_simple_ensemble(cls, ensemble_method: str = "reward_based", model_selection_method: str = "all") -> 'EnsembleFramework':
        """
        Factory method for creating simple ensemble configurations.
        
        Args:
            ensemble_method: Output aggregation method
            model_selection_method: Model selection method
            
        Returns:
            Configured EnsembleFramework
        """
        config = EnsembleConfig(
            use_model_selection=True,
            model_selection_method=model_selection_method,
            use_output_aggregation=True,
            aggregation_method=ensemble_method,
            aggregation_level="sentence"
        )
        return cls(config)
    
    @classmethod
    def create_selection_only(cls, model_selection_method: str = "zscore", **selection_params) -> 'EnsembleFramework':
        """
        Factory method for model selection only (no output aggregation).
        
        Args:
            model_selection_method: Model selection method
            **selection_params: Parameters for model selection
            
        Returns:
            Configured EnsembleFramework
        """
        config = EnsembleConfig(
            use_model_selection=True,
            model_selection_method=model_selection_method,
            model_selection_params=selection_params,
            use_output_aggregation=False
        )
        return cls(config)
    
    @classmethod
    def create_aggregation_only(cls, aggregation_method: str = "reward_based", aggregation_level: str = "sentence", **aggregation_params) -> 'EnsembleFramework':
        """
        Factory method for output aggregation only (use all models).
        
        Args:
            aggregation_method: Output aggregation method
            aggregation_level: Aggregation level (sentence/token/response)
            **aggregation_params: Parameters for output aggregation
            
        Returns:
            Configured EnsembleFramework
        """
        config = EnsembleConfig(
            use_model_selection=False,
            use_output_aggregation=True,
            aggregation_method=aggregation_method,
            aggregation_level=aggregation_level,
            aggregation_params=aggregation_params
        )
        return cls(config)


# Convenience functions for backward compatibility and easy usage
def run_simple_ensemble(
    example: Dict[str, Any],
    model_specs: List[Dict[str, Any]],
    generators,
    scorers,
    ensemble_method: str = "reward_based",
    model_selection_method: str = "all",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for simple ensemble usage.
    """
    framework = EnsembleFramework.create_simple_ensemble(ensemble_method, model_selection_method)
    return framework.run_ensemble(example, model_specs, generators, scorers, **kwargs)


def run_selection_only(
    example: Dict[str, Any],
    model_specs: List[Dict[str, Any]],
    generators,
    scorers,
    model_selection_method: str = "zscore",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for model selection only.
    """
    framework = EnsembleFramework.create_selection_only(model_selection_method, **kwargs)
    return framework.run_ensemble(example, model_specs, generators, scorers, **kwargs)


def run_aggregation_only(
    example: Dict[str, Any],
    model_specs: List[Dict[str, Any]],
    generators,
    scorers,
    aggregation_method: str = "reward_based",
    aggregation_level: str = "sentence",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for output aggregation only.
    """
    framework = EnsembleFramework.create_aggregation_only(aggregation_method, aggregation_level, **kwargs)
    return framework.run_ensemble(example, model_specs, generators, scorers, **kwargs)