"""
Argument parser for Ensemble-Hub, inspired by LlamaFactory.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml
from transformers import HfArgumentParser

from .ensemble_args import EnsembleArguments
from .generator_args import GeneratorArguments
from .method_args import MethodArguments


_ENSEMBLE_ARGS = [EnsembleArguments, MethodArguments, GeneratorArguments]
_ENSEMBLE_CLS = Tuple[EnsembleArguments, MethodArguments, GeneratorArguments]


def read_args(args: Optional[Union[Dict[str, Any], list[str]]] = None) -> Union[Dict[str, Any], list[str]]:
    """Get arguments from the command line or a config file."""
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(yaml_path).absolute()
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    if args is not None:
        # Check if args is a list with a YAML file
        if isinstance(args, list) and len(args) > 0 and (args[0].endswith(".yaml") or args[0].endswith(".yml")):
            return load_yaml(args[0])
        return args
    
    # Check command line arguments for YAML file
    if len(sys.argv) > 1 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return load_yaml(sys.argv[1])
    
    # No YAML file, use command line arguments
    return sys.argv[1:]


def _parse_args(parser: HfArgumentParser, args: Optional[Union[Dict[str, Any], list[str]]] = None) -> _ENSEMBLE_CLS:
    """Parse arguments using HfArgumentParser."""
    args = read_args(args)
    
    if isinstance(args, dict):
        # Parse from dictionary (YAML config)
        # Flatten nested config for HfArgumentParser
        flat_args = {}
        
        # Extract server configuration
        server_section = args.get('server', {})
        flat_args['api_host'] = server_section.get('host', '0.0.0.0')
        flat_args['api_port'] = server_section.get('port', 8000)
        
        # Extract ensemble section
        ensemble_section = args.get('ensemble', {})
        flat_args['max_rounds'] = ensemble_section.get('max_rounds', 500)
        flat_args['model_specs'] = args.get('model_specs', [])
        
        # Extract debug section  
        debug_section = args.get('debug', {})
        flat_args['show_output_details'] = debug_section.get('show_output_details', False)
        flat_args['show_input_details'] = debug_section.get('show_input_details', False)
        flat_args['enable_thinking'] = debug_section.get('enable_thinking', False)
        flat_args['save_results'] = debug_section.get('save_results', False)
        
        # Extract method parameters
        flat_args['model_selection_method'] = ensemble_section.get('model_selection_method', 'all')
        flat_args['model_selection_params'] = ensemble_section.get('model_selection_params', {})
        flat_args['output_aggregation_method'] = ensemble_section.get('output_aggregation_method', 'loop')
        flat_args['output_aggregation_params'] = ensemble_section.get('output_aggregation_params', {})
        
        # Extract generation parameters
        generation_section = ensemble_section.get('generation', {})
        flat_args['max_tokens'] = generation_section.get('max_tokens')
        flat_args['temperature'] = generation_section.get('temperature', 1.0)
        flat_args['top_p'] = generation_section.get('top_p', 1.0)
        flat_args['top_k'] = generation_section.get('top_k', 50)
        flat_args['repetition_penalty'] = generation_section.get('repetition_penalty', 1.0)
        stop_strings = generation_section.get('stop_strings', [])
        if isinstance(stop_strings, str):
            stop_strings = [stop_strings]
        flat_args['stop_strings'] = stop_strings

        extract_after = generation_section.get('extract_after', [])
        if isinstance(extract_after, str):
            extract_after = [extract_after]
        elif extract_after is None:
            extract_after = []
        flat_args['extract_after'] = extract_after
        flat_args['seed'] = generation_section.get('seed')
        
        # Extract generator parameters
        engine_options = args.get('engine_options', {})
        hf_options = engine_options.get('hf', {})
        flat_args['hf_use_eager_attention'] = hf_options.get('use_eager_attention', True)
        flat_args['hf_low_cpu_mem'] = hf_options.get('low_cpu_mem', True)
        
        return parser.parse_dict(flat_args, allow_extra_keys=True)  # type: ignore
    else:
        # Parse from command line
        parsed = parser.parse_args_into_dataclasses(args=args)
        return parsed  # type: ignore


def get_ensemble_args(args: Optional[Union[Dict[str, Any], list[str]]] = None) -> _ENSEMBLE_CLS:
    """Parse all ensemble-related arguments."""
    parser = HfArgumentParser(_ENSEMBLE_ARGS)
    return _parse_args(parser, args)
