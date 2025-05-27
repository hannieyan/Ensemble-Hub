# Generators Module

This module contains all generator implementations for different inference backends.

## Structure

- `base.py` - Base generator abstract class and common types
- `hf.py` - HuggingFace Transformers generator implementation
- `vllm.py` - vLLM generator implementation  
- `vllm_ray.py` - Ray-based vLLM generator for multi-GPU support
- `pool.py` - GeneratorPool for caching and managing generator instances

## Usage

```python
# Import individual generators
from ensemblehub.generators import HFGenerator, VLLMGenerator

# Import the generator pool
from ensemblehub.generators import GeneratorPool

# Get a generator instance (cached)
generator = GeneratorPool.get_generator(
    path="model_name",
    engine="hf",  # or "vllm", "vllm_ray"
    device="cuda:0",
    quantization="none"  # or "8bit", "4bit"
)

# Generate text
output = generator.generate(
    {"instruction": "...", "input": "...", "output": ""},
    max_tokens=256,
    temperature=0.95,
    seed=42
)
```

## Migration from Old Import

The generator module has been reorganized. Please update your imports:

**Old:**
```python
from ensemblehub.generator import GeneratorPool, BaseGenerator, HFGenerator
```

**New:**
```python
from ensemblehub.generators import GeneratorPool, BaseGenerator, HFGenerator
```

The old `ensemblehub.generator` module still exists for backward compatibility but will show a deprecation warning.