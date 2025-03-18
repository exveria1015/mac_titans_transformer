# Titans: Learning to Memorize at Test Time

[![arXiv](https://img.shields.io/badge/arXiv-2501.00663-b31b1b.svg)](https://arxiv.org/abs/2501.00663)

*[日本語版はこちら](README.ja.md)*

This repository contains an unofficial PyTorch implementation of the paper [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) by Ali Behrouz, Peilin Zhong, and Vahab Mirrokni.

**Disclaimer**: This is an independent reproduction effort and is not affiliated with or endorsed by the original authors. The implementation is based solely on the descriptions provided in the paper and may not fully capture all aspects of the original work.

## Overview

The following overview is based on the descriptions from the original paper. As this implementation has not been fully tested, the actual behavior may differ from what is described below.

Titans is a novel neural architecture that combines the strengths of attention mechanisms and recurrent models. It introduces a neural long-term memory module that learns to memorize historical context, enabling the model to:

- Attend to the current context while utilizing long past information
- Scale to context windows larger than 2M tokens
- Maintain fast parallelizable training and efficient inference

The architecture addresses the limitations of traditional approaches:
- **Recurrent models**: Compress data into a fixed-size memory (hidden state)
- **Attention mechanisms**: Allow attending to the entire context window but with quadratic cost

Titans introduces a dual memory system:
- **Short-term memory**: Attention with limited context but accurate dependency modeling
- **Long-term memory**: Neural memory with the ability to memorize data for persistent storage

## Implementation Status

This repository provides a PyTorch implementation attempt of the Titans architecture, focusing on the MAC (Memory-Augmented Context) variant described in the paper. The implementation is a work in progress and may not fully replicate all the details or performance characteristics of the original model.

The repository includes:

- `modeling_titans_mac.py`: Implementation of the Titans MAC model based on the paper description
- `configuration_titans_mac.py`: Configuration classes for the model
- `example.py`: Basic example script for testing the model implementation

**Note**: This implementation is experimental and has not been validated against the original authors' implementation or results. Users should verify the behavior and performance of this code for their specific use cases.

## Requirements

```
torch==2.6.0+cu124
transformers==4.49.0
```

## Usage

The following example demonstrates how to use the current implementation. Please note that this is a basic example and the model's behavior may not fully match the paper's description:

```python
import torch
from modeling_titans_mac import TitansMACConfig, TitansMACForCausalLM

# Create a configuration
config = TitansMACConfig(
    num_tokens=10000,
    d_model=256,
    depth=2,
    num_heads=4,
    dim_head=64,
    segment_len=32,
    num_longterm_mem_tokens=4,
    persistent_size=8,
    neural_memory_segment_len=36,
    context_window=256,
)

# Initialize the model
model = TitansMACForCausalLM(config)

# Create input
input_ids = torch.randint(0, config.num_tokens, (1, 16))

# Forward pass
outputs = model(input_ids=input_ids)

# Generate text
# See example.py for a custom generation function
```

For more details on the implementation and usage, please refer to `example.py`. As this is an experimental implementation, you may need to adjust parameters or modify the code to suit your specific needs.

## Citation

If you find this implementation useful, please cite the original paper:

```bibtex
@misc{behrouz2024titans,
      title={Titans: Learning to Memorize at Test Time},
      author={Ali Behrouz and Peilin Zhong and Vahab Mirrokni},
      year={2024},
      eprint={2501.00663},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Disclaimer

This is an independent project created out of academic interest in the Titans architecture. The implementation is based on my understanding of the paper and may differ from the authors' original implementation. I am not affiliated with the original authors, and this work has not been reviewed by them.

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.
