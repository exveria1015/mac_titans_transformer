"""
Example script to demonstrate the use of the native PyTorch implementation of Titans.
"""

import torch

from modeling_titans_mac import TitansMACConfig, TitansMACForCausalLM


def main():
    # Create a small model for testing
    config = TitansMACConfig(
        # Model architecture parameters
        num_tokens=10000,  # Smaller vocabulary
        d_model=256,  # Smaller embedding dimension
        depth=2,  # Minimal layers
        num_heads=4,  # Fewer attention heads
        dim_head=64,
        segment_len=32,  # Smaller segments
        num_longterm_mem_tokens=4,
        persistent_size=8,
        neural_memory_segment_len=36,  # segment_len + num_longterm_mem_tokens
        neural_mem_gate_attn_output=True,
        neural_memory_qkv_receives_diff_views=False,  # Disable different views for simplicity
        num_residual_streams=1,  # Single stream for simplicity
        sliding_window_attn=False,  # Disable sliding window for simplicity
        context_window=256,  # Match with d_model to avoid dimension mismatch
        # Generation parameters
        max_length=100,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

    # Create model
    model = TitansMACForCausalLM(config)

    # Print model info
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create a sample input with a small length to test
    input_ids = torch.randint(0, config.num_tokens, (1, 16))  # Shorter than segment_len

    # Forward pass
    outputs = model(input_ids=input_ids)

    # Print output shape
    print(f"Output logits shape: {outputs.logits.shape}")

    # Custom generation function
    def custom_generate(model, input_ids, max_length=20):
        # Start with the input sequence
        current_ids = input_ids.clone()

        # Generate tokens one by one
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            with torch.no_grad():
                outputs = model(current_ids, use_cache=True)

            # If logits is None, continue to the next token
            if outputs.logits is None:
                continue

            # Get the next token probabilities
            next_token_logits = outputs.logits[:, -1, :]

            # Sample from the distribution
            probs = torch.softmax(next_token_logits / 0.7, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids

    # Generate some text using our custom function
    generated = custom_generate(model, input_ids, max_length=20)

    print(f"Generated sequence shape: {generated.shape}")

    print("Model test successful!")


if __name__ == "__main__":
    main()
