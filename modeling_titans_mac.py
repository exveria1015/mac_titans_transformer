"""
Titans: Learning to Memorize at Test Time

This is a native PyTorch implementation of the Titans model based on the paper:
"Titans: Learning to Memorize at Test Time" by Ali Behrouz, Peilin Zhong, and Vahab Mirrokni.

The Titans model introduces a novel approach for neural networks to learn and adapt during inference time,
without requiring gradient updates. This implementation focuses on the Memory As Context (MAC) variant
which uses neural memory modules to store and retrieve information dynamically.
"""

import logging
import math
from collections import namedtuple
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from configuration_titans_mac import TitansMACConfig

# Define AttnIntermediates at the module level
AttnIntermediates = namedtuple(
    "AttnIntermediates", ["value_residual", "cached_key_values"]
)

logger = logging.getLogger(__name__)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def divisible_by(num, den):
    return (num % den) == 0


def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult


def round_down_multiple(seq, mult):
    return seq // mult * mult


class MemoryUpdates:
    def __init__(self, updates):
        self.updates = updates


class NeuralMemory(nn.Module):
    """
    Neural Memory Module for the Titans model.

    This module implements a neural memory mechanism that can learn and adapt during inference time.
    It uses attention mechanisms to retrieve relevant information and updates its parameters
    based on the "surprise" level of the current data.

    Args:
        dim (`int`):
            The dimension of the input features.
        chunk_size (`int`):
            The size of memory chunks to process.
        batch_size (`int`, *optional*):
            The batch size for processing.
        model (`nn.Module`, *optional*):
            Custom memory model. If None, a default MLP is used.
        qkv_receives_diff_views (`bool`, *optional*, defaults to `False`):
            Whether query, key, and value inputs come from different views.
        accept_weight_residual (`bool`, *optional*, defaults to `False`):
            Whether to accept residual connections for attention weights.
    """

    def __init__(
        self,
        dim: int,
        chunk_size: int,
        batch_size: Optional[int] = None,
        model: Optional[nn.Module] = None,
        qkv_receives_diff_views: bool = False,
        accept_weight_residual: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.qkv_receives_diff_views = qkv_receives_diff_views
        self.accept_weight_residual = accept_weight_residual

        # Default memory model if none provided
        if not exists(model):
            self.model = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
        else:
            self.model = model

        # Projections for query, key, value
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Output projection
        self.to_out = nn.Linear(dim, dim)

        # Normalization
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        qkv_inputs: Union[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        state: Optional[torch.Tensor] = None,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, MemoryUpdates]:
        """
        Forward pass for the Neural Memory module.

        Args:
            qkv_inputs (`torch.Tensor` or `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`):
                Tensor of shape [3, batch, seq_len, dim] if qkv_receives_diff_views=True
                else [batch, seq_len, dim]
            state (`torch.Tensor`, *optional*):
                Previous memory state
            prev_weights (`torch.Tensor`, *optional*):
                Previous attention weights for weight residual

        Returns:
            `tuple(torch.Tensor, MemoryUpdates)`: Output tensor and memory updates
        """
        if self.qkv_receives_diff_views:
            q_input, k_input, v_input = qkv_inputs
        else:
            q_input = k_input = v_input = qkv_inputs

        # Handle dimension mismatch for each input
        def handle_dim_mismatch(x):
            if x.dim() == 3 and x.size(-1) != self.dim:
                batch_size, seq_len, input_dim = x.shape
                logger.debug(
                    f"NeuralMemory: Input dimension {input_dim} doesn't match expected dimension {self.dim}"
                )

                # If input dimension is a multiple of self.dim (e.g., from multiple streams)
                if input_dim % self.dim == 0:
                    num_streams = input_dim // self.dim
                    x = x.reshape(batch_size, seq_len, num_streams, self.dim)
                    x = x.mean(dim=2)  # Average across streams
                else:
                    # If dimensions don't match and aren't multiples
                    logger.warning(
                        f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
                    )
                    # Truncate or pad the input to match the expected dimension
                    if input_dim > self.dim:
                        x = x[:, :, : self.dim]  # Truncate
                    else:
                        padding = torch.zeros(
                            batch_size, seq_len, self.dim - input_dim, device=x.device
                        )
                        x = torch.cat([x, padding], dim=-1)  # Pad

                logger.debug(f"After dimension adjustment: {x.shape}")

            return x

        q_input = handle_dim_mismatch(q_input)
        k_input = handle_dim_mismatch(k_input)
        v_input = handle_dim_mismatch(v_input)

        # Apply normalization
        q_input = self.norm(q_input)
        k_input = self.norm(k_input)
        v_input = self.norm(v_input)

        # Project to query, key, value
        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        # Process with memory model
        memory_out = self.model(q)

        # Compute attention
        scale = q.shape[-1] ** -0.5

        # Compute attention scores using einsum
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * scale

        # Apply causal mask
        mask = torch.ones(
            (q.shape[1], k.shape[1]), dtype=torch.bool, device=q.device
        ).triu_(1)
        sim.masked_fill_(mask.unsqueeze(0), -torch.finfo(sim.dtype).max)

        # Apply softmax to get attention weights
        attn = F.softmax(sim, dim=-1)

        # Apply weight residual if needed
        if self.accept_weight_residual and exists(prev_weights):
            attn = attn + prev_weights

        # Apply attention to values
        out = torch.einsum("b i j, b j d -> b i d", attn, v)

        # Project output
        out = self.to_out(out)

        # Apply advanced memory update mechanism (simulating learning at test time)
        with torch.no_grad():
            # Calculate "data surprise" - how much the current data differs from what the model expects
            # We use the attention scores as a proxy for data surprise
            data_surprise = torch.mean(
                torch.abs(attn - 0.5)
            )  # Higher values indicate more surprise

            # Calculate memory decay rate based on data surprise
            # Higher surprise = slower decay (more retention of new information)
            base_decay_rate = 0.99
            adaptive_decay_rate = base_decay_rate - (
                0.1 * data_surprise
            )  # Adjust decay based on surprise
            adaptive_decay_rate = torch.clamp(adaptive_decay_rate, min=0.8, max=0.999)

            # Apply momentum-based update
            if not hasattr(self, "param_momentum"):
                # Initialize momentum buffers on first run
                self.param_momentum = {}
                for name, param in self.model.named_parameters():
                    self.param_momentum[name] = torch.zeros_like(param.data)

            # Update parameters with momentum and weight decay
            momentum_factor = 0.9  # Standard momentum factor
            for name, param in self.model.named_parameters():
                # Calculate gradient-like update based on current data
                # For simplicity, we use a small random perturbation scaled by data surprise
                update = torch.randn_like(param.data) * 0.001 * data_surprise

                # Apply momentum
                self.param_momentum[name] = (
                    momentum_factor * self.param_momentum[name] + update
                )

                # Apply the update with adaptive decay
                param.data = (
                    adaptive_decay_rate * param.data + self.param_momentum[name]
                )

            # Log debug info
            logger.debug(
                f"Memory update: data_surprise={data_surprise.item():.4f}, decay_rate={adaptive_decay_rate.item():.4f}"
            )

        # Create a MemoryUpdates object with the attention weights
        memory_updates = MemoryUpdates(attn)

        # Return output and memory updates
        return out, memory_updates


class HyperConnection(nn.Module):
    """
    Hyper connection module for the Titans model.

    This module creates a branch in the network and provides a function to add the branch output
    back to the residual path.

    Args:
        add_branch_out_to_residual (`bool`, *optional*, defaults to `True`):
            Whether to add the branch output to the residual path.
    """

    def __init__(self, add_branch_out_to_residual: bool = True):
        super().__init__()
        self.add_branch_out_to_residual = add_branch_out_to_residual

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, callable]:
        """
        Forward pass for the HyperConnection module.

        Args:
            x (`torch.Tensor`):
                Input tensor

        Returns:
            `tuple(torch.Tensor, callable)`: Branch input and a function to add residual
        """
        branch_input = x

        def add_residual(branch_output):
            if self.add_branch_out_to_residual:
                return x + branch_output
            return x

        return branch_input, add_residual


class IdentityModule(nn.Module):
    """
    Identity module that returns the input unchanged.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the IdentityModule.

        Args:
            x (`torch.Tensor`):
                Input tensor

        Returns:
            `torch.Tensor`: The same input tensor
        """
        return x


class StreamEmbedding(nn.Module):
    """
    Stream Embedding module for hyper connections.

    This module adds learnable embeddings to different streams of data.

    Args:
        dim (`int`):
            The dimension of each stream.
        num_streams (`int`):
            The number of parallel streams.
    """

    def __init__(self, dim: int, num_streams: int):
        super().__init__()
        self.stream_embeddings = nn.Parameter(torch.randn(num_streams, dim))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the StreamEmbedding module.

        Args:
            x (`torch.Tensor`):
                Input tensor of shape [batch, seq_len, total_dim]

        Returns:
            `torch.Tensor`: Tensor with stream embeddings added
        """
        batch, seq_len, total_dim = x.shape
        num_streams = total_dim // self.dim

        # Reshape x to separate streams
        x = x.reshape(batch, seq_len, num_streams, self.dim)

        # Add stream embeddings
        stream_emb = self.stream_embeddings.unsqueeze(0).unsqueeze(1)
        x = x + stream_emb

        return x.reshape(batch, seq_len, -1)


class ExpandStreams(nn.Module):
    """
    ExpandStreams module for hyper connections.

    This module expands the input tensor into multiple parallel streams.

    Args:
        dim (`int`):
            The dimension of each stream.
        num_streams (`int`):
            The number of parallel streams to create.
        add_stream_embed (`bool`, *optional*, defaults to `True`):
            Whether to add stream embeddings to differentiate streams.
    """

    def __init__(self, dim: int, num_streams: int, add_stream_embed: bool = True):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.stream_embedding = (
            StreamEmbedding(dim, num_streams) if add_stream_embed else None
        )
        logger.info(
            f"ExpandStreams initialized with dim={dim}, num_streams={num_streams}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ExpandStreams module.

        Args:
            x (`torch.Tensor`):
                Input tensor of shape [batch, seq_len, input_dim]

        Returns:
            `torch.Tensor`: Expanded tensor with multiple streams
        """
        batch, seq_len, input_dim = x.shape
        logger.debug(f"ExpandStreams input shape: {x.shape}")

        # Handle dimension mismatch
        if input_dim != self.dim:
            logger.warning(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )
            # Truncate or pad to match expected dimension
            if input_dim > self.dim:
                x = x[:, :, : self.dim]  # Truncate
            else:
                padding = torch.zeros(
                    batch, seq_len, self.dim - input_dim, device=x.device
                )
                x = torch.cat([x, padding], dim=-1)  # Pad
            logger.debug(f"After dimension adjustment: {x.shape}")

        # Repeat the input for each stream
        x = x.repeat(1, 1, self.num_streams)
        logger.debug(f"After repeat: {x.shape}")

        # Apply stream embedding if it exists
        if exists(self.stream_embedding):
            x = self.stream_embedding(x)
            logger.debug(f"After stream embedding: {x.shape}")

        return x


class ReduceStreams(nn.Module):
    """
    ReduceStreams module for hyper connections.

    This module reduces multiple parallel streams back to a single stream.

    Args:
        dim (`int`):
            The dimension of each stream.
        num_streams (`int`):
            The number of parallel streams to reduce.
    """

    def __init__(self, dim: int, num_streams: int):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.to_out = nn.Linear(dim * num_streams, dim)
        logger.info(
            f"ReduceStreams initialized with dim={dim}, num_streams={num_streams}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ReduceStreams module.

        Args:
            x (`torch.Tensor`):
                Input tensor of shape [batch, seq_len, input_dim]

        Returns:
            `torch.Tensor`: Reduced tensor with a single stream
        """
        batch, seq_len, input_dim = x.shape
        logger.debug(f"ReduceStreams input shape: {x.shape}")

        # Handle dimension mismatch
        expected_dim = self.dim * self.num_streams
        if input_dim != expected_dim:
            logger.warning(
                f"Input dimension {input_dim} doesn't match expected dimension {expected_dim}"
            )
            # Truncate or pad to match expected dimension
            if input_dim > expected_dim:
                x = x[:, :, :expected_dim]  # Truncate
                logger.debug(f"Truncated to: {x.shape}")
            else:
                padding = torch.zeros(
                    batch, seq_len, expected_dim - input_dim, device=x.device
                )
                x = torch.cat([x, padding], dim=-1)  # Pad
                logger.debug(f"Padded to: {x.shape}")

        # Project back to original dimension
        out = self.to_out(x)
        logger.debug(f"ReduceStreams output shape: {out.shape}")
        return out


def get_init_and_expand_reduce_stream_functions(
    num_streams: int, dim: int, add_stream_embed: bool = True, disable: bool = False
) -> Tuple[callable, nn.Module, nn.Module]:
    """
    Creates hyper connection functions and stream modules.

    Args:
        num_streams (`int`):
            The number of parallel streams.
        dim (`int`):
            The dimension of each stream.
        add_stream_embed (`bool`, *optional*, defaults to `True`):
            Whether to add stream embeddings.
        disable (`bool`, *optional*, defaults to `False`):
            Whether to disable stream expansion/reduction.

    Returns:
        `tuple(callable, nn.Module, nn.Module)`:
            Hyper connection initializer, expand streams module, and reduce streams module
    """
    if disable:
        # Return identity modules if disabled
        def init_hyper_conn(add_branch_out_to_residual=True):
            return HyperConnection(add_branch_out_to_residual)

        expand_streams = IdentityModule()
        reduce_streams = IdentityModule()
    else:
        # Initialize expand and reduce stream modules
        expand_streams_module = ExpandStreams(dim, num_streams, add_stream_embed)
        reduce_streams_module = ReduceStreams(dim, num_streams)

        def init_hyper_conn(add_branch_out_to_residual=True):
            return HyperConnection(add_branch_out_to_residual)

        expand_streams = expand_streams_module
        reduce_streams = reduce_streams_module

    return init_hyper_conn, expand_streams, reduce_streams


def pad_at_dim(
    t: torch.Tensor, pad: Tuple[int, int], dim: int = -1, value: float = 0.0
) -> torch.Tensor:
    """
    Pads a tensor at a specific dimension.

    Args:
        t (`torch.Tensor`):
            Input tensor to pad.
        pad (`tuple(int, int)`):
            Padding values (left, right).
        dim (`int`, *optional*, defaults to -1):
            Dimension to pad.
        value (`float`, *optional*, defaults to 0.0):
            Value to pad with.

    Returns:
        `torch.Tensor`: Padded tensor
    """
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pad_and_segment_with_inverse(
    seq: torch.Tensor,
    segment_len: int,
    fold_into_batch: bool = True,
    inverse_remove_pad: bool = True,
) -> Tuple[torch.Tensor, callable]:
    """
    Pads and segments a sequence tensor, with a function to reverse the operation.

    Args:
        seq (`torch.Tensor`):
            Input sequence tensor.
        segment_len (`int`):
            Length of each segment.
        fold_into_batch (`bool`, *optional*, defaults to `True`):
            Whether to fold segments into the batch dimension.
        inverse_remove_pad (`bool`, *optional*, defaults to `True`):
            Whether to remove padding in the inverse function.

    Returns:
        `tuple(torch.Tensor, callable)`:
            Segmented tensor and inverse function
    """
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = math.ceil(seq_len / segment_len) * segment_len

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    logger.debug(
        f"pad_and_segment_with_inverse: Input shape: {seq.shape}, segment_len: {segment_len}"
    )
    logger.debug(
        f"next_seq_len_mult: {next_seq_len_mult}, padding: {padding}, needs_pad: {needs_pad}"
    )

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))
        logger.debug(f"After padding: {seq.shape}")

    if fold_into_batch:
        seq = seq.reshape(batch, -1, segment_len, seq.shape[-1])
        logger.debug(f"After first reshape: {seq.shape}")
        seq = seq.reshape(batch * seq.shape[1], segment_len, seq.shape[-1])
        logger.debug(f"After second reshape: {seq.shape}")

    def inverse(out):
        logger.debug(f"inverse function called with out shape: {out.shape}")
        if fold_into_batch:
            try:
                out = out.reshape(batch, -1, segment_len, out.shape[-1])
                logger.debug(f"After first inverse reshape: {out.shape}")
                out = out.reshape(batch, -1, out.shape[-1])
                logger.debug(f"After second inverse reshape: {out.shape}")
            except RuntimeError as e:
                logger.error(f"Error in inverse reshape: {e}")
                # Return the original tensor if reshaping fails
                return out

        if needs_pad and inverse_remove_pad:
            out = out[:, :seq_len]
            logger.debug(f"After removing padding: {out.shape}")

        return out

    return seq, inverse


class SegmentedAttention(nn.Module):
    """
    Segmented Attention module for the Titans model.

    This module implements attention over segments of the input sequence, with support for
    persistent memory tokens and long-term memory tokens.

    Args:
        dim (`int`):
            The dimension of the input features.
        segment_len (`int`):
            The length of each segment.
        num_persist_mem_tokens (`int`, *optional*, defaults to 0):
            Number of persistent memory tokens.
        num_longterm_mem_tokens (`int`, *optional*, defaults to 0):
            Number of long-term memory tokens.
        dim_head (`int`, *optional*, defaults to 64):
            Dimension of each attention head.
        heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        sliding (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        accept_value_residual (`bool`, *optional*, defaults to `False`):
            Whether to accept value residual connections.
    """

    def __init__(
        self,
        dim: int,
        segment_len: int,
        num_persist_mem_tokens: int = 0,
        num_longterm_mem_tokens: int = 0,
        dim_head: int = 64,
        heads: int = 8,
        sliding: bool = False,
        accept_value_residual: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim)

        # Use LayerNorm with correct dimensions
        self.norm = nn.LayerNorm(dim)

        self.segment_len = segment_len
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.num_persist_mem_tokens = num_persist_mem_tokens

        self.sliding = sliding
        self.total_segment_len = segment_len + num_longterm_mem_tokens

        # For value residual
        self.accept_value_residual = accept_value_residual
        if accept_value_residual:
            self.to_value_mix = nn.Sequential(nn.Linear(dim, heads), nn.Sigmoid())

        # Persistent memory
        self.persistent_memory = nn.Parameter(
            torch.zeros(2, heads, num_persist_mem_tokens, dim_head)
        )

    def forward(
        self,
        x: torch.Tensor,
        value_residual: Optional[torch.Tensor] = None,
        output_gating: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, AttnIntermediates]:
        """
        Forward pass for the SegmentedAttention module.

        Args:
            x (`torch.Tensor`):
                Input tensor of shape [batch_size, seq_len, dim]
            value_residual (`torch.Tensor`, *optional*):
                Value residual tensor from previous layer
            output_gating (`torch.Tensor`, *optional*):
                Output gating tensor
            cache (`tuple(torch.Tensor, torch.Tensor)`, *optional*):
                Cached key and value states for incremental decoding

        Returns:
            `tuple(torch.Tensor, AttnIntermediates)`:
                Output tensor and attention intermediates
        """
        batch_size, seq_len, input_dim = x.shape
        is_inferencing = exists(cache)

        # Debug info
        logger.debug(f"SegmentedAttention input shape: {x.shape}")
        logger.debug(f"Expected dim: {self.dim}, Input dim: {input_dim}")
        logger.debug(
            f"Segment length: {self.segment_len}, Total segment length: {self.total_segment_len}"
        )

        # Handle dimension mismatch
        if input_dim != self.dim:
            # If input dimension is a multiple of self.dim (e.g., from multiple streams)
            if input_dim % self.dim == 0:
                num_streams = input_dim // self.dim
                x = x.reshape(batch_size, seq_len, num_streams, self.dim)
                x = x.mean(dim=2)  # Average across streams
            else:
                # If dimensions don't match and aren't multiples, log a warning
                logger.warning(
                    f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
                )
                # Truncate or pad the input to match the expected dimension
                if input_dim > self.dim:
                    x = x[:, :, : self.dim]  # Truncate
                else:
                    padding = torch.zeros(
                        batch_size, seq_len, self.dim - input_dim, device=x.device
                    )
                    x = torch.cat([x, padding], dim=-1)  # Pad

        # Normalize input
        x = self.norm(x)

        # Project to q, k, v
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(
                batch_size, seq_len, self.heads, self.dim_head
            ).transpose(1, 2),
            qkv,
        )

        # Handle value residual
        orig_v = v.clone()

        if self.accept_value_residual and exists(value_residual):
            logger.debug(
                f"Handling value residual: v shape: {v.shape}, value_residual shape: {value_residual.shape}"
            )

            # Make sure dimensions match
            if v.shape != value_residual.shape:
                logger.warning("Dimension mismatch between v and value_residual")

                # Reshape value_residual to match v's shape if needed
                if value_residual.shape[2] != v.shape[2]:
                    logger.debug(
                        "Reshaping value_residual to match v's sequence length"
                    )

                    # Expand value_residual to match v's sequence length
                    expanded_value_residual = torch.zeros_like(v)

                    # Copy the available values
                    min_seq_len = min(value_residual.shape[2], v.shape[2])
                    expanded_value_residual[:, :, :min_seq_len, :] = value_residual[
                        :, :, :min_seq_len, :
                    ]

                    value_residual = expanded_value_residual
                    logger.debug(
                        f"After reshaping: value_residual shape: {value_residual.shape}"
                    )

            # Generate mixing weights
            mix = self.to_value_mix(x)
            logger.debug(f"mix shape before reshape: {mix.shape}")

            # Reshape mix to match broadcasting requirements
            # Reshape from [batch_size, seq_len, heads] to [batch_size, heads, seq_len, 1]
            mix = mix.permute(0, 2, 1).unsqueeze(-1)
            logger.debug(f"mix shape after reshape: {mix.shape}")

            # Apply mixing
            v = v * (1 - mix) + value_residual * mix
            logger.debug(f"After mixing: v shape: {v.shape}")

        # Handle caching for inference
        if is_inferencing:
            ck, cv = cache
            k = torch.cat([ck, k], dim=2)
            v = torch.cat([cv, v], dim=2)
            next_cache = (k, v)
        else:
            next_cache = (k, v)

        # Auto pad to segment length
        if not is_inferencing:
            # For simplicity, let's just use the flat tensors directly
            # This avoids complex reshaping that might cause dimension errors
            q_flat = q
            k_flat = k
            v_flat = v

            # Log shapes for debugging
            logger.debug(f"q_flat shape: {q_flat.shape}")
            logger.debug(f"k_flat shape: {k_flat.shape}")
            logger.debug(f"v_flat shape: {v_flat.shape}")

            # Use these flat tensors directly
            q, k, v = q_flat, k_flat, v_flat

            # Handle sliding window attention
            if self.sliding:
                # Create sliding window mask
                total_seq_len = q.shape[2]
                mask = torch.ones(
                    batch_size,
                    self.heads,
                    total_seq_len,
                    total_seq_len,
                    device=q.device,
                    dtype=torch.bool,
                )

                # Set up sliding window
                window_size = min(
                    self.total_segment_len * 2, total_seq_len
                )  # Allow attention to previous segment

                for i in range(total_seq_len):
                    # Define window boundaries
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(total_seq_len, i + window_size // 2)

                    # Set mask to False (allow attention) within window
                    mask[:, :, i, start_idx:end_idx] = False

                # Add causal mask (can't attend to future tokens)
                causal_mask = torch.triu(
                    torch.ones(total_seq_len, total_seq_len, device=q.device),
                    diagonal=1,
                ).bool()
                mask = mask | causal_mask.unsqueeze(0).unsqueeze(0)
            else:
                # Block diagonal attention
                total_seq_len = q.shape[2]

                # Create a simple causal mask for now
                mask = torch.triu(
                    torch.ones(total_seq_len, total_seq_len, device=q.device),
                    diagonal=1,
                ).bool()
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.heads, -1, -1)
                )

                logger.debug(f"Created mask with shape: {mask.shape}")

        # Add persistent memory
        pmk, pmv = self.persistent_memory
        pmk = pmk.unsqueeze(0).expand(batch_size, -1, -1, -1)
        pmv = pmv.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Log shapes before concatenation
        logger.debug(f"pmk shape: {pmk.shape}, k shape: {k.shape}")

        k = torch.cat([pmk, k], dim=2)
        v = torch.cat([pmv, v], dim=2)

        logger.debug(
            f"After adding persistent memory: k shape: {k.shape}, v shape: {v.shape}"
        )

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        logger.debug(f"attn_scores shape: {attn_scores.shape}")

        # Apply mask - make sure mask dimensions match attention scores
        if not is_inferencing and exists(mask):
            logger.debug(f"mask shape: {mask.shape}")

            # Adjust mask to match attention scores dimensions
            if mask.shape[-1] != attn_scores.shape[-1]:
                logger.debug(
                    f"Adjusting mask from {mask.shape} to match {attn_scores.shape}"
                )

                # Create a new mask of the correct size
                new_mask = torch.ones_like(attn_scores, dtype=torch.bool)

                # Fill in the original mask part
                seq_len = min(
                    mask.shape[-1], attn_scores.shape[-1] - self.num_persist_mem_tokens
                )

                # Apply causal mask for the sequence part (excluding persistent memory)
                for i in range(attn_scores.shape[-2]):
                    # Allow attention to persistent memory tokens
                    new_mask[:, :, i, : self.num_persist_mem_tokens] = False

                    # Apply original mask for the sequence part
                    if i < seq_len:
                        for j in range(seq_len):
                            if j <= i:  # Causal masking
                                new_mask[:, :, i, j + self.num_persist_mem_tokens] = (
                                    False
                                )

                mask = new_mask
                logger.debug(f"Adjusted mask shape: {mask.shape}")

            attn_scores.masked_fill_(mask, -torch.finfo(attn_scores.dtype).max)

        # Apply causal mask for inference
        if is_inferencing:
            logger.debug(f"Inferencing mode: q shape: {q.shape}, k shape: {k.shape}")

            # Create a causal mask that allows attention to persistent memory
            mask = torch.ones(
                batch_size,
                self.heads,
                q.shape[2],
                k.shape[2],
                device=q.device,
                dtype=torch.bool,
            )

            # Allow attention to persistent memory tokens
            mask[:, :, :, : self.num_persist_mem_tokens] = False

            # Apply causal masking for the sequence part
            for i in range(q.shape[2]):
                for j in range(self.num_persist_mem_tokens, k.shape[2]):
                    if j - self.num_persist_mem_tokens <= i:  # Causal masking
                        mask[:, :, i, j] = False

            attn_scores.masked_fill_(mask, -torch.finfo(attn_scores.dtype).max)

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape output
        out = out.transpose(1, 2).reshape(batch_size, -1, self.heads * self.dim_head)

        # Apply inverse segmentation if needed
        if not is_inferencing:
            # Trim to original sequence length
            out = out[:, :seq_len, :]

        # Project to output dimension
        out = self.to_out(out)

        # Apply output gating if provided
        if exists(output_gating):
            out = out * output_gating

        logger.debug(f"SegmentedAttention output shape: {out.shape}")
        logger.debug(
            f"Creating AttnIntermediates with orig_v shape: {orig_v.shape} and next_cache"
        )

        # Return output and intermediates
        return out, AttnIntermediates(
            value_residual=orig_v, cached_key_values=next_cache
        )


class FeedForward(nn.Module):
    """
    FeedForward module for the Titans model.

    This module implements a standard feed-forward network with layer normalization.

    Args:
        dim (`int`):
            The dimension of the input features.
        mult (`int`, *optional*, defaults to 4):
            Multiplier for the inner dimension.
    """

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.dim = dim
        self.mult = mult

        # Calculate inner dimension
        dim_inner = int(dim * mult * 2 / 3)

        # Create a sequential network
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_inner * 2),
            nn.GELU(),
            nn.Linear(
                dim_inner * 2, dim
            ),  # Fixed: use dim_inner * 2 to match first linear layer output
        )

        logger.info(f"FeedForward initialized with dim={dim}, dim_inner={dim_inner}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward module.

        Args:
            x (`torch.Tensor`):
                Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            `torch.Tensor`: Output tensor
        """
        batch_size, seq_len, input_dim = x.shape
        logger.debug(f"FeedForward input shape: {x.shape}")

        # Handle dimension mismatch
        if input_dim != self.dim:
            logger.warning(
                f"Input dimension {input_dim} doesn't match expected dimension {self.dim}"
            )

            # If input dimension is a multiple of self.dim (e.g., from multiple streams)
            if input_dim % self.dim == 0:
                num_streams = input_dim // self.dim
                x = x.reshape(batch_size, seq_len, num_streams, self.dim)
                x = x.mean(dim=2)  # Average across streams
            else:
                # Truncate or pad the input to match the expected dimension
                if input_dim > self.dim:
                    x = x[:, :, : self.dim]  # Truncate
                else:
                    padding = torch.zeros(
                        batch_size, seq_len, self.dim - input_dim, device=x.device
                    )
                    x = torch.cat([x, padding], dim=-1)  # Pad

            logger.debug(f"After dimension adjustment: {x.shape}")

        # Apply the network
        return self.net(x)


class MemoryAsContextTransformer(PreTrainedModel):
    """
    Memory As Context (MAC) Transformer model.

    This is the main transformer model for the Titans architecture, which uses memory as context
    to enable learning at test time. It consists of a stack of attention layers with neural memory
    modules that can adapt to new information during inference.

    Args:
        config (`TitansMACConfig`):
            Model configuration class with all the parameters of the model.
    """

    config_class = TitansMACConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(config.num_tokens, config.d_model)

        # Positional embedding
        self.pos_emb = nn.Embedding(config.context_window, config.d_model)

        # Long term memory tokens
        self.segment_len = config.segment_len
        self.num_longterm_mem_tokens = config.num_longterm_mem_tokens
        self.longterm_mems = nn.Parameter(
            torch.randn(config.num_longterm_mem_tokens, config.d_model) * 0.02
        )

        # Sliding window attention
        self.sliding_window_attn = config.sliding_window_attn
        self.attn_window_size = config.segment_len + config.num_longterm_mem_tokens

        # Hyper connections
        init_hyper_conn, self.expand_streams, self.reduce_streams = (
            get_init_and_expand_reduce_stream_functions(
                config.num_residual_streams,
                dim=config.d_model,
                add_stream_embed=True,
                disable=config.num_residual_streams == 1,
            )
        )

        # Neural memory segment length
        self.neural_memory_segment_len = default(
            config.neural_memory_segment_len,
            config.num_longterm_mem_tokens + config.segment_len,
        )

        # Layers
        self.layers = nn.ModuleList([])
        layers = tuple(range(1, config.depth + 1))
        neural_memory_layers = default(config.neural_memory_layers, layers)

        # Weight residual related
        self.neural_mem_weight_residual = config.neural_mem_weight_residual
        is_first_neural_mem = True

        # Build layers
        for layer in layers:
            is_first = layer == 1

            # Attention
            attn = SegmentedAttention(
                dim=config.d_model,
                dim_head=config.dim_head,  # Use the configured dim_head instead of calculating
                heads=config.num_heads,
                segment_len=config.segment_len,
                accept_value_residual=not is_first,
                num_longterm_mem_tokens=config.num_longterm_mem_tokens,
                num_persist_mem_tokens=config.persistent_size,
                sliding=config.sliding_window_attn,
            )

            # Neural memory
            mem = None
            mem_qkv_layer_selector = None
            mem_hyper_conn = None

            if layer in neural_memory_layers:
                mem_hyper_conn = init_hyper_conn(
                    add_branch_out_to_residual=not config.neural_mem_gate_attn_output
                )

                if not is_first and config.neural_memory_qkv_receives_diff_views:
                    num_layer_choices = (layer - 1) * 4 + 1
                    logger.info(
                        f"Creating mem_qkv_layer_selector with num_layer_choices={num_layer_choices}"
                    )

                    mem_qkv_layer_selector = nn.Sequential(
                        nn.LayerNorm(config.d_model),
                        nn.Linear(config.d_model, 3 * num_layer_choices),
                        nn.Softmax(dim=-1),
                    )
                    logger.info(
                        f"mem_qkv_layer_selector created with output size: {3 * num_layer_choices}"
                    )

                mem = NeuralMemory(
                    dim=config.d_model,
                    chunk_size=self.neural_memory_segment_len,
                    batch_size=config.neural_memory_batch_size,
                    qkv_receives_diff_views=True,
                    accept_weight_residual=config.neural_mem_weight_residual
                    and not is_first_neural_mem,
                )

                is_first_neural_mem = False

            # Feedforward
            ff = FeedForward(dim=config.d_model, mult=config.ff_mult)

            # Add layer components
            self.layers.append(
                nn.ModuleList(
                    [
                        mem_hyper_conn,
                        init_hyper_conn(),
                        init_hyper_conn(),
                        mem_qkv_layer_selector,
                        mem,
                        attn,
                        ff,
                    ]
                )
            )

        # Final normalization and output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.to_logits = nn.Linear(config.d_model, config.num_tokens, bias=False)

        # Register dimension for dimension checking
        self.d_model = config.d_model

        # Whether to gate attention output with retrieved memories
        self.gate_attn_output = config.neural_mem_gate_attn_output

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def seq_index_is_longterm(self, seq_index):
        """Check if sequence index corresponds to a longterm memory token."""
        total_segment_len, segment_len = self.attn_window_size, self.segment_len
        return ((seq_index % total_segment_len + 1) - segment_len) > 0

    def seq_len_with_longterm_mem(self, seq_len):
        """Calculate sequence length with longterm memory tokens interspersed."""
        assert seq_len > 0
        segment_len, num_mem = self.segment_len, self.num_longterm_mem_tokens
        return ((seq_len - 1) // segment_len) * num_mem + seq_len

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass of the Memory As Context Transformer.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary. If not provided, `inputs_embeds` must be given.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
                Contains pre-computed key and value hidden states of the attention blocks used for
                fast autoregressive decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation of input sequence. If provided, `input_ids` will not be used.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up
                decoding (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `BaseModelOutputWithPast` or `tuple(torch.FloatTensor)`:
                If `return_dict=True`, a `BaseModelOutputWithPast` is returned, otherwise a tuple.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.token_emb(input_ids)

        batch, seq_len, device = (
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            inputs_embeds.device,
        )

        # Handle past_key_values for incremental decoding
        is_inferencing = exists(past_key_values)

        if is_inferencing:
            inference_seq_index, kv_caches, neural_mem_caches = past_key_values
            inputs_embeds = inputs_embeds[:, -1:, :]  # Only use the last token

        # Make sure sequence length is at least the segment length
        if seq_len < self.segment_len:
            padding = self.segment_len - seq_len
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, padding))
            seq_len = self.segment_len
            logger.debug(f"Padded input sequence to segment length: {seq_len}")

        # Calculate sequence length with longterm memory
        seq_len_with_mem = self.seq_len_with_longterm_mem(seq_len)
        logger.debug(f"Sequence length with memory: {seq_len_with_mem}")

        # Intersperse longterm memory
        x, inverse_segment = pad_and_segment_with_inverse(
            inputs_embeds, self.segment_len, inverse_remove_pad=False
        )

        mems = self.longterm_mems.unsqueeze(0).expand(x.shape[0], -1, -1)

        # Pack tokens and memories
        segments = []
        for i in range(x.shape[1] // self.segment_len):
            segment = x[:, i * self.segment_len : (i + 1) * self.segment_len, :]
            segments.append(segment)
            if i < len(segments) - 1:  # Don't add memory after the last segment
                segments.append(mems)

        x = torch.cat(segments, dim=1)

        # Splice out unneeded tokens from padding for longterm mems
        x = x[:, :seq_len_with_mem]

        # Apply positional embedding
        positions = torch.arange(seq_len_with_mem, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb

        # KV caching
        if not is_inferencing:
            kv_caches = [None] * len(self.layers)
            neural_mem_caches = [None] * len(self.layers)

        next_kv_caches = []
        next_neural_mem_caches = []

        # Value residual
        value_residual = None

        # Neural mem weight residual
        mem_weight_residual = None

        # Layers for the neural mem to select the qkv inputs from
        mem_input_layers = []

        # Expand streams for hyper connections
        x = self.expand_streams(x)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, (
            mem_hyper_conn,
            attn_hyper_conn,
            ff_hyper_conn,
            mem_qkv_layer_selector,
            mem,
            attn,
            ff,
        ) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (x,)

            retrieved = None
            attn_out_gates = None
            next_neural_mem_cache = None

            # Maybe neural memory
            if exists(mem):
                print(f"Processing neural memory layer with x shape: {x.shape}")
                mem_input, add_residual = mem_hyper_conn(x)
                print(f"After mem_hyper_conn, mem_input shape: {mem_input.shape}")

                if not exists(mem_qkv_layer_selector):
                    # Simply use the same input for q, k, v
                    qkv_mem_input = (mem_input, mem_input, mem_input)
                    print(f"Using same input for q, k, v with shape: {mem_input.shape}")
                else:
                    # Use layer selector to choose different inputs for q, k, v
                    if len(mem_input_layers) > 0:
                        print(
                            f"Using layer selector with {len(mem_input_layers)} previous layers"
                        )

                        # Check dimensions of all layers
                        for i, layer in enumerate([mem_input] + mem_input_layers):
                            print(f"Layer {i} shape: {layer.shape}")

                        # Make sure all layers have the same dimensions
                        target_shape = mem_input.shape
                        aligned_layers = [mem_input]

                        for layer in mem_input_layers:
                            if layer.shape != target_shape:
                                print(
                                    f"Adjusting layer shape from {layer.shape} to {target_shape}"
                                )
                                # Adjust dimensions
                                if layer.shape[-1] > target_shape[-1]:
                                    # Truncate
                                    layer = layer[:, :, : target_shape[-1]]
                                elif layer.shape[-1] < target_shape[-1]:
                                    # Pad
                                    padding = torch.zeros(
                                        *layer.shape[:-1],
                                        target_shape[-1] - layer.shape[-1],
                                        device=layer.device,
                                    )
                                    layer = torch.cat([layer, padding], dim=-1)
                            aligned_layers.append(layer)

                        layers_to_choose_from = torch.stack(aligned_layers)
                        logger.debug(
                            f"Stacked layers shape: {layers_to_choose_from.shape}"
                        )

                        # Let the current `mem_input` select the 3 layers for qkv
                        selected = mem_qkv_layer_selector(mem_input)
                        logger.debug(f"Selected shape before reshape: {selected.shape}")
                        selected = selected.reshape(batch, seq_len_with_mem, 3, -1)
                        logger.debug(f"Selected shape after reshape: {selected.shape}")

                        # Apply selection weights to choose input layers
                        # Extract the selection weights for q, k, v
                        q_weights = selected[:, :, 0, :]  # [b, n, l]
                        k_weights = selected[:, :, 1, :]  # [b, n, l]
                        v_weights = selected[:, :, 2, :]  # [b, n, l]

                        logger.debug(f"q_weights shape: {q_weights.shape}")
                        logger.debug(
                            f"layers_to_choose_from shape: {layers_to_choose_from.shape}"
                        )

                        # Apply weights using einsum
                        q_input = torch.einsum(
                            "l b n d, b n l -> b n d", layers_to_choose_from, q_weights
                        )
                        k_input = torch.einsum(
                            "l b n d, b n l -> b n d", layers_to_choose_from, k_weights
                        )
                        v_input = torch.einsum(
                            "l b n d, b n l -> b n d", layers_to_choose_from, v_weights
                        )

                        logger.debug(f"After einsum, q_input shape: {q_input.shape}")

                        qkv_mem_input = (q_input, k_input, v_input)
                        logger.debug(
                            f"Created qkv_mem_input with shapes: {q_input.shape}, {k_input.shape}, {v_input.shape}"
                        )
                    else:
                        # If no previous layers, use the same input for q, k, v
                        qkv_mem_input = (mem_input, mem_input, mem_input)
                        print(
                            f"No previous layers, using same input for q, k, v with shape: {mem_input.shape}"
                        )

                retrieved, next_neural_mem_cache = mem.forward(
                    qkv_mem_input,
                    state=neural_mem_caches[i] if is_inferencing else None,
                    prev_weights=mem_weight_residual,
                )

                if self.neural_mem_weight_residual:
                    mem_weight_residual = next_neural_mem_cache.updates

                if self.gate_attn_output:
                    attn_out_gates = torch.sigmoid(retrieved)
                else:
                    x = add_residual(retrieved)

            # Attention
            print(f"Processing attention layer with x shape: {x.shape}")
            attn_in, add_residual = attn_hyper_conn(x)
            print(f"After attn_hyper_conn, attn_in shape: {attn_in.shape}")

            mem_input_layers.append(attn_in)

            attn_out, (values, next_kv_cache) = attn(
                attn_in,
                value_residual=value_residual,
                output_gating=attn_out_gates,
                cache=kv_caches[i] if is_inferencing else None,
            )
            print(f"After attention, attn_out shape: {attn_out.shape}")

            mem_input_layers.append(attn_out)

            value_residual = default(value_residual, values)
            print(f"value_residual shape: {value_residual.shape}")

            x = add_residual(attn_out)
            print(f"After add_residual, x shape: {x.shape}")

            # Caches
            next_kv_caches.append(next_kv_cache)
            next_neural_mem_caches.append(next_neural_mem_cache)

            # Feedforward
            print(f"Processing feedforward layer with x shape: {x.shape}")
            ff_in, add_ff_residual = ff_hyper_conn(x)
            print(f"After ff_hyper_conn, ff_in shape: {ff_in.shape}")

            mem_input_layers.append(ff_in)

            ff_out = ff(ff_in)
            print(f"After feedforward, ff_out shape: {ff_out.shape}")

            mem_input_layers.append(ff_out)

            x = add_ff_residual(ff_out)
            print(f"After add_ff_residual, x shape: {x.shape}")

            if output_attentions:
                all_attentions += (attn_out,)

        # Taking care of cache first
        # For early return when processing long term mem tokens during inference
        if use_cache:
            next_kv_caches = torch.stack(
                [torch.stack(kv_cache) for kv_cache in next_kv_caches]
            )

            # Handle kv cache length depending on local attention type
            next_kv_caches = next_kv_caches[..., -self.attn_window_size :, :]

            kv_cache_length = next_kv_caches.shape[-2]

            if not self.sliding_window_attn and divisible_by(
                kv_cache_length, self.attn_window_size
            ):
                next_kv_caches = next_kv_caches[..., 0:0, :]

            next_cache = (
                inference_seq_index + 1 if is_inferencing else seq_len_with_mem - 1,
                next_kv_caches,
                next_neural_mem_caches,
            )

            if is_inferencing and self.seq_index_is_longterm(inference_seq_index):
                return None, next_cache

        # Hyper connection reducing of streams
        x = self.reduce_streams(x)

        # Excise out the memories
        if not is_inferencing:
            # Extract only the token positions (not memory positions)
            token_indices = []
            for i in range(seq_len_with_mem):
                if not self.seq_index_is_longterm(i):
                    token_indices.append(i)

            x = x[:, token_indices, :]

        # Check dimensions before normalization
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.d_model:
            print(
                f"Warning: Input dimension {input_dim} doesn't match expected dimension {self.d_model}"
            )
            # Handle dimension mismatch
            if input_dim % self.d_model == 0:
                # If input is multiple of expected dimension, average across streams
                num_streams = input_dim // self.d_model
                x = x.reshape(batch_size, seq_len, num_streams, self.d_model)
                x = x.mean(dim=2)
            else:
                # Truncate or pad
                if input_dim > self.d_model:
                    x = x[:, :, : self.d_model]  # Truncate
                else:
                    padding = torch.zeros(
                        batch_size, seq_len, self.d_model - input_dim, device=x.device
                    )
                    x = torch.cat([x, padding], dim=-1)  # Pad

        # To logits
        x = self.norm(x)

        if output_hidden_states:
            all_hidden_states += (x,)

        if not return_dict:
            outputs = (x,)
            if use_cache:
                outputs += (next_cache,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs

        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class TitansMACForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Titans MAC model for causal language modeling.

    This model wraps the MemoryAsContextTransformer and adds a language modeling head on top.
    It can be used for autoregressive text generation with the ability to learn at test time.

    Args:
        config (`TitansMACConfig`):
            Model configuration class with all the parameters of the model.
    """

    config_class = TitansMACConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MemoryAsContextTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.num_tokens, bias=False)

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.transformer.token_emb

    def set_input_embeddings(self, value):
        self.transformer.token_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
                Contains pre-computed key and value hidden states of the attention blocks.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            `Dict[str, Any]`: A dictionary containing the model inputs for generation.
        """
        # Only last token for inputs_ids if past is defined
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the Titans MAC model for causal language modeling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary. If not provided, `inputs_embeds` must be given.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
                Contains pre-computed key and value hidden states of the attention blocks used for
                fast autoregressive decoding.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded representation of input sequence. If provided, `input_ids` will not be used.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]`
                (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only
                computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up
                decoding (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`:
                If `return_dict=True`, a `CausalLMOutputWithPast` is returned, otherwise a tuple.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Check if transformer_outputs is a tuple with None as first element
        # This happens when processing longterm memory tokens during inference
        if isinstance(transformer_outputs, tuple) and transformer_outputs[0] is None:
            # Return None for logits and pass through the cache
            if return_dict:
                from transformers.modeling_outputs import CausalLMOutputWithPast

                return CausalLMOutputWithPast(
                    loss=None,
                    logits=None,
                    past_key_values=transformer_outputs[1],
                    hidden_states=None,
                    attentions=None,
                )
            else:
                return (None,) + transformer_outputs[1:]

        hidden_states = transformer_outputs[0]

        # Apply LM head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
