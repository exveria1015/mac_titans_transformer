from transformers.configuration_utils import PretrainedConfig


# Configuration class for Titans MAC
class TitansMACConfig(PretrainedConfig):
    model_type = "titans_mac"

    def __init__(
        self,
        num_tokens=50257,
        d_model=768,
        depth=12,
        num_heads=12,
        dim_head=64,
        num_memory_layers=2,
        segment_len=64,
        neural_memory_segment_len=None,
        neural_mem_gate_attn_output=False,
        neural_memory_add_value_residual=False,
        num_longterm_mem_tokens=0,
        persistent_size=64,
        neural_memory_batch_size=None,
        neural_memory_qkv_receives_diff_views=False,
        dropout=0.1,
        ff_mult=4,
        num_residual_streams=4,
        neural_memory_model=None,
        neural_memory_kwargs=None,
        neural_memory_layers=None,
        use_flex_attn=False,
        sliding_window_attn=False,
        neural_mem_weight_residual=False,
        context_window=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        max_length=20,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.num_tokens = num_tokens
        self.d_model = d_model
        self.depth = depth
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.num_memory_layers = num_memory_layers
        self.segment_len = segment_len
        self.neural_memory_segment_len = neural_memory_segment_len
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output
        self.neural_memory_add_value_residual = neural_memory_add_value_residual
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.persistent_size = persistent_size
        self.neural_memory_batch_size = neural_memory_batch_size
        self.neural_memory_qkv_receives_diff_views = (
            neural_memory_qkv_receives_diff_views
        )
        self.dropout = dropout
        self.ff_mult = ff_mult
        self.num_residual_streams = num_residual_streams
        self.neural_memory_model = neural_memory_model
        self.neural_memory_kwargs = (
            {} if neural_memory_kwargs is None else dict(neural_memory_kwargs)
        )
        self.neural_memory_layers = neural_memory_layers
        self.use_flex_attn = use_flex_attn
        self.sliding_window_attn = sliding_window_attn
        self.neural_mem_weight_residual = neural_mem_weight_residual
        self.context_window = context_window
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_length = max_length
