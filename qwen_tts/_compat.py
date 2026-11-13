# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for running the vendored 4.x-era model code on
transformers>=5. Imported first by ``qwen_tts/__init__.py`` — keep this
module side-effect-only and free of qwen_tts imports (it must run before
the modeling modules are defined)."""

import torch
from transformers import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# transformers>=5 no longer defaults pad/bos/eos_token_id to None on
# PretrainedConfig (4.x did). The vendored modeling code reads
# config.pad_token_id directly — restore the 4.x semantics.
_PretrainedConfig_init = PretrainedConfig.__init__


def _pretrained_config_init_compat(self, *args, **kwargs):
    _PretrainedConfig_init(self, *args, **kwargs)
    for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if not hasattr(self, _attr):
            setattr(self, _attr, None)


PretrainedConfig.__init__ = _pretrained_config_init_compat


# transformers>=5 removed "default" from ROPE_INIT_FUNCTIONS (each model
# now supplies its own compute_default_rope_parameters). The vendored RoPE
# classes index the dict unconditionally — resolve "default" to the 4.x
# implementation here instead of mutating the shared dict (which would
# shadow transformers' own per-module default during weight init).
def _compute_default_rope_parameters_compat(
    config=None, device=None, seq_len=None, **kwargs
):
    rope_params = getattr(config, "rope_parameters", None) or {}
    base = getattr(config, "rope_theta", None) or rope_params.get(
        "rope_theta", 10000.0
    )
    partial_rotary_factor = getattr(config, "partial_rotary_factor", None)
    if partial_rotary_factor is None:
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or rope_params.get("head_dim")
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64)
            .to(device=device, dtype=torch.float)
            / dim
        )
    )
    return inv_freq, 1.0


def get_rope_init_fn(rope_type: str):
    if rope_type == "default":
        return _compute_default_rope_parameters_compat
    return ROPE_INIT_FUNCTIONS[rope_type]
