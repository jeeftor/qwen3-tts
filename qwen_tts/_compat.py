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
# now supplies its own default impl). The vendored RoPE classes index the
# dict unconditionally — restore the 4.x entry verbatim.
def _compute_default_rope_parameters_compat(
    config=None, device=None, seq_len=None, **kwargs
):
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
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


ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters_compat)
