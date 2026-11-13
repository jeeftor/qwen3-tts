# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
qwen_tts: Qwen-TTS package.
"""

# transformers>=5 no longer defaults pad/bos/eos_token_id to None on
# PretrainedConfig (4.x did). The vendored Qwen3-TTS modeling code was
# written against 4.x semantics — restore them.
from transformers import PretrainedConfig as _PretrainedConfig

_PretrainedConfig_init = _PretrainedConfig.__init__


def _pretrained_config_init_compat(self, *args, **kwargs):
    _PretrainedConfig_init(self, *args, **kwargs)
    for _attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        if not hasattr(self, _attr):
            setattr(self, _attr, None)


_PretrainedConfig.__init__ = _pretrained_config_init_compat

from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["__version__"]