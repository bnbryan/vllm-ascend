# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def compiler_interface():
    # The CPU UT environment has no npugraph_ex; only its registration hook
    # is needed to exercise the production runtime initializer.
    module_name = "vllm_ascend.compilation.compiler_interface"
    static_name = "npugraph_ex._acl_concrete_graph.static_kernel"
    static = ModuleType(static_name)
    static._set_static_kernel_blacklist = Mock()

    existing = sys.modules.get(module_name)
    parent = importlib.import_module("vllm_ascend.compilation")
    missing = object()
    previous_attribute = getattr(parent, "compiler_interface", missing)
    with patch.dict(sys.modules, {static_name: static}):
        module = importlib.import_module(module_name)
        try:
            yield module
        finally:
            if existing is None:
                sys.modules.pop(module_name, None)
            if previous_attribute is missing:
                delattr(parent, "compiler_interface")
            else:
                parent.compiler_interface = previous_attribute


@pytest.mark.parametrize(
    "model_type",
    ["glm5_next", "glm5_next_text", "glm5_next_mtp", "deepseek_v3", "qwen3"],
)
def test_static_exclusions_are_scoped_to_glm5_next(compiler_interface, monkeypatch, model_type):
    """GLM keeps every operator eligible; other models keep their workarounds."""
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    register = Mock()
    monkeypatch.setattr(compiler_interface, "_set_static_kernel_blacklist", register)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type)),
        parallel_config=SimpleNamespace(
            local_world_size=8,
            data_parallel_size_local=1,
            data_parallel_external_lb=False,
        ),
    )

    compiler_interface._initialize_static_kernel_runtime(config)

    register.assert_called_once()
    excluded = register.call_args.args[0]
    if model_type in compiler_interface.GLM5_NEXT_MODEL_TYPES:
        # Only GLM's own workaround applies; other models' exclusions must not leak in.
        assert excluded == ["PoolKeyIndexer"]
    else:
        assert set(excluded) == {
            "TopKV2",
            "GenericBlockSparseAttention",
            "QuantBatchMatmulV3",
            "TransposeQuantBatchMatMul",
            "QuantLightningIndexer",
        }
