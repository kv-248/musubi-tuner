"""Importing the Wan T5 encoder must not require a CUDA build of torch.

``T5EncoderModel.__init__`` used to default ``device`` to ``torch.cuda.current_device()``.
Python evaluates default arguments while the class body executes, so importing
``musubi_tuner.wan.modules.t5`` (and ``musubi_tuner.wan.modules``, which re-exports it)
raised ``AssertionError: Torch not compiled with CUDA enabled`` on a CPU or MPS build
before any encoder was constructed.
"""

import importlib
import inspect
import sys

import torch


def test_wan_t5_encoder_imports_when_cuda_current_device_raises(monkeypatch):
    def _cuda_unavailable(*_args, **_kwargs):
        raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(torch.cuda, "current_device", _cuda_unavailable)

    # A previous import in this process would have already bound the default.
    # Drop it so the class body runs again while current_device raises.
    for name in ("musubi_tuner.wan.modules.t5", "musubi_tuner.wan.modules"):
        sys.modules.pop(name, None)

    t5 = importlib.import_module("musubi_tuner.wan.modules.t5")

    device_default = inspect.signature(t5.T5EncoderModel.__init__).parameters["device"].default
    assert device_default is None

    wan_modules = importlib.import_module("musubi_tuner.wan.modules")
    assert wan_modules.T5EncoderModel is t5.T5EncoderModel
    assert wan_modules.WanModel.__name__ == "WanModel"
