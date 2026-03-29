import numpy as np
import pytest

from src.utils import tf_runtime


class _DummyTF:
    class config:
        class experimental:
            @staticmethod
            def set_memory_growth(_gpu, _enabled):
                return None

        @staticmethod
        def list_physical_devices(kind):
            if kind == "GPU":
                return []
            return []

    class random:
        last_seed = None

        @staticmethod
        def set_seed(seed):
            _DummyTF.random.last_seed = seed


def test_configure_tensorflow_runtime_invalid_mode_raises():
    with pytest.raises(ValueError, match="tf_device invalido"):
        tf_runtime.configure_tensorflow_runtime("invalido")


def test_configure_tensorflow_runtime_cpu_uses_stub(monkeypatch):
    # Evita importar TensorFlow real no teste.
    monkeypatch.setitem(__import__("sys").modules, "tensorflow", _DummyTF)

    tf, runtime = tf_runtime.configure_tensorflow_runtime("cpu")

    assert tf is _DummyTF
    assert runtime["requested"] == "cpu"
    assert runtime["active"] == "cpu"
    assert runtime["gpu_count"] == 0


def test_setup_deterministic_runtime_reseeds(monkeypatch):
    monkeypatch.setattr(
        tf_runtime,
        "configure_tensorflow_runtime",
        lambda tf_device="auto": (
            _DummyTF,
            {
                "requested": tf_device,
                "active": "cpu",
                "gpu_count": 0,
                "gpu_names": [],
            },
        ),
    )

    _, runtime_1 = tf_runtime.setup_deterministic_runtime(seed=321, tf_device="auto")
    py_1 = __import__("random").random()
    np_1 = float(np.random.rand())

    _, runtime_2 = tf_runtime.setup_deterministic_runtime(seed=321, tf_device="auto")
    py_2 = __import__("random").random()
    np_2 = float(np.random.rand())

    assert runtime_1["seed"] == 321
    assert runtime_2["seed"] == 321
    assert _DummyTF.random.last_seed == 321
    assert py_1 == py_2
    assert np_1 == np_2
