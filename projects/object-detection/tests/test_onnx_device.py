import pytest

from onnx_device import providers_for

CPU_ONLY = ["CPUExecutionProvider"]
NVIDIA = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
WINDOWS = ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_auto_picks_the_gpu_when_there_is_one():
    assert providers_for("auto", CPU_ONLY) == ["CPUExecutionProvider"]
    assert providers_for("auto", NVIDIA) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert providers_for("auto", WINDOWS) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert providers_for(None, NVIDIA)[0] == "CUDAExecutionProvider"  # type: ignore[arg-type]


def test_explicit_device():
    assert providers_for("cpu", NVIDIA) == ["CPUExecutionProvider"]
    assert providers_for("CUDA:1", NVIDIA) == [
        ("CUDAExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]


def test_missing_or_unknown_device_is_an_error():
    with pytest.raises(RuntimeError, match="no CUDAExecutionProvider"):
        providers_for("cuda", CPU_ONLY)
    with pytest.raises(ValueError, match="Unknown device"):
        providers_for("tpu", NVIDIA)


def test_session_on_this_machine(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from onnx_device import create_session

    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])], "relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    path = tmp_path / "relu.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                                ir_version=8), path)
    session = create_session(path, "auto")
    assert "CPUExecutionProvider" in session.get_providers()
