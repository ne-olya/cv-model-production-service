import torch

from app.model import ModelService


def test_smoke_model_contract():
    model = ModelService(["a", "b", "c"], "test", image_size=16)
    result = model.predict_batch([torch.zeros(3, 16, 16)])[0]
    assert result.label in model.classes
    assert abs(sum(result.probabilities.values()) - 1) < 1e-6


def test_missing_artifact_fails_fast(tmp_path):
    try:
        ModelService(["a"], "test", model_path=tmp_path / "missing.pt")
    except FileNotFoundError:
        return
    raise AssertionError("Missing artifact must fail")


def test_torchscript_contract_and_hash(tmp_path):
    class Valid(torch.nn.Module):
        def forward(self, value):
            return value.mean((2, 3))

    path = tmp_path / "model.ts"
    torch.jit.trace(Valid(), torch.zeros(1, 3, 8, 8)).save(str(path))
    service = ModelService(["a", "b", "c"], "test", image_size=8, model_path=path)
    assert service.info()["artifact_sha256"]
    assert service.backend == "torchscript"
