import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ModelOutput:
    label: str
    confidence: float
    probabilities: dict[str, float]


class ModelService:
    def __init__(self, classes, version, image_size=224, device="cpu", model_path=None):
        self.classes, self.version, self.image_size, self.device = (
            classes,
            version,
            image_size,
            device,
        )
        self.backend = "color-smoke"
        self.artifact_sha256 = None
        self.temperature = 1.0
        if model_path:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            self.model = torch.jit.load(model_path, map_location=device).eval()
            self.backend = "torchscript"
            self.artifact_sha256 = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            metadata_path = Path(model_path).with_suffix(".json")
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("sha256") and metadata["sha256"] != self.artifact_sha256:
                    raise ValueError("Model SHA-256 does not match sidecar metadata")
                if metadata.get("classes") and metadata["classes"] != self.classes:
                    raise ValueError("Model classes do not match VISION_CLASSES")
                self.temperature = float(metadata.get("temperature", 1.0))
            self._validate_contract()
        else:
            self.model = None

    @torch.inference_mode()
    def predict_batch(self, tensors):
        batch = torch.stack(tensors).to(self.device)
        if self.model is not None:
            logits = self.model(batch) / self.temperature
        else:
            # Deterministic smoke backend; useful for API/CI, not a trained classifier.
            channel_means = batch.mean((2, 3))
            base = torch.cat(
                [channel_means, -channel_means, channel_means.mean(1, keepdim=True)], dim=1
            )
            repeats = (len(self.classes) + base.shape[1] - 1) // base.shape[1]
            logits = base.repeat(1, repeats)[:, : len(self.classes)]
        probabilities = logits.softmax(1).cpu()
        results = []
        for row in probabilities:
            index = int(row.argmax())
            results.append(
                ModelOutput(
                    self.classes[index], float(row[index]), dict(zip(self.classes, row.tolist()))
                )
            )
        return results

    def info(self):
        return {
            "version": self.version,
            "backend": self.backend,
            "device": self.device,
            "image_size": self.image_size,
            "classes": self.classes,
            "artifact_sha256": self.artifact_sha256,
            "temperature": self.temperature,
        }

    def _validate_contract(self):
        with torch.inference_mode():
            output = self.model(
                torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            )
        expected = (1, len(self.classes))
        if tuple(output.shape) != expected:
            raise ValueError(f"Model must return logits {expected}, got {tuple(output.shape)}")
