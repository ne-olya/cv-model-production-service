import threading
from collections import Counter


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.latencies = []
        self.confidences = []
        self.labels = Counter()

    def observe(self, latency_ms, outputs=None, error=False):
        with self.lock:
            self.requests += 1
            self.latencies.append(latency_ms)
            if error:
                self.errors += 1
            for output in outputs or []:
                self.labels[output.label] += 1
                self.confidences.append(output.confidence)

    def summary(self):
        with self.lock:
            ordered = sorted(self.latencies)
            return {
                "requests": self.requests,
                "errors": self.errors,
                "average_latency_ms": sum(self.latencies) / len(self.latencies)
                if self.latencies
                else 0,
                "average_confidence": sum(self.confidences) / len(self.confidences)
                if self.confidences
                else 0,
                "p95_latency_ms": ordered[int(0.95 * (len(ordered) - 1))] if ordered else 0,
                "confidence_histogram": {
                    f"{lower / 10:.1f}-{(lower + 1) / 10:.1f}": sum(
                        lower / 10 <= value < (lower + 1) / 10 for value in self.confidences
                    )
                    for lower in range(10)
                },
                "predicted_classes": dict(self.labels),
            }

    def prometheus(self):
        data = self.summary()
        lines = [
            f"vision_requests_total {data['requests']}",
            f"vision_errors_total {data['errors']}",
            f"vision_latency_ms_avg {data['average_latency_ms']}",
            f"vision_confidence_avg {data['average_confidence']}",
        ]
        lines += [
            f'vision_predictions_total{{class="{label}"}} {count}'
            for label, count in data["predicted_classes"].items()
        ]
        return "\n".join(lines) + "\n"
