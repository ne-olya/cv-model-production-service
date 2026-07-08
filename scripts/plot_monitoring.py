import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="assets/monitoring-snapshot.png")
    args = parser.parse_args()
    with urllib.request.urlopen(f"{args.api}/monitoring-summary", timeout=5) as response:
        data = json.loads(response.read())
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    classes = data["predicted_classes"]
    axes[0].bar(classes.keys(), classes.values(), color="#2e7d32")
    axes[0].set(title="Распределение предсказаний", ylabel="Запросы")
    axes[0].tick_params(axis="x", rotation=25)
    histogram = data["confidence_histogram"]
    axes[1].bar(histogram.keys(), histogram.values(), color="#e09f3e")
    axes[1].set(title=f"Confidence · p95={data['p95_latency_ms']:.1f} ms", ylabel="Предсказания")
    axes[1].tick_params(axis="x", rotation=45)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
