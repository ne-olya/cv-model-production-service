import argparse
import json
from pathlib import Path


def check_quality(
    candidate,
    baseline=None,
    min_macro_f1=0.75,
    max_regression=0.01,
    max_latency_ms=None,
    max_model_size_mb=None,
):
    failures = []
    macro_f1 = candidate.get("macro avg", {}).get("f1-score", candidate.get("macro_f1"))
    if macro_f1 is None or macro_f1 < min_macro_f1:
        failures.append(f"macro_f1={macro_f1} below threshold {min_macro_f1}")
    if baseline:
        baseline_f1 = baseline.get("macro avg", {}).get("f1-score", baseline.get("macro_f1"))
        if (
            baseline_f1 is not None
            and macro_f1 is not None
            and baseline_f1 - macro_f1 > max_regression
        ):
            failures.append(
                f"macro_f1 regression {baseline_f1 - macro_f1:.4f} exceeds {max_regression}"
            )
    if max_latency_ms and candidate.get("latency_ms_per_image", float("inf")) > max_latency_ms:
        failures.append("latency exceeds limit")
    if max_model_size_mb and candidate.get("model_size_mb", float("inf")) > max_model_size_mb:
        failures.append("model size exceeds limit")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("--baseline")
    parser.add_argument("--min-macro-f1", type=float, default=0.75)
    parser.add_argument("--max-regression", type=float, default=0.01)
    parser.add_argument("--max-latency-ms", type=float)
    parser.add_argument("--max-model-size-mb", type=float)
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text())
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    failures = check_quality(
        candidate,
        baseline,
        args.min_macro_f1,
        args.max_regression,
        args.max_latency_ms,
        args.max_model_size_mb,
    )
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
