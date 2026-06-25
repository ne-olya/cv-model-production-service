from app.quality_gate import check_quality


def test_quality_gate_catches_regression():
    failures = check_quality(
        {"macro_f1": 0.80}, {"macro_f1": 0.85}, min_macro_f1=0.75, max_regression=0.01
    )
    assert failures and "regression" in failures[0]


def test_quality_gate_accepts_candidate():
    assert not check_quality(
        {"macro_f1": 0.86, "latency_ms_per_image": 12, "model_size_mb": 20},
        {"macro_f1": 0.85},
        max_latency_ms=20,
        max_model_size_mb=30,
    )
