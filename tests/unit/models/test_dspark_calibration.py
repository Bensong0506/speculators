"""Unit tests for DSpark Sequential Temperature Scaling (pure Python)."""

from speculators.models.dspark.calibration import (
    calibrated_prefix_survival,
    expected_calibration_error,
    fit_sequential_temperature_scaling,
    temperature_scale,
)


def test_temperature_scale_fixes_half_and_is_monotone():
    # logit(0.5) == 0, so 0.5 is a fixed point of temperature scaling.
    assert abs(temperature_scale(0.5, 3.0) - 0.5) < 1e-9
    # Order preserving: a higher raw prob stays higher after scaling.
    assert temperature_scale(0.9, 4.0) > temperature_scale(0.6, 4.0)
    # T > 1 pulls a confident prob toward 0.5.
    assert temperature_scale(0.9, 4.0) < 0.9


def test_ece_zero_when_perfectly_calibrated():
    preds = [0.0, 0.0, 1.0, 1.0]
    labels = [0.0, 0.0, 1.0, 1.0]
    assert expected_calibration_error(preds, labels, num_bins=10) < 1e-9


def test_calibrated_prefix_survival_is_cumprod():
    conf = [0.8, 0.5]
    out = calibrated_prefix_survival(conf, [1.0, 1.0])
    assert abs(out[0] - 0.8) < 1e-9
    assert abs(out[1] - 0.4) < 1e-9


def test_sts_reduces_ece_on_overconfident_head():
    # Raw confidence is a fixed 0.9 but the empirical survival rate is 0.6:
    # a temperature > 1 should pull the prediction toward 0.6 and cut ECE.
    conf = [[0.9] for _ in range(100)]
    labels = [[1.0] for _ in range(60)] + [[0.0] for _ in range(40)]

    temps = fit_sequential_temperature_scaling(conf, labels, block_size=1)

    ece_raw = expected_calibration_error([0.9] * 100, [1.0] * 60 + [0.0] * 40)
    calibrated = temperature_scale(0.9, temps[0])
    ece_cal = expected_calibration_error([calibrated] * 100, [1.0] * 60 + [0.0] * 40)

    assert temps[0] > 1.5  # de-confidence: T pushed above 1
    assert ece_cal < ece_raw
    assert ece_cal < 0.05  # calibrated prediction lands near the 0.6 truth


def _run_all():
    test_temperature_scale_fixes_half_and_is_monotone()
    test_ece_zero_when_perfectly_calibrated()
    test_calibrated_prefix_survival_is_cumprod()
    test_sts_reduces_ece_on_overconfident_head()
    print("test_dspark_calibration: all passed")


if __name__ == "__main__":
    _run_all()
