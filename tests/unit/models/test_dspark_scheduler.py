"""Unit tests for the DSpark hardware-aware prefix scheduler (pure Python)."""

from speculators.models.dspark.scheduler import (
    make_sps_lookup,
    schedule_prefix_lengths,
)


def test_empty_batch_returns_empty():
    assert schedule_prefix_lengths([], make_sps_lookup([1.0])) == []


def test_flat_throughput_admits_all_positive_survival():
    # Constant SPS => throughput == tau, which only grows as we admit positive
    # survival mass, so every draft position is admitted.
    conf = [[0.9, 0.8], [0.5, 0.5]]
    sps = make_sps_lookup([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert schedule_prefix_lengths(conf, sps) == [2, 2]


def test_throughput_cliff_stops_admission_early():
    # SPS is flat up to B=4 then cliffs at B=5, so the low-survival 4th token
    # (which pushes B to 5) is rejected even though a_{r,4} > 0.
    conf = [[0.99, 0.95, 0.5, 0.1]]  # survivals ~ .99/.94/.47/.047
    sps = make_sps_lookup([1.0, 1.0, 1.0, 1.0, 1.0, 0.1])
    assert schedule_prefix_lengths(conf, sps) == [3]


def test_zero_confidence_truncates_candidate():
    # A zero mid-block confidence kills all later survivals for that request.
    conf = [[0.9, 0.0, 0.9]]
    sps = make_sps_lookup([1.0, 1.0, 1.0, 1.0])
    assert schedule_prefix_lengths(conf, sps) == [1]


def _run_all():
    test_empty_batch_returns_empty()
    test_flat_throughput_admits_all_positive_survival()
    test_throughput_cliff_stops_admission_early()
    test_zero_confidence_truncates_candidate()
    print("test_dspark_scheduler: all passed")


if __name__ == "__main__":
    _run_all()
