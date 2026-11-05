"""The paired guard-off / guard-on comparison and its exact McNemar test."""

from __future__ import annotations

import pytest

from backstop.recover.paired import mcnemar_exact, paired_result


def test_only_discordant_episodes_count() -> None:
    off = [True, True, False, False, False]
    on = [True, False, True, True, False]
    result = paired_result(off, on)
    assert (result.rescued, result.spoiled) == (2, 1)
    assert (result.off_success, result.on_success) == (2, 3)
    assert result.success_rate_change == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("rescued", "spoiled", "expected"),
    [
        (0, 0, 1.0),
        (3, 3, 1.0),
        (6, 0, 2 / 64),  # all six discordant pairs one way: 2 * (1/2)^6
        (8, 2, 2 * (1 + 10 + 45) / 1024),
    ],
)
def test_mcnemar_exact_is_the_two_sided_binomial(
    rescued: int, spoiled: int, expected: float
) -> None:
    assert mcnemar_exact(rescued, spoiled) == pytest.approx(expected)
    assert mcnemar_exact(spoiled, rescued) == pytest.approx(expected)


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="guard-off"):
        paired_result([True], [True, False])
