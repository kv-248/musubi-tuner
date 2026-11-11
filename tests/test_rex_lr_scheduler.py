"""RexLR warmup must cover the learning rate the first optimizer step actually uses.

``LRScheduler`` applies ``get_lr`` once during construction, and the training loop
calls ``optimizer.step()`` before ``lr_scheduler.step()``. The rate at ``last_epoch == 0``
is therefore the first update. With ``num_warmup_steps > 1`` that epoch used to miss the
warmup branch; ``step_after`` was negative and ``step_after % remaining_steps`` wrapped it
into the middle of the decay, so training started near half of ``max_lr`` and then jumped
down to ``min_lr``. A single warmup step was applied twice for the same reason.
"""

import pytest
import torch

from musubi_tuner.modules.lr_schedulers import RexLR


def _lrs_used_by_optimizer(scheduler: RexLR, updates: int) -> list[float]:
    """Learning rates consumed by ``updates`` optimizer steps, in training order."""
    optimizer = scheduler.optimizer
    used = []
    for _ in range(updates):
        used.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return used


def _make_scheduler(*, num_steps: int, num_warmup_steps: int, max_lr: float = 1e-4, min_lr: float = 1e-6) -> RexLR:
    optimizer = torch.optim.SGD([torch.zeros(1)], lr=max_lr)
    return RexLR(
        optimizer,
        max_lr=max_lr,
        min_lr=min_lr,
        num_steps=num_steps,
        num_warmup_steps=num_warmup_steps,
    )


def test_multi_step_warmup_starts_at_min_lr_and_ramps_to_max():
    max_lr = 1e-4
    min_lr = 1e-6
    warmup = 100
    scheduler = _make_scheduler(num_steps=1000, num_warmup_steps=warmup, max_lr=max_lr, min_lr=min_lr)

    lrs = _lrs_used_by_optimizer(scheduler, warmup + 2)

    assert lrs[0] == pytest.approx(min_lr)
    for step, lr in enumerate(lrs[:warmup]):
        expected = min_lr + (max_lr - min_lr) * step / (warmup - 1)
        assert lr == pytest.approx(expected)
    assert lrs[warmup - 1] == pytest.approx(max_lr)
    # The first decay step is the REX peak (z = 1), then the curve decreases.
    assert lrs[warmup] == pytest.approx(max_lr)
    assert lrs[warmup + 1] < lrs[warmup]


def test_two_warmup_steps_do_not_spike_before_the_ramp():
    scheduler = _make_scheduler(num_steps=20, num_warmup_steps=2)

    lrs = _lrs_used_by_optimizer(scheduler, 4)

    assert lrs[0] == pytest.approx(1e-6)
    assert lrs[1] == pytest.approx(1e-4)
    assert lrs[2] == pytest.approx(1e-4)
    assert lrs[3] < lrs[2]


def test_single_warmup_step_is_only_the_first_update():
    scheduler = _make_scheduler(num_steps=100, num_warmup_steps=1)

    lrs = _lrs_used_by_optimizer(scheduler, 3)

    assert lrs[0] == pytest.approx(1e-6)
    assert lrs[1] == pytest.approx(1e-4)
    assert lrs[2] < lrs[1]


def test_no_warmup_starts_at_the_rex_peak_and_decays():
    scheduler = _make_scheduler(num_steps=100, num_warmup_steps=0)

    lrs = _lrs_used_by_optimizer(scheduler, 4)

    assert lrs[0] == pytest.approx(1e-4)
    assert lrs[1] < lrs[0]
    assert lrs[2] < lrs[1]
    assert lrs[3] > 1e-6
