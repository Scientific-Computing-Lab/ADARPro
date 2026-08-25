import numpy as np
import pandas as pd

from adarpro.models.joint_expanded_oof import TISSUES, sample_indices


def test_joint_sampler_balances_tissue_and_state():
    rows = []
    for context_index, context in enumerate(TISSUES):
        for state in (0.0, 20.0):
            for pair in range(3):
                rows.append(
                    {
                        "context": context,
                        "editing_index": state,
                        "pair_id": 100 * context_index + pair,
                    }
                )
    table = pd.DataFrame(rows)
    selected = sample_indices(
        table,
        np.arange(len(table)),
        per_tissue_samples=10,
        pair_balanced=False,
        rng=np.random.default_rng(1),
    )
    sampled = table.iloc[selected]
    counts = sampled.groupby(["context", sampled["editing_index"] >= 15]).size()
    assert len(selected) == 40
    assert set(counts) == {5}


def test_joint_sampler_can_share_a_fixed_total_budget_across_tissues():
    rows = []
    for context_index, context in enumerate(TISSUES):
        for state in (0.0, 20.0):
            for pair in range(3):
                rows.append(
                    {
                        "context": context,
                        "editing_index": state,
                        "pair_id": 100 * context_index + pair,
                    }
                )
    table = pd.DataFrame(rows)
    total_budget = 40
    selected = sample_indices(
        table,
        np.arange(len(table)),
        per_tissue_samples=total_budget // len(TISSUES),
        pair_balanced=True,
        rng=np.random.default_rng(2),
    )
    assert len(selected) == total_budget


def test_joint_sampler_accepts_per_tissue_budgets():
    rows = []
    budgets = {context: 8 + 2 * index for index, context in enumerate(TISSUES)}
    for context_index, context in enumerate(TISSUES):
        for state in (0.0, 20.0):
            for pair in range(3):
                rows.append(
                    {
                        "context": context,
                        "editing_index": state,
                        "pair_id": 100 * context_index + pair,
                    }
                )
    table = pd.DataFrame(rows)
    selected = sample_indices(
        table,
        np.arange(len(table)),
        per_tissue_samples=budgets,
        pair_balanced=True,
        rng=np.random.default_rng(3),
    )
    sampled = table.iloc[selected]
    assert len(selected) == sum(budgets.values())
    assert sampled.groupby("context").size().to_dict() == budgets
