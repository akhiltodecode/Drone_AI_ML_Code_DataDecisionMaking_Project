"""Scientific contracts for grouped model selection and ablations."""

from __future__ import annotations

import unittest

import pandas as pd

from drone_study.extended import (
    FAMILIES,
    candidate_grid,
    feature_columns,
    grouped_folds,
)
from drone_study.pipeline import FEATURE_COLS


class ExtendedStudyTests(unittest.TestCase):
    def test_prespecified_candidate_space_is_complete(self) -> None:
        self.assertEqual(len(FAMILIES), 3)
        for family in FAMILIES:
            candidates = candidate_grid(family)
            self.assertEqual(len(candidates), 12)
            serialized = {repr(sorted(candidate.items())) for candidate in candidates}
            self.assertEqual(len(serialized), 12)

    def test_grouped_folds_are_deterministic_and_disjoint(self) -> None:
        groups = pd.Series([f"asset_{index // 3:02d}" for index in range(36)])
        first = list(grouped_folds(groups, folds=3, seed=17))
        repeated = list(grouped_folds(groups, folds=3, seed=17))
        self.assertEqual(len(first), 3)
        for (fit, validation), (fit_again, validation_again) in zip(first, repeated):
            self.assertEqual(fit.tolist(), fit_again.tolist())
            self.assertEqual(validation.tolist(), validation_again.tolist())
            fit_assets = set(groups.iloc[fit])
            validation_assets = set(groups.iloc[validation])
            self.assertTrue(fit_assets.isdisjoint(validation_assets))

    def test_ablation_feature_sets_are_strict_subsets(self) -> None:
        complete = feature_columns("all_features")
        self.assertEqual(complete, FEATURE_COLS)
        for name in ("without_rgb", "without_thermal", "without_flight_quality"):
            ablated = feature_columns(name)
            self.assertLess(len(ablated), len(complete))
            self.assertTrue(set(ablated).issubset(complete))


if __name__ == "__main__":
    unittest.main()
