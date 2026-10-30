import unittest
import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestMerchantGuardPipeline(unittest.TestCase):

    def test_feature_columns_schema_exists(self):
        schema_path = os.path.join(PROJECT_ROOT, "data", "splits", "feature_columns_v3b.json")
        self.assertTrue(os.path.exists(schema_path), f"Schema path does not exist: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 10, "Feature list should have over 10 features")

    def test_score_normalization_logic(self):
        from src.models.ensemble import normalize_scores
        raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        norm = normalize_scores(raw, fixed_min=0.0, fixed_max=100.0)
        self.assertEqual(len(norm), 5)
        self.assertTrue((norm >= 0.0).all() and (norm <= 1.0).all())
        self.assertAlmostEqual(norm[0], 0.1)
        self.assertAlmostEqual(norm[-1], 0.5)

    def test_single_row_normalization_with_fixed_bounds(self):
        from src.models.ensemble import normalize_scores
        single_row = np.array([25.0])
        norm = normalize_scores(single_row, fixed_min=0.0, fixed_max=50.0)
        self.assertAlmostEqual(norm[0], 0.5)

    def test_severity_bucketing_logic(self):
        from src.policy.severity import severity_bucket
        self.assertIn(severity_bucket(0.05), ["Low", "Cleared", "Allow"])
        self.assertIn(severity_bucket(0.99), ["Critical", "High"])

    def test_action_engine_policy_mapping(self):
        from src.policy.action_engine import ACTION_MAP
        self.assertIn("Low", ACTION_MAP)
        self.assertIn("Critical", ACTION_MAP)
        self.assertEqual(ACTION_MAP["Critical"]["action"], "Block")
        self.assertTrue(ACTION_MAP["Critical"]["auto_execute"])
        self.assertTrue(ACTION_MAP["Critical"]["requires_review"])

if __name__ == "__main__":
    unittest.main()
