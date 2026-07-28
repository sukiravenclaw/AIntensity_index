import unittest

import pandas as pd

from src.derive import intensity
from src.normalize.org_taxonomy import OrgTaxonomy


def _taxonomy():
    return OrgTaxonomy(
        org_type_by_id={"openai": "industry_lab", "moonshot": "industry_lab", "stanford": "academic"},
        parent_by_id={},
    )


class PersonWeightTests(unittest.TestCase):
    def test_fractional_credit_and_weight(self):
        persons = pd.DataFrame([
            {"author_dblp_pid": "a", "display_name": "A"},
            {"author_dblp_pid": "b", "display_name": "B"},
        ])
        # w1: 2 authors, 100 citations -> 50 each. w2: 1 author (b), 0 cites.
        works = pd.DataFrame([
            {"work_id": "w1", "venue_normalized": "NeurIPS", "cited_by_count": 100},
            {"work_id": "w2", "venue_normalized": "NeurIPS", "cited_by_count": 0},
        ])
        seed = pd.DataFrame([
            {"work_id": "w1", "author_dblp_pid": "a"},
            {"work_id": "w1", "author_dblp_pid": "b"},
            {"work_id": "w2", "author_dblp_pid": "b"},
        ])
        out = intensity.compute_person_weights(persons, seed, works).set_index("author_dblp_pid")
        self.assertAlmostEqual(out.loc["a", "fractional_citations"], 50.0)
        self.assertAlmostEqual(out.loc["b", "fractional_citations"], 50.0)
        self.assertAlmostEqual(out.loc["a", "fractional_pubs"], 0.5)
        self.assertAlmostEqual(out.loc["b", "fractional_pubs"], 1.5)
        self.assertTrue((out["weight"] >= 0).all())  # shifted non-negative


class IntensityTests(unittest.TestCase):
    def test_founder_bonus_and_confidence(self):
        weights = pd.DataFrame([
            {"author_dblp_pid": "a", "weight": 1.0},
            {"author_dblp_pid": "b", "weight": 1.0},
        ])
        employers = pd.DataFrame([
            # a: founder of moonshot -> contributes weight*conf(1.0) + beta*weight = 2.0
            {"author_dblp_pid": "a", "employer_org_id": "moonshot", "confidence": 1.0, "is_founder": True},
            # b: openai via s2 (conf 0.4), not founder -> 0.4
            {"author_dblp_pid": "b", "employer_org_id": "openai", "confidence": 0.4, "is_founder": False},
        ])
        out = intensity.compute_intensity(weights, employers, _taxonomy()).set_index("employer_org_id")
        self.assertAlmostEqual(out.loc["moonshot", "intensity_raw"], 2.0)
        self.assertAlmostEqual(out.loc["openai", "intensity_raw"], 0.4)
        self.assertAlmostEqual(out.loc["moonshot", "intensity_index"], 100.0)  # max
        self.assertEqual(int(out.loc["moonshot", "n_founders"]), 1)


if __name__ == "__main__":
    unittest.main()
