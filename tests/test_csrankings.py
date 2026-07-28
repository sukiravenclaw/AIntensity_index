import unittest
from pathlib import Path

import pandas as pd

from src.collect import csrankings

ORGS = Path(__file__).resolve().parent.parent / "data" / "orgs.csv"


class AnnotateFacultyTests(unittest.TestCase):
    def _cfg(self):
        return {"paths": {"orgs": str(ORGS)}}

    def test_adds_normalized_name_and_org_id(self):
        combined = pd.DataFrame(
            [
                {"name": "Ruslan Salakhutdinov", "affiliation": "Carnegie Mellon University"},
                {"name": "Fei-Fei Li", "affiliation": "Stanford University"},
                {"name": "Nobody", "affiliation": "Nonexistent College"},
            ]
        )
        out = csrankings._annotate_faculty(combined, self._cfg())
        self.assertIn("normalized_name", out.columns)
        self.assertIn("org_id", out.columns)
        self.assertEqual(out.loc[1, "org_id"], "stanford")
        self.assertTrue(pd.notna(out.loc[0, "org_id"]))  # CMU resolves to some org_id
        self.assertTrue(pd.isna(out.loc[2, "org_id"]))    # unknown affiliation

    def test_empty_frame_gets_columns(self):
        out = csrankings._annotate_faculty(pd.DataFrame(columns=["name", "affiliation"]), self._cfg())
        self.assertIn("normalized_name", out.columns)
        self.assertIn("org_id", out.columns)


if __name__ == "__main__":
    unittest.main()
