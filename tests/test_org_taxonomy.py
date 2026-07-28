import unittest
from pathlib import Path

from src.normalize.org_taxonomy import (
    OrgTaxonomy,
    is_academic,
    is_industry,
    is_valid_destination,
)

ORGS = Path(__file__).resolve().parent.parent / "data" / "orgs.csv"


class TaxonomyFunctionTests(unittest.TestCase):
    def test_plain_industry_counts_as_industry(self):
        # Regression: the old isin(['industry_lab','industry_parent']) dropped these.
        self.assertTrue(is_industry("industry"))
        self.assertTrue(is_industry("industry_lab"))
        self.assertTrue(is_industry("industry_parent"))

    def test_academic(self):
        self.assertTrue(is_academic("academic"))
        self.assertFalse(is_academic("education"))  # legacy OpenAlex value, dropped

    def test_shell_parent_is_not_a_destination(self):
        self.assertFalse(is_valid_destination("industry_parent"))
        self.assertTrue(is_valid_destination("industry_lab"))
        self.assertTrue(is_valid_destination("academic"))
        self.assertFalse(is_valid_destination(""))
        self.assertFalse(is_valid_destination(None))


class TaxonomyFromCsvTests(unittest.TestCase):
    def setUp(self):
        self.tax = OrgTaxonomy.from_csv(ORGS)

    def test_curated_frontier_labs_present(self):
        for oid in ("moonshot", "mistral", "world_labs", "safe_superintelligence"):
            self.assertTrue(self.tax.is_industry(oid), oid)
            self.assertTrue(self.tax.is_valid_destination(oid), oid)

    def test_rollup_to_group(self):
        self.assertEqual(self.tax.rollup_to_group("google_deepmind"), "alphabet")
        # A top-level lab with no parent rolls up to itself.
        self.assertEqual(self.tax.rollup_to_group("moonshot"), "moonshot")

    def test_shell_parent_not_destination(self):
        self.assertFalse(self.tax.is_valid_destination("alphabet"))


if __name__ == "__main__":
    unittest.main()
