import unittest

import pandas as pd

from src.collect import orcid
from src.derive import employer


class OrcidParseTests(unittest.TestCase):
    def test_parse_and_most_recent(self):
        payload = {
            "affiliation-group": [
                {"summaries": [{"employment-summary": {
                    "organization": {"name": "Carnegie Mellon University"},
                    "start-date": {"year": {"value": "2016"}},
                    "end-date": {"year": {"value": "2019"}},
                    "role-title": "PhD Student"}}]},
                {"summaries": [{"employment-summary": {
                    "organization": {"name": "Moonshot AI"},
                    "start-date": {"year": {"value": "2023"}},
                    "end-date": None,
                    "role-title": "CEO"}}]},
            ]
        }
        rows = orcid._parse_employments("0000-0001", payload, "t")
        self.assertEqual(len(rows), 2)
        recent = orcid.most_recent_employer(pd.DataFrame(rows))
        self.assertEqual(recent.iloc[0]["employer_name"], "Moonshot AI")  # current wins
        self.assertTrue(bool(recent.iloc[0]["is_current"]))

    def test_clean_orcid_valid(self):
        # Real ORCID with a valid MOD 11-2 checksum (ORCID's own documented example).
        self.assertEqual(orcid._clean_orcid("https://orcid.org/0000-0002-1825-0097"),
                         "0000-0002-1825-0097")
        self.assertEqual(orcid._clean_orcid("0000-0002-1825-0097"), "0000-0002-1825-0097")

    def test_clean_orcid_rejects_garbage(self):
        # Placeholder / bad-checksum IDs (the 404 spam) are filtered out.
        self.assertIsNone(orcid._clean_orcid("0000-0000-0000-0000"))
        self.assertIsNone(orcid._clean_orcid("0000-0002-1234-5678"))  # bad checksum
        self.assertIsNone(orcid._clean_orcid("not-an-orcid"))
        self.assertIsNone(orcid._clean_orcid(""))

    def test_orcid_checksum_x(self):
        self.assertTrue(orcid.is_valid_orcid("0000-0002-1694-233X"))


class HomepageTests(unittest.TestCase):
    def test_domain_extraction(self):
        self.assertEqual(employer.homepage_domain("https://www.cs.cmu.edu/~rsalakhu/"), "cs.cmu.edu")
        self.assertEqual(employer.homepage_domain("openai.com/blog"), "openai.com")
        self.assertEqual(employer.homepage_domain("wired.com"), "wired.com")  # not "ired.com"

    def test_corporate_domain_maps_to_org(self):
        self.assertEqual(employer.employer_from_homepage("https://openai.com/x", {}), "openai")
        self.assertEqual(employer.employer_from_homepage("https://research.moonshot.ai", {}), "moonshot")
        self.assertIsNone(employer.employer_from_homepage("https://cs.cmu.edu/~x", {}))


class CascadeTests(unittest.TestCase):
    def test_priority_order(self):
        persons = pd.DataFrame([
            {"author_dblp_pid": "p1", "orcid": None, "homepage_url": "https://openai.com/x"},
            {"author_dblp_pid": "p2", "orcid": None, "homepage_url": None},
        ])
        # p2 has both a founder signal and an s2 signal -> founder (higher) must win.
        founders = pd.DataFrame([{"founder_dblp_pid": "p2", "org_id": "moonshot"}])
        s2 = pd.DataFrame([{"author_dblp_pid": "p2", "affiliations": "SomeUniv"}])
        out = employer.build_employer_table(
            persons,
            org_indexes=({}, {}),
            founders=founders,
            s2_metrics=s2,
        ).set_index("author_dblp_pid")
        self.assertEqual(out.loc["p1", "source"], "homepage")
        self.assertEqual(out.loc["p1", "employer_org_id"], "openai")
        self.assertEqual(out.loc["p2", "source"], "founder")
        self.assertEqual(out.loc["p2", "employer_org_id"], "moonshot")


if __name__ == "__main__":
    unittest.main()
