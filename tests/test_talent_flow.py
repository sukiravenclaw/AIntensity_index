import unittest

import pandas as pd

from src.derive import talent_flow as tf
from src.normalize.org_taxonomy import OrgTaxonomy


def _taxonomy():
    return OrgTaxonomy(
        org_type_by_id={"cmu": "academic", "moonshot": "industry_lab"},
        parent_by_id={},
    )


class TalentFlowGoldenLogicTests(unittest.TestCase):
    """Synthetic Salakhutdinov -> Zhilin Yang -> Moonshot chain."""

    def setUp(self):
        # Yang's full history: two CMU-era papers (2016, 2017) with Salakhutdinov
        # as last author, plus a 2024 seed paper.
        self.history = pd.DataFrame([
            # 2016 paper
            {"dblp_work_id": "w2016", "author_dblp_pid": "yang", "author_name": "Zhilin Yang",
             "author_position": "first", "publication_year": 2016},
            {"dblp_work_id": "w2016", "author_dblp_pid": "russ", "author_name": "Ruslan Salakhutdinov",
             "author_position": "last", "publication_year": 2016},
            # 2017 paper
            {"dblp_work_id": "w2017", "author_dblp_pid": "yang", "author_name": "Zhilin Yang",
             "author_position": "first", "publication_year": 2017},
            {"dblp_work_id": "w2017", "author_dblp_pid": "russ", "author_name": "Ruslan Salakhutdinov",
             "author_position": "last", "publication_year": 2017},
            # 2024 seed paper (solo-ish, no faculty)
            {"dblp_work_id": "w2024", "author_dblp_pid": "yang", "author_name": "Zhilin Yang",
             "author_position": "first", "publication_year": 2024},
        ])
        self.seed = pd.DataFrame([
            {"work_id": "w2024", "author_dblp_pid": "yang", "author_name": "Zhilin Yang"},
        ])
        self.faculty = pd.DataFrame([
            {"name": "Ruslan Salakhutdinov", "affiliation": "Carnegie Mellon University",
             "normalized_name": "ruslan salakhutdinov", "org_id": "cmu", "orcid": None, "homepage": None},
        ])

    def test_persons_and_training(self):
        persons = tf.build_persons(self.seed, self.history, faculty=self.faculty)
        self.assertEqual(len(persons), 1)
        self.assertEqual(int(persons.iloc[0]["first_pub_year"]), 2016)

        faculty_map = tf.build_faculty_map(self.faculty)
        training = tf.derive_training(persons, self.history, faculty_map)
        self.assertEqual(len(training), 1)
        row = training.iloc[0]
        self.assertEqual(row["training_org_id"], "cmu")
        self.assertEqual(row["advisor_dblp_pid"], "russ")
        self.assertEqual(row["n_advisor_copubs"], 2)

    def test_founder_employer_and_flow_edge(self):
        persons = tf.build_persons(self.seed, self.history, faculty=self.faculty)
        faculty_map = tf.build_faculty_map(self.faculty)
        training = tf.derive_training(persons, self.history, faculty_map)

        founders = pd.DataFrame([{
            "org_id": "moonshot", "company_label": "Moonshot AI", "founder_name": "Zhilin Yang",
            "founder_dblp_pid": "yang", "founder_orcid": None, "founder_qid": "Q1",
        }])
        employers = tf.derive_employers(persons, ({}, {}), founders=founders)
        yang = employers.set_index("author_dblp_pid").loc["yang"]
        self.assertEqual(yang["employer_org_id"], "moonshot")
        self.assertEqual(yang["source"], "founder")
        self.assertTrue(bool(yang["is_founder"]))

        full, published = tf.build_flow_matrix(training, employers, _taxonomy())
        # One cell: (cmu, russ) -> moonshot, n=1, founder-exempt from suppression.
        self.assertEqual(len(published), 1)
        cell = published.iloc[0]
        self.assertEqual(cell["training_org_id"], "cmu")
        self.assertEqual(cell["employer_org_id"], "moonshot")
        self.assertTrue(bool(cell["is_industry"]))
        self.assertEqual(int(cell["n_founders"]), 1)
        self.assertFalse(bool(cell["suppressed"]))

    def test_coverage_cascade(self):
        persons = tf.build_persons(self.seed, self.history, faculty=self.faculty)
        faculty_map = tf.build_faculty_map(self.faculty)
        training = tf.derive_training(persons, self.history, faculty_map)
        founders = pd.DataFrame([{"org_id": "moonshot", "founder_name": "Zhilin Yang",
                                  "founder_dblp_pid": "yang", "founder_orcid": None}])
        employers = tf.derive_employers(persons, ({}, {}), founders=founders)
        cov = tf.coverage_cascade(persons, training, employers)
        self.assertEqual(cov["total_talent"], 1)
        self.assertEqual(cov["with_training"], 1)
        self.assertEqual(cov["founders"], 1)


if __name__ == "__main__":
    unittest.main()
