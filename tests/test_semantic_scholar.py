import unittest

import pandas as pd

from src.collect import semantic_scholar as s2


class ResolveAuthorIdsTests(unittest.TestCase):
    def test_matches_s2_author_to_dblp_pid_by_name(self):
        authorships = pd.DataFrame(
            [
                {"dblp_work_id": "w1", "author_dblp_pid": "200/8258", "author_name": "Zhilin Yang"},
                {"dblp_work_id": "w1", "author_dblp_pid": "s/RuslanSalakhutdinov", "author_name": "Ruslan Salakhutdinov"},
            ]
        )
        work_s2_authors = {
            "w1": [
                {"authorId": "111", "name": "Zhilin Yang"},
                {"authorId": "222", "name": "Ruslan Salakhutdinov"},
            ]
        }
        out = s2._resolve_author_ids(authorships, work_s2_authors, "dblp_work_id", "author_name")
        self.assertEqual(out["111"], "200/8258")
        self.assertEqual(out["222"], "s/RuslanSalakhutdinov")

    def test_ambiguous_name_is_skipped(self):
        # Two different PIDs share a normalized name within the work -> unsafe, skip.
        authorships = pd.DataFrame(
            [
                {"dblp_work_id": "w1", "author_dblp_pid": "wang/0001", "author_name": "Wei Wang"},
                {"dblp_work_id": "w1", "author_dblp_pid": "wang/0002", "author_name": "Wei Wang"},
            ]
        )
        work_s2_authors = {"w1": [{"authorId": "999", "name": "Wei Wang"}]}
        out = s2._resolve_author_ids(authorships, work_s2_authors, "dblp_work_id", "author_name")
        self.assertNotIn("999", out)

    def test_no_pid_column_returns_empty(self):
        authorships = pd.DataFrame([{"dblp_work_id": "w1", "author_name": "X"}])
        out = s2._resolve_author_ids(authorships, {"w1": []}, "dblp_work_id", "author_name")
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
