import unittest

import pandas as pd

from src.collect import dblp, dblp_person


class SignaturePositionTests(unittest.TestCase):
    def test_first_middle_last_from_ordinal(self):
        rows = [
            {"dblp_work_id": "w1", "author_dblp_pid": "a", "author_name": "A",
             "author_ordinal": 1, "retrieved_at": "t", "source": "dblp"},
            {"dblp_work_id": "w1", "author_dblp_pid": "b", "author_name": "B",
             "author_ordinal": 2, "retrieved_at": "t", "source": "dblp"},
            {"dblp_work_id": "w1", "author_dblp_pid": "c", "author_name": "C",
             "author_ordinal": 3, "retrieved_at": "t", "source": "dblp"},
        ]
        out = dblp._finalize_authorships(rows).set_index("author_dblp_pid")
        self.assertEqual(out.loc["a", "author_position"], "first")
        self.assertEqual(out.loc["b", "author_position"], "middle")
        self.assertEqual(out.loc["c", "author_position"], "last")

    def test_single_author_is_first_not_last(self):
        rows = [
            {"dblp_work_id": "w1", "author_dblp_pid": "a", "author_name": "A",
             "author_ordinal": 1, "retrieved_at": "t", "source": "dblp"},
        ]
        out = dblp._finalize_authorships(rows)
        self.assertEqual(out.iloc[0]["author_position"], "first")

    def test_pid_extraction(self):
        self.assertEqual(dblp._pid("https://dblp.org/pid/78/1234"), "78/1234")
        self.assertEqual(dblp._pid("https://dblp.org/pid/l/YannLeCun"), "l/YannLeCun")
        self.assertIsNone(dblp._pid(""))


PERSON_XML = b"""<?xml version="1.0"?>
<dblpperson name="Zhilin Yang" pid="200/8258">
  <r><inproceedings key="conf/nips/YangDYCSL19">
    <author pid="200/8258">Zhilin Yang</author>
    <author pid="z/ZihangDai">Zihang Dai</author>
    <author pid="s/RuslanSalakhutdinov">Ruslan Salakhutdinov</author>
    <title>XLNet.</title>
    <year>2019</year>
    <booktitle>NeurIPS</booktitle>
  </inproceedings></r>
  <r><inproceedings key="conf/acl/YangCS16">
    <author pid="200/8258">Zhilin Yang</author>
    <author pid="s/RuslanSalakhutdinov">Ruslan Salakhutdinov</author>
    <title>Multi-task learning.</title>
    <year>2016</year>
    <booktitle>ACL</booktitle>
  </inproceedings></r>
</dblpperson>
"""


class PersonHistoryTests(unittest.TestCase):
    def test_parse_person_xml(self):
        rows = dblp_person._parse_person_xml(PERSON_XML, "2026-01-01")
        frame = dblp_person._finalize(pd.DataFrame(rows).to_dict("records"))
        # Two works, Salakhutdinov present as a co-author on both.
        self.assertEqual(frame["dblp_work_id"].nunique(), 2)
        self.assertIn("s/RuslanSalakhutdinov", set(frame["author_dblp_pid"]))
        # XLNet: Yang first, Salakhutdinov last.
        xlnet = frame[frame["dblp_work_id"].str.endswith("YangDYCSL19")].set_index("author_dblp_pid")
        self.assertEqual(xlnet.loc["200/8258", "author_position"], "first")
        self.assertEqual(xlnet.loc["s/RuslanSalakhutdinov", "author_position"], "last")
        self.assertEqual(int(xlnet.loc["200/8258", "publication_year"]), 2019)
        self.assertEqual(xlnet.loc["200/8258", "venue_raw"], "NeurIPS")

    def test_work_id_matches_rec_uri_space(self):
        rows = dblp_person._parse_person_xml(PERSON_XML, "t")
        ids = {r["dblp_work_id"] for r in rows}
        self.assertIn("https://dblp.org/rec/conf/nips/YangDYCSL19", ids)


if __name__ == "__main__":
    unittest.main()
