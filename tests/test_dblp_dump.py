import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.collect import dblp_dump

# A miniature dblp.xml (no custom entities / DOCTYPE, so it parses without the DTD).
DUMP_XML = b"""<?xml version="1.0"?>
<dblp>
<inproceedings key="conf/nips/YangS16">
  <author>Zhilin Yang</author>
  <author orcid="0000-0001-2345-6789">Ruslan Salakhutdinov</author>
  <title>Early CMU work.</title>
  <year>2016</year>
  <booktitle>NeurIPS</booktitle>
</inproceedings>
<article key="journals/x/Unrelated19">
  <author>Someone Else</author>
  <year>2019</year>
  <journal>JMLR</journal>
</article>
</dblp>
"""


class ParseDumpTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.xml = Path(self.dir) / "dblp.xml"
        self.xml.write_bytes(DUMP_XML)

    def test_only_pubs_touching_a_target_are_emitted(self):
        target_names = {"Zhilin Yang"}
        name_to_pid = {"Zhilin Yang": "yang", "Ruslan Salakhutdinov": "russ"}
        rows, matched = dblp_dump._parse_dump(self.xml, target_names, name_to_pid)
        frame = dblp_dump._finalize(rows)
        self.assertEqual(matched, 1)  # unrelated article skipped
        self.assertEqual(frame["dblp_work_id"].nunique(), 1)
        by_pid = frame.set_index("author_dblp_pid")
        self.assertEqual(by_pid.loc["yang", "author_position"], "first")
        self.assertEqual(by_pid.loc["russ", "author_position"], "last")
        self.assertEqual(by_pid.loc["russ", "author_orcid"], "0000-0001-2345-6789")
        self.assertEqual(int(by_pid.loc["yang", "publication_year"]), 2016)
        self.assertEqual(by_pid.loc["yang", "venue_raw"], "NeurIPS")

    def test_columns_match_dblp_person_history(self):
        from src.collect.dblp_person import HISTORY_COLUMNS
        rows, _ = dblp_dump._parse_dump(self.xml, {"Zhilin Yang"}, {"Zhilin Yang": "yang"})
        frame = dblp_dump._finalize(rows)
        self.assertEqual(list(frame.columns), HISTORY_COLUMNS)


if __name__ == "__main__":
    unittest.main()
