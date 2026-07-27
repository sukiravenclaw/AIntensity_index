from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.collect.openalex import apply_metric_fallbacks, _parse_work
from src.common import RawCacheClient
from src.normalize.orgs import OrganizationFileError, match_organizations
from src.qa.disambiguation import build_sample
from src.schemas import AUTHORSHIPS_SCHEMA, WORKS_SCHEMA, write_parquet
from src.pipeline import run


class OrganizationMatchingTests(unittest.TestCase):
    def test_ror_then_alias_and_unmatched(self):
        with tempfile.TemporaryDirectory() as directory:
            org_path = Path(directory) / "orgs.csv"
            org_path.write_text(
                "org_id,display_name,ror,aliases\n"
                "o1,Example AI Lab,https://ror.org/012345678,Example Research\n"
                "o2,Second Institute,,Second Inst|Institute Two\n",
                encoding="utf-8",
            )
            authorships = pd.DataFrame(
                [
                    _authorship("w1", "https://ror.org/012345678", "Department, Example AI Lab"),
                    _authorship("w2", None, "Second Inst"),
                    _authorship("w3", None, "Unknown Place"),
                ]
            )
            matched, unmatched = match_organizations(authorships, org_path)
            self.assertEqual(matched["org_id"].tolist()[:2], ["o1", "o2"])
            self.assertTrue(pd.isna(matched["org_id"].iloc[2]))
            self.assertEqual(unmatched["raw_affiliation_string"].tolist(), ["Unknown Place"])

    def test_empty_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            org_path = Path(directory) / "orgs.csv"
            org_path.write_text("org_id,display_name,ror,aliases\n", encoding="utf-8")
            with self.assertRaises(OrganizationFileError):
                match_organizations(pd.DataFrame([_authorship("w", None, "X")]), org_path)


class RawCacheTests(unittest.TestCase):
    def test_response_is_persisted_and_second_call_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            response = requests.Response()
            response.status_code = 200
            response._content = b'{"ok": true}'
            response.url = "https://example.test/data"
            client = RawCacheClient(
                "example",
                directory,
                "test-agent (mailto:test@example.org)",
                0,
            )
            client.session.request = Mock(return_value=response)
            first = client.request(
                "GET",
                "https://example.test/data",
                params={"api_key": "secret", "q": "public"},
            )
            second = client.request(
                "GET",
                "https://example.test/data",
                params={"api_key": "different-secret", "q": "public"},
            )
            self.assertEqual(first.json(), {"ok": True})
            self.assertEqual(second.json(), {"ok": True})
            self.assertEqual(client.session.request.call_count, 1)
            metadata = list(Path(directory).rglob("metadata.json"))
            self.assertTrue(metadata)
            self.assertNotIn("secret", metadata[0].read_text(encoding="utf-8"))


class OpenAlexNormalizationTests(unittest.TestCase):
    def test_one_row_per_author_institution_and_raw_only_affiliation(self):
        work = {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1/test",
            "title": "A Test",
            "publication_year": 2024,
            "publication_date": "2024-01-02",
            "cited_by_count": 3,
            "fwci": 1.2,
            "citation_normalized_percentile": {"value": 0.8},
            "topics": [{"display_name": "Machine Learning"}],
            "primary_topic": {"display_name": "Machine Learning"},
            "authorships": [
                {
                    "author": {
                        "id": "https://openalex.org/A1",
                        "display_name": "Alex Example",
                        "orcid": "https://orcid.org/0000-0000-0000-0001",
                    },
                    "author_position": "first",
                    "raw_affiliation_strings": ["Example Lab", "Second Institute"],
                    "institutions": [
                        {
                            "display_name": "Example Lab",
                            "ror": "https://ror.org/012345678",
                            "type": "facility",
                            "country_code": "US",
                        }
                    ],
                }
            ],
            "locations": [
                {"landing_page_url": "https://arxiv.org/abs/2401.01234v2"}
            ],
            "primary_location": {"source": {"display_name": "Example Proceedings"}},
        }
        seed = pd.Series(
            {
                "venue_normalized": "ICML",
                "venue_raw": "ICML",
                "decision": "poster",
                "arxiv_id": None,
                "openreview_id": "or1",
            }
        )
        cfg = {"venues": {"ICML": {"tier": "A*"}}}
        parsed_work, authorships = _parse_work(
            work, seed, "2026-01-01T00:00:00+00:00", cfg
        )
        self.assertEqual(parsed_work["arxiv_id"], "2401.01234")
        self.assertEqual(len(authorships), 2)
        self.assertEqual({row["n_institutions"] for row in authorships}, {2})
        self.assertIsNone(authorships[1]["institution_ror"])

    def test_local_metrics_only_when_api_field_is_absent(self):
        frame = pd.DataFrame(
            [
                {
                    "publication_year": 2024,
                    "primary_topic": "ML",
                    "cited_by_count": 1,
                    "fwci": None,
                    "citation_normalized_percentile": None,
                    "source": "openalex",
                    "_fwci_field_present": False,
                    "_percentile_field_present": False,
                },
                {
                    "publication_year": 2024,
                    "primary_topic": "ML",
                    "cited_by_count": 3,
                    "fwci": None,
                    "citation_normalized_percentile": None,
                    "source": "openalex",
                    "_fwci_field_present": False,
                    "_percentile_field_present": False,
                },
            ]
        )
        result = apply_metric_fallbacks(frame)
        self.assertEqual(result["fwci"].round(2).tolist(), [0.5, 1.5])
        self.assertEqual(result["citation_normalized_percentile"].tolist(), [0.5, 1.0])
        self.assertTrue(result["source"].str.contains("local_year_topic_metrics").all())


class OutputTests(unittest.TestCase):
    def test_typed_empty_parquet_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            works_path = Path(directory) / "works.parquet"
            authorships_path = Path(directory) / "authorships.parquet"
            write_parquet(pd.DataFrame(), works_path, WORKS_SCHEMA)
            write_parquet(pd.DataFrame(), authorships_path, AUTHORSHIPS_SCHEMA)
            self.assertEqual(pd.read_parquet(works_path).columns.tolist(), WORKS_SCHEMA.names)
            self.assertEqual(
                pd.read_parquet(authorships_path).columns.tolist(),
                AUTHORSHIPS_SCHEMA.names,
            )

    def test_disambiguation_is_long_format_without_people_table(self):
        works = pd.DataFrame(
            [
                {
                    "work_id": f"w{index}",
                    "title": f"Paper {index}",
                    "publication_year": 2022 + index,
                    "venue_normalized": "ICML",
                }
                for index in range(3)
            ]
        )
        authorships = pd.DataFrame(
            [
                {
                    **_authorship(f"w{index}", None, f"Affiliation {index}"),
                    "author_openalex_id": "a1",
                    "author_display_name": "Same Name",
                }
                for index in range(3)
            ]
            + [
                {
                    **_authorship("w0", None, "Other"),
                    "author_openalex_id": "a2",
                    "author_display_name": "Same Name",
                }
            ]
        )
        sample = build_sample(
            works,
            authorships,
            n_authors=100,
            minimum_works=3,
            random_seed=1,
        )
        self.assertEqual(len(sample), 3)
        self.assertEqual(sample["author_openalex_id"].unique().tolist(), ["a1"])
        self.assertTrue(sample["name_collision_candidates"].str.contains("a2").all())

    def test_fixture_backed_end_to_end_writes_all_deliverables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "orgs.csv").write_text(
                "org_id,display_name,ror,aliases\n"
                "o1,Example Lab,https://ror.org/012345678,Example Research\n",
                encoding="utf-8",
            )
            cfg = {
                "paths": {
                    "raw": str(root / "raw"),
                    "external": str(root / "external"),
                    "interim": str(root / "interim"),
                    "processed": str(root / "processed"),
                    "orgs": str(root / "orgs.csv"),
                    "report": str(root / "REPORT.md"),
                },
                "qa": {"disambiguation_authors": 100, "minimum_works": 3},
                "project": {"random_seed": 7},
                "semantic_scholar": {"enabled": False},
                "csrankings": {"enabled": True},
            }
            works = pd.DataFrame(
                [
                    {
                        "work_id": "https://openalex.org/W1",
                        "doi": "10.1/test",
                        "arxiv_id": "2401.00001",
                        "openreview_id": "or1",
                        "s2_paper_id": None,
                        "title": "A Test",
                        "publication_year": 2024,
                        "publication_date": "2024-01-02",
                        "venue_raw": "ICML",
                        "venue_normalized": "ICML",
                        "venue_tier": "A*",
                        "decision": "poster",
                        "cited_by_count": 2,
                        "fwci": 1.0,
                        "citation_normalized_percentile": 0.5,
                        "topics": ["Machine Learning"],
                        "primary_topic": "Machine Learning",
                        "n_authors": 1,
                        "retrieved_at": "2026-01-01T00:00:00+00:00",
                        "source": "openalex",
                        "_fwci_field_present": True,
                        "_percentile_field_present": True,
                    }
                ]
            )
            authorships = pd.DataFrame(
                [
                    {
                        **_authorship(
                            "https://openalex.org/W1",
                            "https://ror.org/012345678",
                            "Example Lab",
                        ),
                        "author_openalex_id": "https://openalex.org/A1",
                        "author_display_name": "Alex Example",
                    }
                ]
            )
            empty_seed = pd.DataFrame(columns=["source"])
            with patch("src.pipeline.dblp.collect", return_value=empty_seed), patch(
                "src.pipeline.openreview.collect", return_value=empty_seed
            ), patch("src.pipeline.arxiv.collect", return_value=empty_seed), patch(
                "src.pipeline.csrankings.collect", return_value=empty_seed
            ), patch(
                "src.pipeline.openalex.collect", return_value=(works, authorships)
            ):
                run(cfg)
            expected = {
                "works.parquet",
                "authorships.parquet",
                "mobility_events.parquet",
                "orgs_unmatched.csv",
                "disambiguation_sample.csv",
                "mobility_validation.csv",
            }
            self.assertEqual(
                {path.name for path in (root / "processed").iterdir()},
                expected,
            )
            self.assertTrue((root / "REPORT.md").exists())
            stored = pd.read_parquet(root / "processed" / "authorships.parquet")
            self.assertEqual(stored["org_id"].tolist(), ["o1"])


def _authorship(work_id, ror, raw):
    return {
        "work_id": work_id,
        "author_openalex_id": "a",
        "author_s2_id": None,
        "orcid": None,
        "author_display_name": "Author",
        "author_position": "first",
        "n_authors": 1,
        "n_institutions": 1,
        "fractional_credit": 1.0,
        "raw_affiliation_string": raw,
        "institution_ror": ror,
        "institution_display_name": None,
        "institution_type": None,
        "institution_country": None,
        "org_id": None,
        "retrieved_at": "2026-01-01T00:00:00+00:00",
        "source": "openalex",
    }


if __name__ == "__main__":
    unittest.main()
