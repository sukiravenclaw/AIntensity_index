# AI talent intensity index — pipeline v0.3 (OpenAlex-free, person-centric)

This pipeline identifies AI talent (authors at top venues in 2024–2025),
backtracks each person to their highest-education training lab and advisor,
maps lab → current-employer flows, flags founders, and computes a per-company
**talent-intensity index** weighted by citations and publications.

**Architecture (OpenAlex is not used).** Institution is a property of the
*person*, reconstructed from the DBLP full-publication history plus the
CSRankings faculty co-authorship graph:

- **DBLP** — person identity (`author_dblp_pid`), venue-year seed, co-authorship
  graph, and full per-person history (`src/collect/dblp.py`, `dblp_person.py`).
- **CSRankings** — authoritative faculty → institution anchor (`csrankings.py`).
- **Semantic Scholar** — paper + author citation metrics (`semantic_scholar.py`).
- **ORCID public API + Wikidata** — current employer and founders
  (`orcid.py`, `wikidata.py`).

Derivation lives in `src/derive/talent_flow.py` (persons, training/advisor,
employer cascade, flow matrix) and `src/derive/intensity.py` (the index). The
legacy affiliation-timeline prototype `src/flow/derive_talent_flow.py` is
superseded and no longer wired into the pipeline.

## Attribution and licensing

This pipeline uses [CSRankings](https://github.com/emeryberger/CSrankings) data
for faculty seeding and mobility validation. CSRankings is licensed under
**CC BY-NC-ND 4.0** (NoDerivatives). The pipeline fetches CSRankings data at
build time and does not redistribute modified versions. CSRankings files are
cached in `data/external/` (gitignored) for validation purposes only.

## Inputs and credentials

1. Replace the header-only `data/orgs.csv` with the hand-curated organization
   file. The expected columns are documented in `data/README.md`.
2. Set a monitored contact email:

   ```sh
   export CONTACT_EMAIL=research-data@example.org
   ```

3. Set an OpenAlex API key. OpenAlex's current documentation describes a free
   key and only a very small anonymous allowance:

   ```sh
   export OPENALEX_API_KEY=...
   ```

4. Optionally set `SEMANTIC_SCHOLAR_API_KEY`. Without it, the Semantic Scholar
   enrichment stage is skipped and the rest of the pipeline still runs.

Install dependencies in an isolated environment, then collect:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make collect
```

## Outputs

The end-to-end target writes:

- `data/processed/works.parquet` — seed publications with S2 citation counts
- `data/processed/authorships.parquet` — work → DBLP-PID authorship rows
- `data/processed/flow/persons.parquet` — the talent set with career metrics
- `data/processed/flow/person_training.parquet` — highest-education lab + advisor
- `data/processed/flow/person_employer.parquet` — current employer (cascade + founder flag)
- `data/processed/flow/flow_matrix.parquet` / `flow_matrix_published.csv` — lab → employer flows
- `data/processed/flow/company_intensity.csv` — the per-company intensity index
- `data/processed/flow/intensity_lab_decomposition.csv` — each company's intensity by source lab
- `REPORT.md`, `flow/FLOW_REPORT.md`, `flow/INTENSITY_REPORT.md` — summaries

Every HTTP response is stored under `data/raw/<source>/<content-hash>/` before
status handling or parsing. A successful response also gets a canonical cache
entry, so reruns skip network calls. HTTP 429 and server errors honor
`Retry-After` or use exponential backoff.

Interim Parquet files preserve the DBLP venue seeds, OpenReview decisions and
review-score aggregates, arXiv metadata, and Semantic Scholar crosswalk.
arXiv author affiliations are intentionally never parsed.

## Reproducibility and scope

`config.yaml` contains the date window, venue aliases and tiers, OpenReview
group patterns, source switches, request spacing, retry policy, and QA seed.
`window.end: null` resolves to the collection date. Pin it to a date for a
bit-for-bit stable scope.

Organization matching is deterministic:

1. exact normalized ROR to `org_id`;
2. exact normalized publication affiliation or OpenAlex institution display
   name to a curated alias;
3. no match.

There is no fuzzy organization matching and no LLM adjudication.

OpenAlex is the required work-ID spine. Public OpenReview rejects that cannot
be resolved to an OpenAlex work are retained in the interim source table but
not assigned a synthetic `work_id`.

Run offline unit tests with:

```sh
make test
```

