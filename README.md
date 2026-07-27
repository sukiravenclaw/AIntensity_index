# AI talent intensity index — collection pipeline v0.2

This repository collects publication records and publication-listed
affiliations only. It does not create scores, rankings, biographical fields, or
persisted person profiles.

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

- `data/processed/works.parquet` — publication records
- `data/processed/authorships.parquet` — author-work-institution rows with fractional credit
- `data/processed/mobility_events.parquet` — derived academia-industry transitions
- `data/processed/orgs_unmatched.csv` — QA: unmatched affiliation strings
- `data/processed/disambiguation_sample.csv` — QA: author name collision sample
- `data/processed/mobility_validation.csv` — QA: mobility vs CSRankings validation
- `REPORT.md` — collection summary with coverage and validation metrics

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

