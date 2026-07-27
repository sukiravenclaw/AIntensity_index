# Changes in v0.2

This document summarizes the changes made to upgrade the AI talent collection pipeline from v0.1 to v0.2, following the data collection spec for the AI talent intensity index.

## Summary

Version 0.2 adds:
- CSRankings data collection for faculty seeding and mobility validation
- Mobility event derivation from publication-listed affiliations
- Fractional credit calculation for multi-institution authorships
- Enhanced QA with mobility validation against CSRankings departure records

## New Files

### Collection modules
- **`src/collect/csrankings.py`** - Fetches CSRankings current and old records (CC BY-NC-ND 4.0)
  - Pulls from GitHub: csrankings.csv and old/old-csrankings.csv
  - Caches to `data/external/` (gitignored)
  - Used for faculty seeding and industry departure validation

### Derivation modules
- **`src/derive/__init__.py`** - Package init for derivation modules
- **`src/derive/mobility.py`** - Derives mobility events from authorships
  - Reconstructs per-author affiliation spells by sorting works chronologically
  - Emits transitions when institution changes
  - Scores confidence (low/medium/high) based on publication density

### QA modules
- **`src/qa/mobility_validation.py`** - Validates derived mobility against CSRankings
  - Compares academia-to-industry transitions
  - Reports matched, missed, and spurious transitions
  - Calculates precision and recall metrics

## Modified Files

### Configuration
- **`config.yaml`**
  - Added `paths.external: data/external` for third-party data
  - Added `csrankings.enabled: true`
  - Added `rate_limits.csrankings_seconds: 2.0`

- **`src/config.py`**
  - Updated `ensure_directories()` to create `external` directory

### Schemas
- **`src/schemas.py`**
  - Added `fractional_credit` field to `AUTHORSHIPS_SCHEMA` (float64)
  - Moved `n_institutions` before `fractional_credit` in schema
  - Added `fractional_credit` to numeric column processing in `write_parquet()`

### Data collection
- **`src/collect/openalex.py`**
  - Added `fractional_credit` calculation in `_parse_work()`: `1.0 / n_institutions`
  - Ensures no double-counting when same author has multiple institutions on one paper

### Pipeline orchestration
- **`src/pipeline.py`**
  - Added CSRankings collection step
  - Added mobility event derivation from authorships
  - Added mobility validation against CSRankings
  - Writes `mobility_events.parquet` to processed output
  - Writes `mobility_validation.csv` to processed output
  - Updated logging to include mobility event counts

### QA and reporting
- **`src/qa/report.py`**
  - Updated `write_report()` signature to accept `mobility_events` and `mobility_validation`
  - Added mobility validation section to REPORT.md with precision/recall metrics
  - Added `mobility_events.parquet` and `mobility_validation.csv` to row counts table

### Documentation
- **`README.md`**
  - Updated version to v0.2
  - Added "Attribution and licensing" section with CSRankings CC BY-NC-ND 4.0 notice
  - Updated outputs list with new parquet and CSV files
  - Added descriptions for mobility_events and mobility_validation outputs

- **`data/README.md`**
  - Added `data/external/` to list of generated directories
  - Documented that external directory contains third-party data (CSRankings)

- **`.gitignore`**
  - Added `data/external/` to ignore list

### Tests
- **`tests/test_pipeline.py`**
  - Updated `_authorship()` fixture to include `n_institutions` and `fractional_credit`
  - Updated test config to include `paths.external` and `csrankings.enabled`
  - Added `csrankings.collect` mock to end-to-end test
  - Updated expected outputs to include `mobility_events.parquet` and `mobility_validation.csv`

## New Outputs

The pipeline now produces:

1. **`data/processed/mobility_events.parquet`** - Derived mobility transitions
   - Columns: `author_openalex_id`, `from_org_id`, `to_org_id`, `from_institution_type`,
     `to_institution_type`, `transition_year`, `evidence_work_ids`, `n_works_before`,
     `n_works_after`, `confidence`

2. **`data/processed/mobility_validation.csv`** - QA validation report
   - Compares derived transitions to CSRankings departure records
   - Summary rows with precision, recall, matched, and missed counts
   - Detail rows for each CSRankings departure

3. **`data/interim/csrankings.parquet`** - Cached CSRankings records
   - Faculty with DBLP, ORCID, Google Scholar IDs
   - Industry departures with company names

## Schema Changes

### authorships.parquet
New fields added:
- `fractional_credit` (float64) - Credit weight: 1/n_institutions, prevents double-counting

Field reordering:
- `n_institutions` moved before `fractional_credit` (previously after `institution_country`)

## Hard Constraints Maintained

1. ✓ Public scholarly record only - CSRankings is public GitHub data
2. ✓ No biographical fields - No degree history, PhD institution, or advisor collected
3. ✓ No persisted person profiles - Mobility derived at analysis time from authorships
4. ✓ Licensing - CSRankings fetched at build time, CC BY-NC-ND 4.0 attribution in README
5. ✓ Rate limits and caching - CSRankings uses RawCacheClient with 2s delay

## Known Limitations

1. **Name matching** - Mobility validation uses simplified name matching (last name substring).
   Production should use DBLP/ORCID linking for higher precision.

2. **Industry classification** - Industry moves identified by keyword matching on company names.
   May miss some companies or misclassify.

3. **Confidence scoring** - Based solely on publication density (works before/after transition).
   Does not account for co-authorship patterns or affiliation string reliability.

## Migration Notes

To upgrade an existing v0.1 installation to v0.2:

1. Pull the latest code
2. No changes needed to `data/orgs.csv` format
3. `data/external/` directory will be auto-created
4. Re-run `make collect` to generate new outputs
5. Existing `works.parquet` and `authorships.parquet` will be regenerated with new schema
6. New `mobility_events.parquet` and `mobility_validation.csv` will be created

## Testing

All existing tests updated and passing (with dependencies installed):
- Organization matching tests
- Raw cache tests
- OpenAlex normalization tests
- Output schema tests
- End-to-end fixture test

Run with: `make test`
