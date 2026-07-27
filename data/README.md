# Data inputs

Replace `orgs.csv` with the hand-curated organization file before collection.
Required columns are `org_id` and at least one of `ror`, `display_name`, or
`aliases`. Put multiple aliases in one cell separated by `|`.

`data/raw/`, `data/external/`, `data/interim/`, and `data/processed/` are
generated and ignored by Git. The raw directory is content-addressed and is
always written before a response is parsed. The external directory contains
third-party data fetched at build time (e.g., CSRankings).

