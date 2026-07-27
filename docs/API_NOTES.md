# API surface verification

Verified against official documentation on 2026-07-26.

- OpenAlex: the current work response still includes `fwci` and
  `citation_normalized_percentile`; the latter is an object whose numeric
  percentile is in `value`. Work authorships include author position,
  publication-level raw affiliation strings, and normalized institutions with
  ROR. The current API documentation requests an `api_key` and describes only
  a very small anonymous allowance.
  - <https://developers.openalex.org/api-reference/works/get-a-single-work>
  - <https://developers.openalex.org/llms.txt>
- OpenReview: current venues use API v2 notes. Submissions are queried by their
  invitation, and reviews/decisions are read from replies.
  - <https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc>
  - <https://docs.openreview.net/reference/api-v2/openapi-definition>
- Semantic Scholar: the Graph API paper-batch endpoint accepts up to 500 paper
  IDs and exposes paper IDs, external IDs, citation counts, and authors.
  - <https://api.semanticscholar.org/api-docs/graph>
- arXiv: the metadata API is a paged Atom feed. The collector uses monthly,
  per-category slices and a three-second request interval. It does not parse
  author affiliation elements.
  - <https://info.arxiv.org/help/api/user-manual.html>
- DBLP: the public knowledge graph exposes `publishedInStream`,
  `yearOfPublication`, `title`, and DOI; automated clients should space
  requests and honor HTTP 429 `Retry-After`.
  - <https://dblp.org/rdf/docu/>
  - <https://dblp.org/faq/How%2Bto%2Buse%2Bthe%2Bdblp%2Bsearch%2BAPI>
  - <https://dblp.org/faq/Am%2BI%2Ballowed%2Bto%2Bcrawl%2Bthe%2Bdblp%2Bwebsite.html>

