# Collection report

This report describes collection outputs only. It contains no scoring or modeling.

## Row counts

| Table | Rows |
|---|---:|
| `works.parquet` | 19,544 |
| `authorships.parquet` | 112,104 |
| `mobility_events.parquet` | 37 |
| `orgs_unmatched.csv` | 8,217 |
| `disambiguation_sample.csv` | 508 (100 authors) |
| `mobility_validation.csv` | 0 |

## Organization mapping

1,829 of 112,104 authorship-institution rows mapped to an `org_id` (1.63%).

## Coverage by venue and year

| venue_normalized | publication_year | works |
|---|---|---|
| AAAI | 2024 | 3261 |
| ACL | 2024 | 1667 |
| CVPR | 2024 | 5127 |
| ECCV | 2024 | 52 |
| EMNLP | 2024 | 4122 |
| ICLR | 2024 | 550 |
| ICML | 2024 | 1524 |
| NeurIPS | 2024 | 3241 |

## Mobility validation

Mobility validation skipped (CSRankings not available).

## Top 50 unmatched affiliation strings

| raw_affiliation_string | frequency |
|---|---|
| Zhejiang University | 479 |
| University of Science and Technology of China | 455 |
| Tsinghua University | 379 |
| Shanghai Jiao Tong University | 351 |
| Shanghai AI Laboratory | 312 |
| KAIST | 298 |
| Carnegie Mellon University | 220 |
| National University of Singapore | 214 |
| University of Electronic Science and Technology of China | 189 |
| Peking University | 182 |
| Nanyang Technological University | 173 |
| Adobe Research | 163 |
| Shanghai Artificial Intelligence Laboratory | 155 |
| Technical University of Munich | 148 |
| Beijing University of Posts and Telecommunications | 145 |
| The Chinese University of Hong Kong | 138 |
| Ant Group | 132 |
| Harbin Institute of Technology | 129 |
| Huazhong University of Science and Technology | 129 |
| IBM Research | 129 |
| Fudan University | 128 |
| Monash University | 128 |
| NVIDIA | 128 |
| Nanjing University | 127 |
| Beihang University | 125 |
| Northwestern Polytechnical University | 125 |
| University of Southern California | 122 |
| Xidian University | 122 |
| Beijing Institute of Technology | 121 |
| The University of Hong Kong | 116 |
| Wuhan University | 115 |
| South China University of Technology | 106 |
| Purdue University | 105 |
| Huawei Noah&#x0027;s Ark Lab | 104 |
| ShanghaiTech University | 104 |
| AWS AI Labs | 103 |
| POSTECH | 100 |
| Seoul National University | 98 |
| Arizona State University | 92 |
| Shenzhen International Graduate School, Tsinghua University | 92 |
| School of Computer Science, Peking University,National Key Laboratory for Multimedia Information Processing | 90 |
| Tencent AI Lab | 87 |
| Xiamen University | 87 |
| Dalian University of Technology | 85 |
| City University of Hong Kong | 83 |
| ETH Z&#x00FC;rich | 83 |
| Tianjin University | 82 |
| University of Illinois Urbana-Champaign | 77 |
| Xi'an Jiaotong University | 77 |
| The Hong Kong University of Science and Technology | 76 |

## Collection notes

- The publication window and venue tiers are configuration values in `config.yaml`.
- OpenAlex authorships are the only persisted person-linked publication records; no people/profile table is produced.
- Semantic Scholar is used only for paper/author cross-reference IDs and missing citation counts.
- arXiv affiliation fields are not parsed or used.
- OpenAlex currently documents both `fwci` and `citation_normalized_percentile`. If either field disappears from all returned work objects, the pipeline computes a collection-local year-and-primary-topic fallback and records that in `source`.
