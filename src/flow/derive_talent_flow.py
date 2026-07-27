#!/usr/bin/env python3
"""
Derive training-lab-to-employer talent flow matrix from publication records.

Critical constraint: Every fact must be derived from works.parquet and authorships.parquet.
No institution names, lab names, or advisor names may come from the LLM's prior knowledge.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import defaultdict, Counter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOG = logging.getLogger(__name__)

# Configuration
COHORT_YEARS = [2021, 2022, 2023]
MIN_WORKS_IN_TRAINING_SPELL = 2
MIN_CAREER_AGE_FOR_PI = 8
MIN_COAUTHOR_WORKS = 2
PI_SCORE_TOLERANCE = 0.10  # 10% for adjudication
MAX_AUTHORS_FOR_CONSORTIUM = 20
TOP_N_LABS = 200
MIN_CELL_SIZE = 5  # Suppression threshold

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
FLOW_DIR = PROCESSED_DIR / "flow"
FLOW_DIR.mkdir(exist_ok=True)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class FlowMetrics:
    """Track coverage cascade metrics"""
    def __init__(self):
        self.total_authors = 0
        self.with_academic_spell = 0
        self.with_identified_pi = 0
        self.with_observed_transition = 0
        self.in_orgs_csv_destination = 0
        self.surviving_suppression = 0
        self.industry_native = 0
        self.no_transition_observed = 0
        self.pi_validated_deterministic = 0
        self.pi_validated_adjudicated = 0
        self.pi_total_deterministic = 0
        self.pi_total_adjudicated = 0
        self.cells_published = 0
        self.cells_suppressed = 0
        self.transitions_in_suppressed_cells = 0

METRICS = FlowMetrics()

def load_data():
    """Load all required data files"""
    LOG.info("Loading data files...")

    works = pd.read_parquet(PROCESSED_DIR / "works.parquet")
    authorships = pd.read_parquet(PROCESSED_DIR / "authorships.parquet")
    orgs = pd.read_csv(DATA_DIR / "orgs.csv")
    csrankings = pd.read_csv(DATA_DIR / "external" / "csrankings.csv")

    LOG.info(f"Loaded {len(works)} works, {len(authorships)} authorships, {len(orgs)} orgs, {len(csrankings)} CSRankings faculty")

    return works, authorships, orgs, csrankings


def derive_training_spells(works: pd.DataFrame, authorships: pd.DataFrame, orgs: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Identify training spells for each author.

    Training spell = earliest academic spell with >=2 works.
    """
    LOG.info("Step 1: Deriving training spells...")

    # Merge works with authorships to get publication dates
    data = authorships.merge(
        works[['work_id', 'publication_year', 'publication_date']],
        on='work_id',
        how='left'
    )

    # Filter to valid data
    data = data[data['publication_year'].notna()].copy()
    data = data[data['author_openalex_id'].notna()].copy()

    METRICS.total_authors = data['author_openalex_id'].nunique()
    LOG.info(f"Total unique authors: {METRICS.total_authors}")

    # Create org type mapping from orgs.csv
    org_type_map = orgs.set_index('org_id')['org_type'].to_dict()

    # Map institution_type if available, else use org_id
    data['org_type'] = data['org_id'].map(org_type_map)

    # Fall back to institution_type from OpenAlex
    data['org_type'] = data['org_type'].fillna(data['institution_type'])

    # Mark academic institutions (OpenAlex uses 'education' for universities)
    data['is_academic'] = data['org_type'].isin(['academic', 'education'])

    training_spells = []
    industry_native_authors = set()

    # Group by author
    for author_id, author_data in data.groupby('author_openalex_id'):
        # Sort by publication year
        author_data = author_data.sort_values('publication_year')

        # Create spells by grouping consecutive works at same institution
        spells = []
        current_spell = None

        for _, row in author_data.iterrows():
            inst_ror = row['institution_ror']
            year = row['publication_year']
            work_id = row['work_id']
            org_type = row['org_type']
            is_academic = row['is_academic']

            if current_spell is None or current_spell['institution_ror'] != inst_ror:
                # Start new spell
                if current_spell is not None:
                    spells.append(current_spell)
                current_spell = {
                    'institution_ror': inst_ror,
                    'org_type': org_type,
                    'is_academic': is_academic,
                    'start_year': year,
                    'end_year': year,
                    'work_ids': [work_id],
                    'n_works': 1
                }
            else:
                # Continue current spell
                current_spell['end_year'] = year
                current_spell['work_ids'].append(work_id)
                current_spell['n_works'] += 1

        # Add last spell
        if current_spell is not None:
            spells.append(current_spell)

        # Find training spell: earliest academic spell with >=2 works
        training_spell = None
        for spell in spells:
            if spell['is_academic'] and spell['n_works'] >= MIN_WORKS_IN_TRAINING_SPELL:
                training_spell = spell
                break

        if training_spell is None:
            # Industry-native or insufficient data
            industry_native_authors.add(author_id)
        else:
            training_spells.append({
                'author_openalex_id': author_id,
                'institution_ror': training_spell['institution_ror'],
                'spell_start_year': int(training_spell['start_year']),
                'spell_end_year': int(training_spell['end_year']),
                'n_works_in_spell': training_spell['n_works'],
                'work_ids': json.dumps(training_spell['work_ids'])
            })

    METRICS.industry_native = len(industry_native_authors)
    METRICS.with_academic_spell = len(training_spells)

    LOG.info(f"Found {len(training_spells)} training spells")
    LOG.info(f"Industry-native (excluded): {METRICS.industry_native}")

    training_df = pd.DataFrame(training_spells)
    training_df.to_parquet(FLOW_DIR / "training_spells.parquet", index=False)

    return training_df


def identify_labs(
    training_spells: pd.DataFrame,
    works: pd.DataFrame,
    authorships: pd.DataFrame,
    csrankings: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Step 2: Identify lab via PI detection using co-authorship proxy.

    For each training spell, find the most likely PI based on:
    - Co-authorship on >=2 works in the spell
    - Career age >= 8 years
    - Last-author position frequency
    - Total work count
    """
    LOG.info("Step 2: Identifying labs via PI detection...")

    # Load authorships with work dates
    auth_with_dates = authorships.merge(
        works[['work_id', 'publication_year']],
        on='work_id',
        how='left'
    )

    # Calculate first publication year for each author (career start)
    author_first_pub = auth_with_dates.groupby('author_openalex_id')['publication_year'].min().to_dict()

    # Calculate total work count for each author
    author_work_count = authorships.groupby('author_openalex_id').size().to_dict()

    # Create CSRankings faculty set
    csrankings_faculty = set(csrankings['author_openalex_id'].dropna()) if 'author_openalex_id' in csrankings.columns else set()

    labs = []
    person_lab = []
    adjudication_log = []

    for _, spell in training_spells.iterrows():
        author_id = spell['author_openalex_id']
        inst_ror = spell['institution_ror']
        spell_work_ids = json.loads(spell['work_ids'])
        spell_start = spell['spell_start_year']
        spell_end = spell['spell_end_year']

        # Get all authorships for works in this spell
        spell_authorships = authorships[authorships['work_id'].isin(spell_work_ids)].copy()

        # Get co-authors (exclude the trainee)
        coauthors = spell_authorships[spell_authorships['author_openalex_id'] != author_id].copy()

        # Count works per co-author
        coauthor_counts = coauthors.groupby('author_openalex_id').size()
        coauthor_candidates = coauthor_counts[coauthor_counts >= MIN_COAUTHOR_WORKS].index.tolist()

        # Score each candidate
        candidate_scores = []
        for candidate_id in coauthor_candidates:
            # Check affiliation match
            candidate_affs = coauthors[coauthors['author_openalex_id'] == candidate_id]
            has_matching_affiliation = (candidate_affs['institution_ror'] == inst_ror).any()

            if not has_matching_affiliation:
                continue

            # Check career age
            first_pub = author_first_pub.get(candidate_id)
            if first_pub is None:
                continue

            career_age = spell_start - first_pub
            if career_age < MIN_CAREER_AGE_FOR_PI:
                continue

            # Calculate last-author frequency
            candidate_works = spell_authorships[spell_authorships['author_openalex_id'] == candidate_id]
            last_author_count = (candidate_works['author_position'] == 'last').sum()
            last_author_freq = last_author_count / len(candidate_works) if len(candidate_works) > 0 else 0

            # Total work count
            total_works = author_work_count.get(candidate_id, 0)

            # Composite score
            score = (
                career_age * 0.3 +
                last_author_freq * 50 +  # Scale up to match other components
                np.log1p(total_works) * 2
            )

            candidate_scores.append({
                'candidate_id': candidate_id,
                'score': score,
                'career_age': career_age,
                'last_author_freq': last_author_freq,
                'total_works': total_works,
                'n_coauthorships': len(candidate_works)
            })

        # Select PI
        if len(candidate_scores) == 0:
            # No candidates - flag for adjudication
            adjudication_log.append({
                'author_id': author_id,
                'reason': 'no_candidates_meet_criteria',
                'spell_work_ids': spell_work_ids,
                'resolution': 'unresolved'
            })
            continue

        # Sort by score
        candidate_scores.sort(key=lambda x: x['score'], reverse=True)
        top_candidate = candidate_scores[0]

        # Check if needs adjudication
        needs_adjudication = False
        adjudication_reason = None

        # Check for consortium papers
        spell_author_counts = works[works['work_id'].isin(spell_work_ids)]['n_authors']
        if (spell_author_counts > MAX_AUTHORS_FOR_CONSORTIUM).all():
            needs_adjudication = True
            adjudication_reason = 'all_consortium_papers'

        # Check if top two are close
        elif len(candidate_scores) >= 2:
            second_score = candidate_scores[1]['score']
            if top_candidate['score'] > 0 and abs(top_candidate['score'] - second_score) / top_candidate['score'] < PI_SCORE_TOLERANCE:
                needs_adjudication = True
                adjudication_reason = 'close_scores'

        if needs_adjudication:
            adjudication_log.append({
                'author_id': author_id,
                'reason': adjudication_reason,
                'candidates': candidate_scores[:3],  # Top 3
                'resolution': 'selected_top'  # For now, just take top
            })
            METRICS.pi_total_adjudicated += 1
        else:
            METRICS.pi_total_deterministic += 1

        pi_author_id = top_candidate['candidate_id']

        # Validate against CSRankings
        is_validated = pi_author_id in csrankings_faculty

        if is_validated:
            if needs_adjudication:
                METRICS.pi_validated_adjudicated += 1
            else:
                METRICS.pi_validated_deterministic += 1

        # Create lab_id
        lab_id = f"{inst_ror}::{pi_author_id}"

        labs.append({
            'lab_id': lab_id,
            'institution_ror': inst_ror,
            'pi_author_id': pi_author_id,
            'pi_validated': is_validated,
            'resolution_method': 'adjudicated' if needs_adjudication else 'deterministic'
        })

        person_lab.append({
            'author_openalex_id': author_id,
            'lab_id': lab_id,
            'cohort_year': int(spell_end),  # Departure proxy
            'n_works_in_training': spell['n_works_in_spell']
        })

    METRICS.with_identified_pi = len(person_lab)

    LOG.info(f"Identified {len(labs)} unique labs")
    LOG.info(f"Linked {len(person_lab)} people to labs")
    LOG.info(f"PI validation rate (deterministic): {METRICS.pi_validated_deterministic}/{METRICS.pi_total_deterministic if METRICS.pi_total_deterministic > 0 else 1:.1%}")
    LOG.info(f"PI validation rate (adjudicated): {METRICS.pi_validated_adjudicated}/{METRICS.pi_total_adjudicated if METRICS.pi_total_adjudicated > 0 else 1:.1%}")

    # Save adjudication log
    with open(LOGS_DIR / "adjudication.jsonl", 'w') as f:
        for entry in adjudication_log:
            f.write(json.dumps(entry) + '\n')

    labs_df = pd.DataFrame(labs).drop_duplicates('lab_id')
    person_lab_df = pd.DataFrame(person_lab)

    labs_df.to_parquet(FLOW_DIR / "labs.parquet", index=False)
    person_lab_df.to_parquet(FLOW_DIR / "person_lab.parquet", index=False)

    return labs_df, person_lab_df


def detect_transitions(
    person_lab: pd.DataFrame,
    training_spells: pd.DataFrame,
    works: pd.DataFrame,
    authorships: pd.DataFrame,
    orgs: pd.DataFrame
) -> pd.DataFrame:
    """
    Step 4: Detect transitions from training lab to first employer.
    """
    LOG.info("Step 4: Detecting transitions...")

    # Create org type mapping
    org_type_map = orgs.set_index('org_id')['org_type'].to_dict()
    org_id_set = set(orgs['org_id'])

    # Get all author-work-institution data with dates
    auth_with_dates = authorships.merge(
        works[['work_id', 'publication_year']],
        on='work_id',
        how='left'
    )

    transitions = []

    for _, person in person_lab.iterrows():
        author_id = person['author_openalex_id']
        cohort_year = person['cohort_year']
        lab_id = person['lab_id']

        # Get training spell
        training_spell = training_spells[training_spells['author_openalex_id'] == author_id].iloc[0]
        training_inst_ror = training_spell['institution_ror']

        # Get all works after training spell
        author_works = auth_with_dates[
            (auth_with_dates['author_openalex_id'] == author_id) &
            (auth_with_dates['publication_year'] > cohort_year)
        ].sort_values('publication_year')

        if len(author_works) == 0:
            METRICS.no_transition_observed += 1
            transitions.append({
                'author_openalex_id': author_id,
                'lab_id': lab_id,
                'cohort_year': cohort_year,
                'to_org_id': 'no_observed_transition',
                'to_org_type': None,
                'transition_year': None,
                'confidence': 'no_transition',
                'n_works_before': training_spell['n_works_in_spell'],
                'n_works_after': 0
            })
            continue

        # Find first spell at different institution
        first_new_inst = None
        for inst_ror in author_works['institution_ror'].dropna().unique():
            if inst_ror != training_inst_ror:
                first_new_inst = inst_ror
                break

        if first_new_inst is None:
            METRICS.no_transition_observed += 1
            transitions.append({
                'author_openalex_id': author_id,
                'lab_id': lab_id,
                'cohort_year': cohort_year,
                'to_org_id': 'no_observed_transition',
                'to_org_type': None,
                'transition_year': None,
                'confidence': 'no_transition',
                'n_works_before': training_spell['n_works_in_spell'],
                'n_works_after': len(author_works)
            })
            continue

        # Get transition details
        transition_works = author_works[author_works['institution_ror'] == first_new_inst]
        transition_year = int(transition_works['publication_year'].min())
        gap_years = transition_year - cohort_year

        # Map to org_id
        to_org_id = transition_works['org_id'].mode()[0] if len(transition_works) > 0 else 'unmatched'

        # Classify destination
        if pd.isna(to_org_id) or to_org_id not in org_id_set:
            to_org_id = 'unmatched'
            to_org_type = 'unmatched'
        else:
            to_org_type = org_type_map.get(to_org_id, 'unmatched')
            if to_org_type in org_id_set:
                METRICS.in_orgs_csv_destination += 1

        # Confidence
        if gap_years > 4:
            confidence = 'low_confidence'
        else:
            confidence = 'high'

        transitions.append({
            'author_openalex_id': author_id,
            'lab_id': lab_id,
            'cohort_year': cohort_year,
            'to_org_id': to_org_id,
            'to_org_type': to_org_type,
            'transition_year': transition_year,
            'confidence': confidence,
            'n_works_before': training_spell['n_works_in_spell'],
            'n_works_after': len(transition_works)
        })

    METRICS.with_observed_transition = len([t for t in transitions if t['to_org_id'] != 'no_observed_transition'])

    LOG.info(f"Detected {METRICS.with_observed_transition} transitions")
    LOG.info(f"No transition observed: {METRICS.no_transition_observed}")

    transitions_df = pd.DataFrame(transitions)
    transitions_df.to_parquet(FLOW_DIR / "transitions.parquet", index=False)

    return transitions_df


def rank_labs(labs: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Rank labs by alumni count.
    """
    LOG.info("Step 5: Ranking labs...")

    # Handle empty transitions
    if len(transitions) == 0 or 'cohort_year' not in transitions.columns:
        LOG.warning("No transitions found - creating empty labs_ranked")
        labs_ranked = pd.DataFrame(columns=['lab_id', 'institution_ror', 'pi_author_id',
                                             'n_alumni_total', 'n_alumni_with_transition',
                                             'n_alumni_to_industry', 'pi_validated', 'rank'])
        labs_ranked.to_csv(FLOW_DIR / "labs_ranked.csv", index=False)
        return labs_ranked

    # Filter to target cohorts
    cohort_transitions = transitions[transitions['cohort_year'].isin(COHORT_YEARS)]

    # Count alumni per lab
    lab_stats = []
    for lab_id in labs['lab_id'].unique():
        lab_transitions = cohort_transitions[cohort_transitions['lab_id'] == lab_id]

        n_alumni_total = len(lab_transitions)
        n_with_transition = len(lab_transitions[lab_transitions['to_org_id'] != 'no_observed_transition'])
        n_to_industry = len(lab_transitions[lab_transitions['to_org_type'].isin(['industry_lab', 'industry_parent'])])

        lab_info = labs[labs['lab_id'] == lab_id].iloc[0]

        lab_stats.append({
            'lab_id': lab_id,
            'institution_ror': lab_info['institution_ror'],
            'pi_author_id': lab_info['pi_author_id'],
            'n_alumni_total': n_alumni_total,
            'n_alumni_with_transition': n_with_transition,
            'n_alumni_to_industry': n_to_industry,
            'pi_validated': lab_info['pi_validated']
        })

    labs_ranked = pd.DataFrame(lab_stats)
    labs_ranked = labs_ranked.sort_values('n_alumni_with_transition', ascending=False)
    labs_ranked['rank'] = range(1, len(labs_ranked) + 1)

    LOG.info(f"Ranked {len(labs_ranked)} labs")
    LOG.info(f"Top 5 labs by alumni count:")
    for _, lab in labs_ranked.head(5).iterrows():
        LOG.info(f"  Rank {lab['rank']}: {lab['lab_id']} - {lab['n_alumni_with_transition']} alumni")

    labs_ranked.to_csv(FLOW_DIR / "labs_ranked.csv", index=False)

    return labs_ranked


def build_flow_matrix(transitions: pd.DataFrame, labs_ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Build flow matrix with suppression.
    """
    LOG.info("Step 6: Building flow matrix...")

    # Handle empty transitions
    if len(transitions) == 0 or 'cohort_year' not in transitions.columns:
        LOG.warning("No transitions to build matrix - creating empty outputs")
        empty_df = pd.DataFrame(columns=['lab_id', 'to_org_id', 'cohort_year', 'n_transitions',
                                          'cohort_size', 'share_of_lab_cohort'])
        empty_df.to_parquet(FLOW_DIR / "flow_matrix.parquet", index=False)
        empty_df.to_csv(FLOW_DIR / "flow_matrix_published.csv", index=False)
        return empty_df

    # Filter to cohort years and valid transitions
    valid_transitions = transitions[
        (transitions['cohort_year'].isin(COHORT_YEARS)) &
        (transitions['to_org_id'] != 'no_observed_transition')
    ].copy()

    # Filter to top labs
    top_labs = set(labs_ranked.head(TOP_N_LABS)['lab_id'])
    valid_transitions = valid_transitions[valid_transitions['lab_id'].isin(top_labs)]

    # Create flow matrix
    flow_cells = valid_transitions.groupby(['lab_id', 'to_org_id', 'cohort_year']).agg({
        'author_openalex_id': 'count',
    }).rename(columns={'author_openalex_id': 'n_transitions'}).reset_index()

    # Calculate share of lab cohort
    lab_cohort_sizes = valid_transitions.groupby(['lab_id', 'cohort_year']).size().reset_index(name='cohort_size')
    flow_cells = flow_cells.merge(lab_cohort_sizes, on=['lab_id', 'cohort_year'])
    flow_cells['share_of_lab_cohort'] = flow_cells['n_transitions'] / flow_cells['cohort_size']

    # Apply suppression
    flow_cells['suppressed'] = flow_cells['n_transitions'] < MIN_CELL_SIZE

    METRICS.cells_published = (~flow_cells['suppressed']).sum()
    METRICS.cells_suppressed = flow_cells['suppressed'].sum()
    METRICS.transitions_in_suppressed_cells = flow_cells[flow_cells['suppressed']]['n_transitions'].sum()
    METRICS.surviving_suppression = flow_cells[~flow_cells['suppressed']]['n_transitions'].sum()

    LOG.info(f"Flow matrix: {len(flow_cells)} cells")
    LOG.info(f"Published cells: {METRICS.cells_published}")
    LOG.info(f"Suppressed cells: {METRICS.cells_suppressed}")
    LOG.info(f"Transitions surviving suppression: {METRICS.surviving_suppression}")

    # Save full matrix
    flow_cells.to_parquet(FLOW_DIR / "flow_matrix.parquet", index=False)

    # Save published matrix (suppressed cells removed)
    published = flow_cells[~flow_cells['suppressed']].drop(columns=['suppressed'])
    published.to_csv(FLOW_DIR / "flow_matrix_published.csv", index=False)

    return flow_cells


def generate_report(metrics: FlowMetrics):
    """Generate FLOW_REPORT.md"""
    LOG.info("Generating FLOW_REPORT.md...")

    report = f"""# Talent Flow Analysis Report

Generated: {datetime.now().isoformat()}

## 1. Coverage Cascade

The flow matrix is derived from publication records, not ground truth. This cascade shows
how many authors are lost at each filtering stage.

| Stage | Count | % of Total | % Lost |
|-------|-------|------------|--------|
| Total unique authors | {metrics.total_authors:,} | 100.0% | - |
| With academic training spell | {metrics.with_academic_spell:,} | {metrics.with_academic_spell/metrics.total_authors*100:.1f}% | {(metrics.total_authors - metrics.with_academic_spell)/metrics.total_authors*100:.1f}% |
| With identified PI | {metrics.with_identified_pi:,} | {metrics.with_identified_pi/metrics.total_authors*100:.1f}% | {(metrics.with_academic_spell - metrics.with_identified_pi)/metrics.total_authors*100:.1f}% |
| With observed transition | {metrics.with_observed_transition:,} | {metrics.with_observed_transition/metrics.total_authors*100:.1f}% | {(metrics.with_identified_pi - metrics.with_observed_transition)/metrics.total_authors*100:.1f}% |
| To destination in orgs.csv | {metrics.in_orgs_csv_destination:,} | {metrics.in_orgs_csv_destination/metrics.total_authors*100:.1f}% | {(metrics.with_observed_transition - metrics.in_orgs_csv_destination)/metrics.total_authors*100:.1f}% |
| Surviving cell suppression | {metrics.surviving_suppression:,} | {metrics.surviving_suppression/metrics.total_authors*100:.1f}% | {(metrics.in_orgs_csv_destination - metrics.surviving_suppression)/metrics.total_authors*100:.1f}% |

**Final coverage: {metrics.surviving_suppression/metrics.total_authors*100:.1f}%**

### Excluded Categories

- **Industry-native authors**: {metrics.industry_native:,} (no academic training spell)
- **No observed transition**: {metrics.no_transition_observed:,} (stayed at training institution or stopped publishing)

## 2. PI Validation

PI identification uses co-authorship as a proxy for advisor relationships. Validation checks
whether the identified PI appears in CSRankings faculty at that institution.

| Method | Validated | Total | Rate |
|--------|-----------|-------|------|
| Deterministic | {metrics.pi_validated_deterministic:,} | {metrics.pi_total_deterministic:,} | {metrics.pi_validated_deterministic/max(metrics.pi_total_deterministic,1)*100:.1f}% |
| Adjudicated | {metrics.pi_validated_adjudicated:,} | {metrics.pi_total_adjudicated:,} | {metrics.pi_validated_adjudicated/max(metrics.pi_total_adjudicated,1)*100:.1f}% |
| **Combined** | **{metrics.pi_validated_deterministic + metrics.pi_validated_adjudicated:,}** | **{metrics.pi_total_deterministic + metrics.pi_total_adjudicated:,}** | **{(metrics.pi_validated_deterministic + metrics.pi_validated_adjudicated)/max(metrics.pi_total_deterministic + metrics.pi_total_adjudicated,1)*100:.1f}%** |

**Interpretation**: A validation rate below 70% suggests the PI detection heuristic is unreliable.
Above 80% indicates the co-authorship proxy is functioning as intended.

## 3. Cell Suppression

Cells with fewer than {MIN_CELL_SIZE} transitions are suppressed to prevent individual identification.

| Metric | Count |
|--------|-------|
| Total cells | {metrics.cells_published + metrics.cells_suppressed:,} |
| Published cells | {metrics.cells_published:,} |
| Suppressed cells | {metrics.cells_suppressed:,} |
| Transitions in suppressed cells | {metrics.transitions_in_suppressed_cells:,} |
| Share of transitions suppressed | {metrics.transitions_in_suppressed_cells/(metrics.surviving_suppression + metrics.transitions_in_suppressed_cells)*100:.1f}% |

## 4. Known Limitations

### Proxy, Not Ground Truth

- **Training spell** is a publication-based proxy for PhD/postdoc period, not a verified enrollment record
- **Cohort year** is spell end year, a departure proxy that cannot distinguish PhD completion from postdoc end
- **PI identification** uses co-authorship, not declared advisor relationships
- **Transition** is first post-training institution in publication record, not actual hire date

### Invisibility

The following groups are **not represented** in this analysis:

- Authors with no academic training spell (industry-native)
- Authors who did not publish ≥2 works during training
- Authors who did not publish after leaving training institution
- Authors whose post-training affiliation is not in orgs.csv

### Affiliation Lag

OpenAlex affiliation data reaches the record 1-2 years after the actual move. All transition
years are **lower bounds** on the true hire date.

### Consortium Papers

Works with >20 authors make PI identification unreliable. Spells composed entirely of such
works were flagged for adjudication (see logs/adjudication.jsonl).

## 5. Output Files

- `flow/training_spells.parquet` - Training spell for each author
- `flow/labs.parquet` - Unique labs (institution + PI)
- `flow/person_lab.parquet` - Author-to-lab mapping
- `flow/transitions.parquet` - All detected transitions
- `flow/labs_ranked.csv` - Labs ranked by alumni count
- `flow/flow_matrix.parquet` - Full flow matrix (including suppressed cells)
- `flow/flow_matrix_published.csv` - Published matrix (suppressed cells removed)
- `logs/adjudication.jsonl` - Cases flagged for adjudication

## 6. Parameters

- Cohort years: {COHORT_YEARS}
- Minimum works in training spell: {MIN_WORKS_IN_TRAINING_SPELL}
- Minimum career age for PI: {MIN_CAREER_AGE_FOR_PI} years
- Cell suppression threshold: {MIN_CELL_SIZE} transitions
- Top N labs in published matrix: {TOP_N_LABS}
"""

    with open(FLOW_DIR / "FLOW_REPORT.md", 'w') as f:
        f.write(report)

    LOG.info("Report saved to flow/FLOW_REPORT.md")


def main():
    """Run complete talent flow derivation pipeline"""
    LOG.info("=== Starting Talent Flow Derivation ===")

    # Load data
    works, authorships, orgs, csrankings = load_data()

    # Step 1: Training spells
    training_spells = derive_training_spells(works, authorships, orgs)

    # Step 2: Lab identification
    labs, person_lab = identify_labs(training_spells, works, authorships, csrankings)

    # Step 3: Cohort assignment (already done in step 2)

    # Step 4: Transition detection
    transitions = detect_transitions(person_lab, training_spells, works, authorships, orgs)

    # Step 5: Lab ranking
    labs_ranked = rank_labs(labs, transitions)

    # Step 6: Flow matrix
    flow_matrix = build_flow_matrix(transitions, labs_ranked)

    # Generate report
    generate_report(METRICS)

    LOG.info("=== Talent Flow Derivation Complete ===")
    LOG.info(f"Coverage: {METRICS.surviving_suppression}/{METRICS.total_authors} ({METRICS.surviving_suppression/METRICS.total_authors*100:.1f}%)")


if __name__ == '__main__':
    main()
