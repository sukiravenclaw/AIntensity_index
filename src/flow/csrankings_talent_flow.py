#!/usr/bin/env python3
"""
CSRankings-based talent flow analysis.

Uses CSRankings faculty as known PIs, identifies their students via co-authorship,
and tracks where students are currently affiliated (as of 2024).

This approach works with limited historical data by leveraging verified faculty.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
FLOW_DIR = PROCESSED_DIR / "flow"
FLOW_DIR.mkdir(exist_ok=True, parents=True)

# Parameters
MIN_COAUTHOR_WORKS = 2  # Minimum co-authored papers to be considered a student
MIN_CELL_SIZE = 3  # Suppression threshold (lower for 2024-only data)
TOP_N_LABS = 100


def match_csrankings_to_openalex(csrankings: pd.DataFrame, authorships: pd.DataFrame) -> pd.DataFrame:
    """
    Match CSRankings faculty to OpenAlex author IDs using name matching.
    """
    LOG.info("Matching CSRankings faculty to OpenAlex authors...")

    # Create mapping by normalizing names
    def normalize_name(name):
        if pd.isna(name):
            return ""
        # Remove dots, lowercase, strip
        return name.replace('.', '').lower().strip()

    # Get unique authors from authorships
    unique_authors = authorships[['author_openalex_id', 'author_display_name']].drop_duplicates()
    unique_authors['normalized_name'] = unique_authors['author_display_name'].apply(normalize_name)

    # Normalize CSRankings names
    csrankings = csrankings.copy()
    csrankings['normalized_name'] = csrankings['name'].apply(normalize_name)

    # Match
    matched = csrankings.merge(
        unique_authors[['author_openalex_id', 'normalized_name']],
        on='normalized_name',
        how='left'
    )

    matched_count = matched['author_openalex_id'].notna().sum()
    LOG.info(f"Matched {matched_count}/{len(csrankings)} CSRankings faculty to OpenAlex IDs ({matched_count/len(csrankings)*100:.1f}%)")

    return matched


def identify_labs_and_students(
    faculty: pd.DataFrame,
    works: pd.DataFrame,
    authorships: pd.DataFrame,
    orgs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Identify labs (PI + institution) and their students via co-authorship.

    Returns:
        labs: DataFrame with lab information
        students: DataFrame with student-to-lab mapping
        student_destinations: DataFrame with current student affiliations
    """
    LOG.info("Identifying labs and students...")

    # Filter to faculty with OpenAlex IDs
    faculty_with_ids = faculty[faculty['author_openalex_id'].notna()].copy()
    LOG.info(f"Processing {len(faculty_with_ids)} faculty with OpenAlex IDs")

    # Get faculty authorships
    faculty_auth = authorships[authorships['author_openalex_id'].isin(faculty_with_ids['author_openalex_id'])]
    LOG.info(f"Found {len(faculty_auth)} faculty authorships")

    # Create institution mapping from CSRankings
    # Normalize institution names
    def normalize_institution(name):
        if pd.isna(name):
            return ""
        return name.strip().lower()

    faculty_with_ids['normalized_institution'] = faculty_with_ids['affiliation'].apply(normalize_institution)

    # Map OpenAlex institutions to CSRankings institutions
    auth_with_works = authorships.merge(works[['work_id', 'publication_year']], on='work_id', how='left')

    labs_list = []
    students_list = []
    student_destinations_list = []

    # For each faculty member
    for _, faculty_row in faculty_with_ids.iterrows():
        pi_id = faculty_row['author_openalex_id']
        pi_name = faculty_row['name']
        pi_institution = faculty_row['affiliation']

        # Get PI's works
        pi_works = set(faculty_auth[faculty_auth['author_openalex_id'] == pi_id]['work_id'])

        if len(pi_works) == 0:
            continue

        # Find co-authors on these works
        coauthors = authorships[
            (authorships['work_id'].isin(pi_works)) &
            (authorships['author_openalex_id'] != pi_id)
        ].copy()

        # Count co-authorships
        coauthor_counts = coauthors.groupby('author_openalex_id').agg({
            'work_id': 'count',
            'author_display_name': 'first'
        }).rename(columns={'work_id': 'n_coauthored_works'})

        # Filter to potential students (>= MIN_COAUTHOR_WORKS)
        potential_students = coauthor_counts[coauthor_counts['n_coauthored_works'] >= MIN_COAUTHOR_WORKS]

        if len(potential_students) == 0:
            continue

        # Create lab
        lab_id = f"{normalize_institution(pi_institution)}::{pi_id}"

        labs_list.append({
            'lab_id': lab_id,
            'pi_author_id': pi_id,
            'pi_name': pi_name,
            'institution_name': pi_institution,
            'n_pi_works': len(pi_works),
            'n_potential_students': len(potential_students)
        })

        # Record students
        for student_id, student_data in potential_students.iterrows():
            students_list.append({
                'student_author_id': student_id,
                'student_name': student_data['author_display_name'],
                'lab_id': lab_id,
                'pi_author_id': pi_id,
                'n_coauthored_works': student_data['n_coauthored_works']
            })

            # Find student's current affiliation (most recent in 2024)
            student_auth = auth_with_works[auth_with_works['author_openalex_id'] == student_id]

            if len(student_auth) == 0:
                continue

            # Get most recent affiliation
            recent_auth = student_auth.sort_values('publication_year', ascending=False).iloc[0]

            # Determine destination type
            org_id = recent_auth['org_id']
            if pd.isna(org_id):
                org_type = 'unmatched'
                org_name = recent_auth.get('institution_display_name', 'Unknown')
            else:
                org_match = orgs[orgs['org_id'] == org_id]
                if len(org_match) > 0:
                    org_type = org_match.iloc[0]['org_type']
                    org_name = org_match.iloc[0]['name']
                else:
                    org_type = recent_auth.get('institution_type', 'unmatched')
                    org_name = recent_auth.get('institution_display_name', 'Unknown')

            student_destinations_list.append({
                'student_author_id': student_id,
                'student_name': student_data['author_display_name'],
                'lab_id': lab_id,
                'current_org_id': org_id if not pd.isna(org_id) else 'unmatched',
                'current_org_name': org_name,
                'current_org_type': org_type,
                'most_recent_year': recent_auth['publication_year']
            })

    labs_df = pd.DataFrame(labs_list)
    students_df = pd.DataFrame(students_list)
    student_destinations_df = pd.DataFrame(student_destinations_list)

    LOG.info(f"Identified {len(labs_df)} labs with students")
    LOG.info(f"Found {len(students_df)} student-PI relationships")
    LOG.info(f"Tracked {len(student_destinations_df)} student destinations")

    return labs_df, students_df, student_destinations_df


def build_flow_matrix(
    labs: pd.DataFrame,
    student_destinations: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lab-to-organization flow matrix.
    """
    LOG.info("Building flow matrix...")

    # Rank labs by number of students
    lab_student_counts = student_destinations.groupby('lab_id').size().reset_index(name='n_students')
    labs_ranked = labs.merge(lab_student_counts, on='lab_id', how='left')
    labs_ranked['n_students'] = labs_ranked['n_students'].fillna(0).astype(int)
    labs_ranked = labs_ranked.sort_values('n_students', ascending=False)
    labs_ranked['rank'] = range(1, len(labs_ranked) + 1)

    # Take top N labs
    top_labs = set(labs_ranked.head(TOP_N_LABS)['lab_id'])
    filtered_destinations = student_destinations[student_destinations['lab_id'].isin(top_labs)]

    # Create flow matrix
    flow_matrix = filtered_destinations.groupby(['lab_id', 'current_org_id', 'current_org_type']).agg({
        'student_author_id': 'count'
    }).rename(columns={'student_author_id': 'n_students'}).reset_index()

    # Add lab totals for share calculation
    lab_totals = filtered_destinations.groupby('lab_id').size().reset_index(name='lab_total_students')
    flow_matrix = flow_matrix.merge(lab_totals, on='lab_id')
    flow_matrix['share_of_lab'] = flow_matrix['n_students'] / flow_matrix['lab_total_students']

    # Apply suppression
    flow_matrix['suppressed'] = flow_matrix['n_students'] < MIN_CELL_SIZE

    n_published = (~flow_matrix['suppressed']).sum()
    n_suppressed = flow_matrix['suppressed'].sum()

    LOG.info(f"Flow matrix: {len(flow_matrix)} cells")
    LOG.info(f"Published: {n_published}, Suppressed: {n_suppressed}")

    # Create published matrix (suppressed cells removed)
    published_matrix = flow_matrix[~flow_matrix['suppressed']].copy()

    # Add lab names for readability
    lab_info = labs_ranked[['lab_id', 'pi_name', 'institution_name', 'rank']]
    published_matrix = published_matrix.merge(lab_info, on='lab_id', how='left')

    # Sort by rank
    published_matrix = published_matrix.sort_values('rank', ascending=True)

    return labs_ranked, published_matrix


def generate_report(
    labs_ranked: pd.DataFrame,
    students: pd.DataFrame,
    student_destinations: pd.DataFrame,
    flow_matrix: pd.DataFrame,
    faculty_matched: int,
    faculty_total: int
):
    """Generate summary report"""
    LOG.info("Generating report...")

    # Calculate statistics
    total_labs = len(labs_ranked)
    labs_with_students = (labs_ranked['n_students'] > 0).sum()
    total_students = len(students)
    students_with_destinations = len(student_destinations)

    # Destination breakdown
    dest_by_type = student_destinations.groupby('current_org_type').size().sort_values(ascending=False)

    # Top labs
    top_10_labs = labs_ranked.head(10)

    report = f"""# CSRankings-Based Talent Flow Analysis

## Summary

**Data Constraints**: This analysis uses 2024 publication data only. Students are identified
via co-authorship with CSRankings faculty, not verified enrollment records.

## Coverage

- **CSRankings faculty**: {faculty_total:,} total, {faculty_matched:,} matched to OpenAlex ({faculty_matched/faculty_total*100:.1f}%)
- **Labs identified**: {total_labs:,} (PI + institution pairs)
- **Labs with students**: {labs_with_students:,} ({labs_with_students/total_labs*100:.1f}%)
- **Student-PI relationships**: {total_students:,}
- **Students with tracked destinations**: {students_with_destinations:,}

## Student Destinations by Organization Type

| Organization Type | Count | Percentage |
|-------------------|-------|------------|
"""

    for org_type, count in dest_by_type.items():
        pct = count / len(student_destinations) * 100
        report += f"| {org_type} | {count:,} | {pct:.1f}% |\n"

    report += f"""
## Top 10 Labs by Student Count

| Rank | PI Name | Institution | Students |
|------|---------|-------------|----------|
"""

    for _, lab in top_10_labs.iterrows():
        report += f"| {lab['rank']} | {lab['pi_name']} | {lab['institution_name']} | {int(lab['n_students'])} |\n"

    report += f"""
## Flow Matrix

- **Total cells**: {len(flow_matrix):,}
- **Published cells** (n ≥ {MIN_CELL_SIZE}): {(~flow_matrix['suppressed']).sum():,}
- **Suppressed cells**: {flow_matrix['suppressed'].sum():,}

## Methodology

### PI Identification
- Used CSRankings verified faculty as PIs
- Matched by name to OpenAlex author IDs
- No career age requirements (faculty are pre-verified)

### Student Identification
- Students identified as authors with ≥{MIN_COAUTHOR_WORKS} co-authored works with a PI
- Co-authorship is a proxy for advisor relationship
- Cannot distinguish PhD students from postdocs, collaborators, or master's students

### Destination Tracking
- Student's most recent affiliation (2024) used as current employer
- Organization type mapped from orgs.csv or OpenAlex institution_type
- "Unmatched" means organization not in orgs.csv

## Limitations

1. **2024 data only**: Cannot track historical training spells or graduation years
2. **Co-authorship proxy**: May include collaborators who are not actual students
3. **Current affiliation only**: Not true hiring/transition dates
4. **No temporal ordering**: Cannot determine if destination is first job or later move
5. **Missing historical context**: Famous advisor-student pairs from pre-2024 are invisible

## Output Files

- `flow/csrankings_labs_ranked.csv` - All labs ranked by student count
- `flow/csrankings_students.csv` - Student-to-lab mappings
- `flow/csrankings_student_destinations.csv` - Student current affiliations
- `flow/csrankings_flow_matrix.csv` - Lab-to-org flow matrix (published, suppressed cells removed)
"""

    with open(FLOW_DIR / "CSRANKINGS_FLOW_REPORT.md", 'w') as f:
        f.write(report)

    LOG.info("Report saved to flow/CSRANKINGS_FLOW_REPORT.md")


def main():
    LOG.info("=== CSRankings-Based Talent Flow Analysis ===")

    # Load data
    LOG.info("Loading data...")
    works = pd.read_parquet(PROCESSED_DIR / "works.parquet")
    authorships = pd.read_parquet(PROCESSED_DIR / "authorships.parquet")
    orgs = pd.read_csv(DATA_DIR / "orgs.csv")
    csrankings = pd.read_csv(DATA_DIR / "external" / "csrankings.csv")

    LOG.info(f"Loaded: {len(works)} works, {len(authorships)} authorships, {len(orgs)} orgs, {len(csrankings)} CSRankings faculty")

    # Match CSRankings to OpenAlex
    faculty = match_csrankings_to_openalex(csrankings, authorships)
    faculty_matched = faculty['author_openalex_id'].notna().sum()

    # Identify labs and students
    labs, students, student_destinations = identify_labs_and_students(
        faculty, works, authorships, orgs
    )

    # Build flow matrix
    labs_ranked, flow_matrix_published = build_flow_matrix(labs, student_destinations)

    # Save outputs
    LOG.info("Saving outputs...")
    labs_ranked.to_csv(FLOW_DIR / "csrankings_labs_ranked.csv", index=False)
    students.to_csv(FLOW_DIR / "csrankings_students.csv", index=False)
    student_destinations.to_csv(FLOW_DIR / "csrankings_student_destinations.csv", index=False)
    flow_matrix_published.to_csv(FLOW_DIR / "csrankings_flow_matrix.csv", index=False)

    # Generate report
    generate_report(
        labs_ranked, students, student_destinations,
        flow_matrix_published, faculty_matched, len(csrankings)
    )

    LOG.info("=== Analysis Complete ===")
    LOG.info(f"Identified {len(labs_ranked)} labs with {len(students)} student relationships")
    LOG.info(f"Top lab: {labs_ranked.iloc[0]['pi_name']} ({labs_ranked.iloc[0]['institution_name']}) - {int(labs_ranked.iloc[0]['n_students'])} students")


if __name__ == '__main__':
    main()
