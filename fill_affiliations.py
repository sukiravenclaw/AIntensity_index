#!/usr/bin/env python3
"""
Fill missing author affiliations in disambiguation_sample.csv
1. Forward-fill within each author (use existing affiliation data)
2. Search Google Scholar for authors with all empty affiliations
3. Create affiliations_filled column with combined results
"""

import pandas as pd
import json
import time
from typing import List, Dict, Optional
import re

def parse_affiliations(aff_str: str) -> List[str]:
    """Parse the listed_affiliations JSON string into a list."""
    if pd.isna(aff_str) or aff_str == '' or aff_str == '[]':
        return []
    try:
        return json.loads(aff_str)
    except:
        return []

def search_author_affiliation(author_name: str, work_title: str = "") -> List[str]:
    """
    Search for author's current affiliation using Semantic Scholar API and web search.
    Returns list of affiliations [current institution, education institution if found]
    """
    affiliations = []

    try:
        import requests
        from bs4 import BeautifulSoup

        print(f"  Searching for: {author_name}")

        # Try Semantic Scholar API first (no auth required, no rate limit for moderate use)
        try:
            # Search by author name
            ss_url = f"https://api.semanticscholar.org/graph/v1/author/search?query={author_name.replace(' ', '+')}&limit=1"
            headers = {'User-Agent': 'AI-Talent-Collection/1.0'}

            response = requests.get(ss_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    author_id = data['data'][0]['authorId']

                    # Get author details
                    detail_url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}?fields=affiliations,papers"
                    detail_response = requests.get(detail_url, headers=headers, timeout=10)

                    if detail_response.status_code == 200:
                        author_data = detail_response.json()

                        # Extract affiliations
                        if author_data.get('affiliations'):
                            for aff in author_data['affiliations'][:2]:  # Get top 2
                                affiliations.append(aff)
                                print(f"    Found (Semantic Scholar): {aff}")
        except Exception as e:
            print(f"    Semantic Scholar failed: {e}")

        # If no affiliations found, try web scraping
        if not affiliations:
            # Try DuckDuckGo search (more lenient than Google)
            search_query = f"{author_name} researcher affiliation university"
            ddg_url = f"https://html.duckduckgo.com/html/?q={search_query.replace(' ', '+')}"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            try:
                response = requests.get(ddg_url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text from results
                results = soup.find_all('a', class_='result__snippet')
                for result in results[:5]:
                    text = result.get_text()

                    # Look for institution patterns
                    patterns = [
                        r'([A-Z][a-zA-Z\s&\-]+University[^,.\n]{0,20})',
                        r'([A-Z][a-zA-Z\s&\-]+Institute[^,.\n]{0,20})',
                        r'([A-Z][a-zA-Z\s&\-]+College[^,.\n]{0,20})',
                        r'([A-Z][a-zA-Z\s&\-]+Laboratory[^,.\n]{0,20})',
                    ]

                    for pattern in patterns:
                        matches = re.findall(pattern, text)
                        for match in matches:
                            inst = match.strip()
                            if inst and inst not in affiliations and len(inst) > 5:
                                affiliations.append(inst)
                                print(f"    Found (Web): {inst}")
                                if len(affiliations) >= 2:
                                    break
                        if len(affiliations) >= 2:
                            break
                    if len(affiliations) >= 2:
                        break
            except Exception as e:
                print(f"    Web search failed: {e}")

        time.sleep(1)  # Rate limiting
        return affiliations[:2]  # Return at most 2 affiliations

    except Exception as e:
        print(f"    Error searching for {author_name}: {e}")
        return []

def main():
    print("Loading disambiguation_sample.csv...")
    df = pd.read_csv('data/processed/disambiguation_sample.csv')

    print(f"Loaded {len(df)} rows with {df['author_openalex_id'].nunique()} unique authors")

    # Parse listed_affiliations into lists
    df['aff_list'] = df['listed_affiliations'].apply(parse_affiliations)

    # Step 1: Forward-fill within each author
    print("\nStep 1: Forward-filling affiliations within each author...")

    def forward_fill_author(group):
        """For each author, collect all unique affiliations and fill empty rows."""
        # Collect all unique affiliations from all rows for this author
        all_affs = []
        for affs in group['aff_list']:
            all_affs.extend(affs)

        # Get unique affiliations, preserving order
        unique_affs = []
        seen = set()
        for aff in all_affs:
            if aff not in seen:
                unique_affs.append(aff)
                seen.add(aff)

        # Fill empty rows with the collected affiliations
        group['aff_list'] = group['aff_list'].apply(
            lambda x: unique_affs if len(x) == 0 and len(unique_affs) > 0 else x
        )
        return group

    df = df.groupby('author_openalex_id', group_keys=False).apply(forward_fill_author)

    filled_count = (df['aff_list'].apply(len) > 0).sum()
    print(f"After forward-fill: {filled_count}/{len(df)} rows have affiliations")

    # Step 2: Find authors with ALL empty affiliations
    print("\nStep 2: Finding authors with no affiliation data...")

    author_has_aff = df.groupby('author_openalex_id')['aff_list'].apply(
        lambda x: any(len(affs) > 0 for affs in x)
    )
    authors_no_aff = author_has_aff[~author_has_aff].index.tolist()

    print(f"Found {len(authors_no_aff)} authors with no affiliation data")

    # Get author names for these authors
    authors_to_search = df[df['author_openalex_id'].isin(authors_no_aff)][
        ['author_openalex_id', 'author_display_name']
    ].drop_duplicates()

    print(f"\nAuthors to search:")
    for idx, row in authors_to_search.head(10).iterrows():
        print(f"  - {row['author_display_name']}")
    if len(authors_to_search) > 10:
        print(f"  ... and {len(authors_to_search) - 10} more")

    # Step 3: Search for affiliations
    print(f"\nStep 3: Searching for affiliations (this may take a while)...")
    print(f"Searching for {len(authors_to_search)} authors...")

    # Create a mapping of author_id -> searched affiliations
    searched_affs = {}

    for i, (idx, row) in enumerate(authors_to_search.iterrows(), 1):
        author_id = row['author_openalex_id']
        author_name = row['author_display_name']

        print(f"\n[{i}/{len(authors_to_search)}] Searching: {author_name}")
        affs = search_author_affiliation(author_name)

        if affs:
            searched_affs[author_id] = affs
            print(f"  ✓ Found {len(affs)} affiliation(s)")
        else:
            searched_affs[author_id] = []  # Mark as searched but not found
            print(f"  ✗ No affiliations found")

    print(f"\nSearch complete: {sum(1 for affs in searched_affs.values() if len(affs) > 0)}/{len(authors_to_search)} successful")

    # Step 4: Create affiliations_filled column
    print("\nStep 4: Creating affiliations_filled column...")

    def create_filled_column(row):
        """Combine existing affiliations with searched ones."""
        existing = row['aff_list']
        author_id = row['author_openalex_id']

        # If we have existing affiliations, use them
        if len(existing) > 0:
            return existing

        # Otherwise, use searched affiliations if available
        if author_id in searched_affs:
            return searched_affs[author_id]

        return []

    df['affiliations_filled'] = df.apply(create_filled_column, axis=1)

    # Convert back to JSON string format
    df['affiliations_filled'] = df['affiliations_filled'].apply(json.dumps)

    # Drop temporary column
    df = df.drop(columns=['aff_list'])

    # Step 5: Save results
    print("\nStep 5: Saving results...")
    df.to_csv('data/processed/disambiguation_sample.csv', index=False)

    # Print summary
    filled_rows = df['affiliations_filled'].apply(lambda x: len(json.loads(x)) > 0).sum()
    print(f"\nSummary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with affiliations: {filled_rows} ({filled_rows/len(df)*100:.1f}%)")
    print(f"  Authors searched: {len(searched_affs)}")
    print(f"  Successful searches: {sum(1 for affs in searched_affs.values() if len(affs) > 0)}")
    print(f"\n✓ Updated data/processed/disambiguation_sample.csv")

if __name__ == '__main__':
    main()
