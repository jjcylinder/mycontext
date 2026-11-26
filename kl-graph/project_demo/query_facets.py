"""Query-time facets derived only from the question and visible source records."""

import re
from collections import defaultdict

_SEPARATORS = re.compile(r'、|以及|及|和|与')
_NUMBERS = {**dict(zip('二三四五六七八九十', range(2, 11))), '两': 2}


def _title_family(records, count, category):
    # Independent document series must name the same set of objects. Avoid
    # resolving a counted category from one coincidental shared title ending.
    suffixes = defaultdict(set)
    for title in sorted({row['title'].strip() for row in records.values()}):
        for cut in range(2, len(title) - 1):
            suffixes[title[cut:]].add(title[:cut])
    families = defaultdict(list)
    for suffix, names in suffixes.items():
        if len(names) == count:
            families[tuple(sorted(names))].append(suffix)
    category_rows = [row['title'] + ' ' + row['content'] for row in records.values()
                     if category.casefold() in (row['title'] + row['content']).casefold()]
    supported = []
    for family, endings in families.items():
        # Nested suffixes of the same series cannot corroborate each other.
        if len(endings) < 2 or sum(any(name in text for text in category_rows)
                                   for name in family) <= count / 2:
            continue
        supported.append(family)
    return supported[0] if len(supported) == 1 else ()


def plan_facets(question, records):
    """Keep qualifiers and requested matters with each independently named object."""
    clauses = re.split(r'[，,。；;！？?]', question.strip())
    for clause in clauses:
        match = re.search(r'按([2-9]|10|[两二三四五六七八九十])类(.+?)(?:汇总|列出|说明|梳理|比较|整理)(.*)', clause)
        if match:
            number, category, matters = match.groups()
            count = int(number) if number.isascii() else _NUMBERS[number]
            return {name: f'{name} {category} {matters}'.strip()
                    for name in _title_family(records, count, category)}
    for i, clause in enumerate(clauses):
        match = re.fullmatch(r'(.+?)分别(.+)', clause)
        if match:
            items, matter = match.groups()
            names = [name.strip() for name in _SEPARATORS.split(items)]
            if 2 <= len(names) <= 10 and all(1 <= len(name) <= 12 for name in names):
                return {name: '，'.join([*clauses[:i], name + matter, *filter(None, clauses[i+1:])])
                        for name in names}
    return {}


def combine_queries(base, branches, depth):
    """Best rank per item. Repetition across similar subqueries adds no votes."""
    if not branches:
        return list(base[:depth])
    base_ranks = {key: i for i, key in enumerate(base, 1)}
    best = base_ranks.copy()
    for ranking in branches:
        for i, key in enumerate(ranking[:depth], 1):
            best[key] = min(i, best.get(key, i))
    return sorted(best, key=lambda key: (best[key], base_ranks.get(key, float('inf')), key))[:depth]
