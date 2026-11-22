"""Compare supported group-level results within the selected source passages."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from itertools import combinations

# Only positive, declarative count statements with a named group qualify.
_GROUP = r"(?:[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+|[甲乙丙丁戊己庚辛壬癸A-Z]组)"
_DIGIT = r"[一二两三四五六七八九]"
_TENS = rf"{_DIGIT}十{_DIGIT}?"
_HUNDREDS = rf"{_DIGIT}百(?:{_TENS}|零{_DIGIT})?"
_THOUSANDS = rf"{_DIGIT}千(?:{_HUNDREDS}|零(?:{_TENS}|{_DIGIT}))?"
_NUM = rf"(?:[0-9]{{1,12}}|{_THOUSANDS}|{_HUNDREDS}|{_DIGIT}?十{_DIGIT}?|{_DIGIT}|零)"
_UNIT = r"(?:个(?:不同|独立)?样例|个(?:不同|独立)?样本|个用例|例|条|项)"
_COUNT = re.compile(rf"(?<![0-9零一二两三四五六七八九十百千万亿])({_NUM})\s*({_UNIT})")
_PREFIX = r"(?:(?:本轮|本次|预演包|正式包)\s*)?"
_RESULT = re.compile(rf"{_PREFIX}(?P<group>{_GROUP})(?:的|有|共)?\s*(?P<body>[^，,。；;！？?\n]{{1,36}}?)(?:均已|均|全部|都|已)?通过")
_CLAUSES = re.compile(r"[。；;\n]+")
_MEMBER = r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*"
_ROSTER = re.compile(rf"(?P<group>{_GROUP})\s*(?:的)?(?:全部通过样例|通过样例完整清单|通过样本完整清单)\s*[为是：:]\s*(?P<ids>{_MEMBER}(?:\s*[、,，]\s*{_MEMBER})*)")
_PAIR = re.compile(rf"(?P<a>{_GROUP})\s*(?:和|与|及)\s*(?P<b>{_GROUP})(?:的)?\s*"
                   r"(?:(?:(?:样例|样本|成员)(?:集合)?)?(?P<membership>互不重叠|互斥|不重叠|有重叠)"
                   r"|(?:检查|验收)?(?P<criterion>口径相同|口径不同|标准相同|标准不同))")
_DIGITS = dict(zip("零一二三四五六七八九两", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2)))


def _number(token):
    if token.isascii():
        return int(token)
    value = digit = 0
    for char in token:
        if char in _DIGITS:
            digit = _DIGITS[char]
        else:
            value += (digit or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            digit = 0
    return value + digit


def asks_for_combined_count(question):
    totals = r"合计|总共|一共|总数|总计|累计|相加|加起来|去重后|合算|合并(?:为|成|报|说|计算)"
    if re.search(r"(?:不要|不用|无需)(?:计算|考虑)?(?:合计|总数|去重|相加|合并)", question):
        return False
    if re.search(rf"(?:{totals}).{{0,10}}(?:多少|几[个条例项])|(?:多少|几[个条例项]).{{0,8}}(?:不同|不重复)", question):
        return True
    counts = list(_COUNT.finditer(question))
    if len(counts) >= 2:
        if re.search(totals, question):
            return True
        # An implicit proposed total follows the group counts in the question.
        if len(counts) >= 3:
            proposed = counts[-1]
            reporting = question[counts[-2].end():proposed.start()]
            return bool(re.search(r"(?:能否|能不能|是否|可以).{0,12}(?:报|称|说|算|计)", reporting)
                        and re.match(r"(?:完整|全部|都)?(?:通过|成功)", question[proposed.end():]))
    return bool(re.search(rf"(?:两组|两批|两个集合|{_GROUP}.{{0,20}}{_GROUP}).{{0,20}}(?:能否相加|合计|合并计算|是否重叠|是否互斥|去重)", question))


def _observation(clause):
    match = _RESULT.fullmatch(clause.strip())
    if not match:
        return None
    body = match["body"]
    counts = list(_COUNT.finditer(body))
    if len(counts) != 1:
        return None
    count = counts[0]
    criterion = (body[:count.start()] + body[count.end():]).strip(" 的")
    if re.search(r"未|不|待|至少|至多|超过|少于|接近|约|预计|部分|失败|如果|可能|分之|[/~～%至到]|[0-9]-", criterion):
        return None
    if criterion and not re.fullmatch(r"[\u4e00-\u9fff]{0,12}(?:检查|核查|评审|审核|测试|复测|验证|验收)", criterion):
        return None
    if re.search(_NUM + r"|[万亿点]", criterion):
        return None
    return match["group"], _number(count[1]), count[2], criterion


def observed_count_fact(text):
    """Index the quantity only; the answer-facing wording must come from the source."""
    return _observation(re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", text).rstrip("。."))


def _selected_observations(evidence, count_facts):
    observations = defaultdict(list)
    rosters = defaultdict(list)
    pair_claims = defaultdict(list)
    for row in evidence:
        if row.get("project_scope") != "requested":
            continue
        ref = row["id"]
        facts = count_facts.get(ref, ())
        body = row.get("model_excerpt", row["content"])
        for clause in _CLAUSES.split(body):
            clause = clause.strip()
            member = _ROSTER.fullmatch(clause)
            if member:
                members = frozenset(re.split(r"\s*[、,，]\s*", member["ids"]))
                rosters[member["group"]].append((members, ref, row.get("source_id") or ref))
            pair = _PAIR.fullmatch(clause)
            if pair:
                prop = pair["membership"] or pair["criterion"]
                pair_claims[frozenset((pair["a"], pair["b"]))].append((prop, ref))
            # Exclude questions, qualifications and hypothetical results as a whole.
            if re.search(r"[?？]|如果|假设|并非|不能确认|尚待确认|能否|若", clause):
                continue
            for fragment in re.split(r"[,，]", clause):
                obs = _observation(fragment)
                if obs and any(tuple(f[:3]) == obs[:3] and
                               (not obs[3] or f[3].replace("和", "") == obs[3].replace("和", ""))
                               for f in facts):
                    observations[obs[0]].append((*obs[1:], ref, fragment.strip()))
    return observations, rosters, pair_claims


def compare_counts(question, evidence, count_facts):
    """Use exact selected quotations for the requested cohorts and their relations."""
    if not count_facts or not asks_for_combined_count(question):
        return None
    observations, rosters, pair_claims = _selected_observations(evidence, count_facts)
    named = set(re.findall(_GROUP, question))
    if named:
        if len(named) < 2 or not named <= observations.keys():
            return None
    else:
        named = set(observations)
        # A proposed total in the question could itself match a third cohort.
        # Only infer the operands when there is a single unambiguous pair.
        if len(named) != 2:
            return None
        quantities = Counter(_number(n) for n, _ in _COUNT.findall(question))
        if quantities and Counter(observations[name][0][0] for name in named) - quantities:
            return None
    groups = []
    proofs = set()
    lists = {}
    roster_sources = {}
    units = set()
    for name in sorted(named):
        results = observations[name]
        if len({(n, unit) for n, unit, *_ in results}) != 1:
            return None
        if len({check.replace("和", "") for _, _, check, *_ in results if check}) > 1:
            return None
        n, unit, check, ref, quote = max(results, key=lambda r: len(r[2]))
        groups.append({"group": name, "count": n, "unit": unit, "check": check,
                       "result": "通过", "source": ref, "quote": quote})
        proofs.add(ref)
        units.add("样例" if unit == "例" else re.sub(r"^个(?:独立|不同)?", "", unit))
        entries = rosters[name]
        distinct = {ids for ids, _, _ in entries}
        if len(distinct) > 1 or any(len(ids) != n for ids in distinct):
            return None
        if distinct:
            lists[name] = next(iter(distinct))
            roster_sources[name] = {source for _, _, source in entries}
        proofs.update(r for _, r, _ in entries)

    all_disjoint = all_same = True
    any_overlap = any_different = False
    for a, b in combinations(sorted(named), 2):
        values = pair_claims[frozenset((a, b))]
        proofs.update(ref for _, ref in values)
        claims = {value for value, _ in values}
        disjoint = bool(claims & {"互斥", "不重叠", "互不重叠"})
        overlap = "有重叠" in claims
        same = bool(claims & {"标准相同", "口径相同"})
        different = bool(claims & {"标准不同", "口径不同"})
        # Local roster IDs have a common namespace within a source document.
        if a in lists and b in lists and roster_sources[a] & roster_sources[b]:
            common = bool(lists[a] & lists[b])
            disjoint |= not common
            overlap |= common
        if (overlap and disjoint) or (same and different):
            return None
        all_disjoint &= disjoint
        any_overlap |= overlap
        all_same &= same
        any_different |= different
    total = None
    label = "两组" if len(named) == 2 else "各组"
    membership = f"当前计数依据未确定{label}样例是否重叠，去重后的样例总数未知。"
    if len(lists) == len(named) and set.intersection(*roster_sources.values()):
        total = len(frozenset().union(*lists.values()))
        membership = "完整通过样例清单" + ("有重叠。" if any_overlap else "互不重叠。")
    elif all_disjoint:
        membership = f"来源明确{label}样例互不重叠。"
        if len(units) == 1 and units <= {"样例", "样本", "用例"}:
            total = sum(g["count"] for g in groups)
    elif any_overlap:
        membership = "来源明确存在重叠，现有依据不足以计算去重总数。"
    criterion = f"来源明确{label}采用相同口径。" if all_same else (
        "来源明确存在检查口径不同的组。" if any_different else "按各组引文核对检查内容；当前计数依据未建立统一的完整验收标准。")
    return {"groups": groups, "membership": membership, "distinct_samples": total,
            "criterion": criterion, "evidence_ids": sorted(proofs)}
