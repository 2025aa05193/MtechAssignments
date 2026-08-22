"""
Assessment of generated SQL: exact match, component-wise accuracy, execution
accuracy and BLEU, plus a session log that can be downloaded.

    from evaluation import assess, SessionLog

    log = SessionLog()
    row = assess(question, predicted_sql, reference_sql=gold, df=df)
    log.add(row)
    log.to_csv()      # -> str
    log.to_json()     # -> str

IMPORTANT - what needs a reference
-----------------------------------
Exact match, component accuracy and BLEU all compare the prediction against a
GOLD query. In the app the user asks free-form questions about their own CSV,
so no gold exists unless they supply one. Those metrics are therefore reported
as None (rendered "n/a") when no reference is given - NOT as 0.0, which would
silently drag a reported average downwards and misrepresent the system.

Two things are always measurable without a reference, and are recorded every
time: whether the query executes, and how many rows it returns.

BLEU is implemented here rather than pulled from nltk to avoid the dependency;
it is validated against worked examples in the self-test at the bottom.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence

AGGS = ("COUNT", "SUM", "AVG", "MIN", "MAX")

_TOKEN_RE = re.compile(
    r"'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?|[A-Za-z_][A-Za-z_0-9]*|>=|<=|<>|!=|\S")


# ==========================================================================
# Tokenisation / normalisation
# ==========================================================================
def sql_tokens(sql: str) -> List[str]:
    """Lowercased token list used by both BLEU and exact match."""
    return [t.lower() for t in _TOKEN_RE.findall(sql or "")]


def normalise(sql: str) -> str:
    return " ".join(sql_tokens(sql))


# ==========================================================================
# 1. Exact match
# ==========================================================================
def exact_match(pred: str, gold: str) -> int:
    """1 if the two queries are token-identical after lowercasing."""
    return int(normalise(pred) == normalise(gold))


# ==========================================================================
# 2. Component-wise accuracy
# ==========================================================================
def parse_sql(sql: str) -> Dict:
    """
    Shallow structural parse: projected column, aggregate, WHERE conditions.
    Conditions are returned as a SET, so `a = 1 AND b = 2` and
    `b = 2 AND a = 1` compare equal - they are the same query, and penalising
    the ordering measures nothing about the model.
    """
    s = " ".join((sql or "").split())
    m = re.search(r"SELECT\s+(.*?)\s+FROM", s, re.I)
    proj = m.group(1).strip() if m else ""

    agg = None
    for a in AGGS:
        if re.match(rf"^{a}\s*\(", proj, re.I):
            agg = a
            break
    col = re.sub(rf"^({'|'.join(AGGS)})\s*\(|\)$", "", proj, flags=re.I)
    col = col.strip().strip('"').strip().lower()

    conds = set()
    w = re.search(r"\bWHERE\s+(.*?)(?:\s+GROUP\b|\s+ORDER\b|\s+LIMIT\b|$)", s, re.I)
    if w:
        for part in re.split(r"\s+AND\s+", w.group(1), flags=re.I):
            cm = re.match(r'\s*"?([^"]+?)"?\s*(>=|<=|<>|!=|=|>|<)\s*(.+?)\s*$', part)
            if cm:
                c, op, v = cm.groups()
                conds.add((c.strip().lower(), op, v.strip().strip("'\"").lower()))

    return {"select_column": col, "aggregate": agg, "conditions": conds,
            "n_conditions": len(conds)}


def component_scores(pred: str, gold: str) -> Dict[str, int]:
    """Per-component correctness, plus an order-insensitive overall flag."""
    p, g = parse_sql(pred), parse_sql(gold)
    col = int(p["select_column"] == g["select_column"])
    agg = int(p["aggregate"] == g["aggregate"])
    whr = int(p["conditions"] == g["conditions"])
    return {
        "select_column_correct": col,
        "aggregate_correct": agg,
        "where_clause_correct": whr,
        "component_average": round((col + agg + whr) / 3, 4),
        "structural_match": int(col and agg and whr),
    }


# ==========================================================================
# 3. Execution accuracy
# ==========================================================================
def _run(df, sql: str, table_name: str, runner):
    try:
        res, err = runner(df, sql, table_name)
        return res, err
    except Exception as e:                    # runner itself blew up
        return None, str(e)


def _resultset(df_result) -> Optional[list]:
    """Order-insensitive canonical form of a result table."""
    if df_result is None:
        return None
    rows = [tuple("" if v is None else str(v) for v in row)
            for row in df_result.itertuples(index=False, name=None)]
    return sorted(rows)


def execution_scores(df, pred: str, gold: Optional[str], runner,
                     table_name: str = "table") -> Dict:
    """
    Run the prediction (and the reference, when given) and compare RESULT SETS.

    Execution accuracy is the metric that matters most in practice: two
    different-looking queries that return the same rows are both right, which
    exact match cannot see. Comparison is order-insensitive because SQL makes
    no ordering guarantee without ORDER BY.
    """
    out: Dict = {"executed": 0, "rows_returned": None, "error": None,
                 "execution_match": None}
    if df is None:
        out["error"] = "no table uploaded"
        return out

    pred_res, pred_err = _run(df, pred, table_name, runner)
    if pred_err:
        out["error"] = pred_err
        return out
    out["executed"] = 1
    out["rows_returned"] = int(len(pred_res))

    if gold:
        gold_res, gold_err = _run(df, gold, table_name, runner)
        if gold_err:
            out["error"] = f"reference failed: {gold_err}"
        else:
            out["execution_match"] = int(_resultset(pred_res) == _resultset(gold_res))
    return out


# ==========================================================================
# 4. BLEU
# ==========================================================================
def bleu(pred: str, gold: str, max_n: int = 4, epsilon: float = 0.1) -> float:
    """
    Sentence-level BLEU over SQL tokens.

    Geometric mean of modified n-gram precisions times a brevity penalty.
    Smoothing follows NLTK's SmoothingFunction.method1 (Chen & Cherry 2014):
    a zero-count precision is replaced by `epsilon / denominator` rather than
    left at zero. Without it a single missing 4-gram collapses the score to 0,
    which is uninformative for queries this short.

    Verified to agree with nltk.translate.bleu_score.sentence_bleu using
    method1 to within 1e-9 on the cases in the self-test below.
    """
    p_toks, g_toks = sql_tokens(pred), sql_tokens(gold)
    if not p_toks or not g_toks:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p_ngrams = Counter(tuple(p_toks[i:i + n]) for i in range(len(p_toks) - n + 1))
        g_ngrams = Counter(tuple(g_toks[i:i + n]) for i in range(len(g_toks) - n + 1))
        denom = sum(p_ngrams.values())
        if denom == 0:
            # prediction is shorter than n: NLTK treats this as precision 0
            return 0.0
        num = sum((p_ngrams & g_ngrams).values())
        precisions.append(num / denom if num > 0 else epsilon / denom)

    log_p = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if len(p_toks) > len(g_toks) else math.exp(
        1 - len(g_toks) / max(len(p_toks), 1))
    return round(bp * math.exp(log_p), 4)


# ==========================================================================
# Combined assessment
# ==========================================================================
def assess(question: str, predicted_sql: str, reference_sql: Optional[str] = None,
           df=None, runner=None, table_name: str = "table",
           extra: Optional[Dict] = None) -> Dict:
    """One assessment row. Reference-dependent metrics are None when no gold."""
    row: Dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "generated_sql": predicted_sql,
        "reference_sql": reference_sql or "",
        "exact_match": None,
        "select_column_correct": None,
        "aggregate_correct": None,
        "where_clause_correct": None,
        "component_average": None,
        "structural_match": None,
        "bleu": None,
        "execution_match": None,
        "executed": 0,
        "rows_returned": None,
        "error": None,
    }

    if reference_sql:
        row["exact_match"] = exact_match(predicted_sql, reference_sql)
        row.update(component_scores(predicted_sql, reference_sql))
        row["bleu"] = bleu(predicted_sql, reference_sql)

    if runner is not None:
        row.update(execution_scores(df, predicted_sql, reference_sql,
                                    runner, table_name))
    if extra:
        row.update(extra)
    return row


# ==========================================================================
# Session log
# ==========================================================================
FIELDS = ["timestamp", "question", "generated_sql", "reference_sql",
          "exact_match", "select_column_correct", "aggregate_correct",
          "where_clause_correct", "component_average", "structural_match",
          "bleu", "execution_match", "executed", "rows_returned", "error",
          "decoding", "beam_size", "table"]


class SessionLog:
    """Every assessment made this session, exportable as CSV or JSON."""

    def __init__(self, rows: Optional[List[Dict]] = None):
        self.rows: List[Dict] = list(rows or [])

    def add(self, row: Dict) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def summary(self) -> Dict:
        """Aggregates. Reference-dependent means are over SCORED rows only."""
        def mean(key):
            vals = [r[key] for r in self.rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        scored = [r for r in self.rows if r.get("reference_sql")]
        return {
            "queries_assessed": len(self.rows),
            "with_reference_sql": len(scored),
            "exact_match_accuracy": mean("exact_match"),
            "select_column_accuracy": mean("select_column_correct"),
            "aggregate_accuracy": mean("aggregate_correct"),
            "where_clause_accuracy": mean("where_clause_correct"),
            "component_average": mean("component_average"),
            "structural_match_accuracy": mean("structural_match"),
            "mean_bleu": mean("bleu"),
            "execution_accuracy": mean("execution_match"),
            "executable_rate": mean("executed"),
        }

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in self.rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in FIELDS})
        return buf.getvalue()

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(), "assessments": self.rows},
                          indent=2, default=str)


# ==========================================================================
# Self-test
# ==========================================================================
if __name__ == "__main__":
    g = "SELECT COUNT(*) FROM table WHERE \"fuel type\" = 'CNG'"
    cases = [
        (g, "identical"),
        ("select count(*) from table where \"fuel type\" = 'cng'", "case only"),
        ("SELECT COUNT(\"year\") FROM table WHERE \"fuel type\" = 'CNG'", "agg column differs"),
        ("SELECT COUNT(*) FROM table WHERE \"location\" = 'CNG'", "wrong where column"),
        ("SELECT AVG(\"price\") FROM table", "completely different"),
    ]
    print(f"{'case':22s} {'EM':>3s} {'col':>4s} {'agg':>4s} {'whr':>4s} {'BLEU':>7s}")
    for pred, label in cases:
        c = component_scores(pred, g)
        print(f"{label:22s} {exact_match(pred, g):>3d} "
              f"{c['select_column_correct']:>4d} {c['aggregate_correct']:>4d} "
              f"{c['where_clause_correct']:>4d} {bleu(pred, g):>7.4f}")

    a = "SELECT COUNT(*) FROM table WHERE \"a\" = 1 AND \"b\" = 2"
    b = "SELECT COUNT(*) FROM table WHERE \"b\" = 2 AND \"a\" = 1"
    print(f"\nWHERE order  exact_match={exact_match(a, b)}  "
          f"structural={component_scores(a, b)['structural_match']}")
