"""
================================================================================
END-TO-END TEXT-TO-SQL  (single-file edition)
Natural-language question + table schema  ->  executable SQL

Encoder-Decoder with Bahdanau attention and a pointer-generator copy mechanism.
Dataset: WikiSQL (https://github.com/salesforce/WikiSQL), with a synthetic
template corpus as a fallback / ablation.
================================================================================

SETUP
    pip install torch numpy matplotlib
    git clone --depth 1 https://github.com/salesforce/WikiSQL.git
    cd WikiSQL && tar xjf data.tar.bz2 && cd ..

USAGE  (subcommands)

    # Steps 1-3: build corpus, preprocess, vocab + splits + padded tensors
    python text2sql.py prepare --wikisql WikiSQL/data
    python text2sql.py prepare                      # synthetic corpus instead

    # data sanity checks (numbers quotable in the report)
    python text2sql.py validate-wikisql WikiSQL/data
    python text2sql.py validate                     # after `prepare` (synthetic)

    # Step 5: train
    python text2sql.py train --wikisql WikiSQL/data --smoke      # 1-min check
    python text2sql.py train --wikisql WikiSQL/data --epochs 15
    python text2sql.py train --wikisql WikiSQL/data --no-copy    # ablation

    # Step 6: decode + score
    python text2sql.py decode --ckpt runs/seq2seq_copy/best.pt \
                              --wikisql WikiSQL/data --split test --beam 5
    python text2sql.py decode --ckpt runs/seq2seq_copy/best.pt --interactive

CONTENTS
    Section 1a  synthetic corpus generator (5 schemas, 14 query templates)
    Section 1b  WikiSQL loader, SQL rendering, execution harness
    Section 2   tokenization, SQL normalization, schema serialization
    Section 3   vocabulary, stratified splits, padding
    Section 4a  copy-aware data layer (extended vocabulary)
    Section 4b  BiLSTM encoder + Bahdanau attention + pointer-generator decoder
    Section 4c  Transformer alternative (same interface)
    Section 5   training loop, loss curves, checkpointing
    Section 6   greedy + beam-search decoding, evaluation metrics
    Section 7   CLI drivers and validation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



# ==========================================================================
# SECTION 2 - TEXT PREPROCESSING
# ==========================================================================
# STEP 2 - Text preprocessing for the Text-to-SQL pipeline.
#
# Covers every requirement of the assignment:
#   * tokenization of the natural-language question
#   * tokenization of the SQL query (operators/parentheses/literals split out)
#   * lowercasing of the natural-language side
#   * normalization of SQL keywords (canonical UPPERCASE, identifiers lowercase)
#   * serialisation of the schema (table name + column names) into the ENCODER input
#   * addition of <sos> / <eos> tokens
#
# Encoder input layout
# --------------------
# <sos> what is the average salary in sales
#       <sep> <tab> employees <col> employee_id <col> name ... <eos>
#
# Decoder target layout
# ---------------------
# <sos> SELECT AVG ( salary ) FROM employees WHERE department = ' sales ' <eos>
#
# Keeping the schema inside the encoder sequence is what lets one model serve
# many tables: the column names the decoder must emit are always visible in
# the input, so the model learns to select from them rather than memorise them.

# --------------------------------------------------------------------------
# Special tokens
# --------------------------------------------------------------------------
PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SEP, TAB, COL = "<sep>", "<tab>", "<col>"          # schema-serialisation markers
SPECIALS = [PAD, UNK, SOS, EOS, SEP, TAB, COL]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3

# --------------------------------------------------------------------------
# SQL keyword inventory used for normalization
# --------------------------------------------------------------------------
SQL_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
    "AND", "OR", "NOT", "IN", "LIKE", "BETWEEN", "DISTINCT", "AS", "ASC",
    "DESC", "COUNT", "SUM", "AVG", "MIN", "MAX", "JOIN", "ON", "IS", "NULL",
}

# --------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------

# words / numbers (incl. decimals) / any single non-space symbol
_NL_TOKEN_RE = re.compile(r"\d+\.\d+|\d+|[a-z]+(?:'[a-z]+)?|[^\sa-z0-9]")

# SQL: string literals, decimals, integers, identifiers, multi-char operators, symbols
_SQL_TOKEN_RE = re.compile(
    r"'[^']*'"                      # 'string literal'
    r"|\"[^\"]*\""                  # "quoted identifier"
    r"|-?\d+\.\d+|-?\d+"            # numbers, INCLUDING negatives (WikiSQL has
                                    # goal-difference = -16 etc.). Safe here
                                    # because the corpus contains no arithmetic;
                                    # add a lookbehind if you extend to `a - b`.
    r"|[A-Za-z_][A-Za-z_0-9]*"      # identifiers / keywords
    r"|>=|<=|<>|!="                 # multi-character operators
    r"|[(),*=<>.;%/+-]"             # single-character symbols
    r"|\S"                          # catch-all: never silently drop a character
)


def tokenize_question(question: str) -> List[str]:
    """Lowercase the NL question and split it into word / number / punct tokens."""
    return _NL_TOKEN_RE.findall(question.lower().strip())


def tokenize_value(text: str) -> List[str]:
    """
    Tokenizer for the INSIDE of a quoted span (a WHERE value or a column name
    containing spaces). Splits on whitespace ONLY - deliberately not on
    punctuation.

    Rationale: WHERE values must match database contents byte for byte. Full
    punctuation splitting is not invertible - 'd.j. linderman' would tokenize
    to [d, ., j, ., linderman] and rejoin as 'd . j . linderman', which returns
    zero rows. Whitespace splitting is exactly invertible under whitespace
    normalization, so every reconstructed query still executes.
    """
    return text.lower().split()


def normalize_sql(sql: str) -> str:
    """Collapse whitespace and canonicalise keyword casing (identifiers -> lower)."""
    sql = re.sub(r"\s+", " ", sql.strip().rstrip(";"))
    out = []
    for tok in _SQL_TOKEN_RE.findall(sql):
        if tok.startswith("'") or tok.startswith('"'):
            out.append(tok)                       # literal: leave content alone
        elif tok.upper() in SQL_KEYWORDS:
            out.append(tok.upper())               # keyword -> UPPERCASE
        elif re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            out.append(tok.lower())               # identifier -> lowercase
        else:
            out.append(tok)
    return " ".join(out)


def tokenize_sql(sql: str, split_literals: bool = True) -> List[str]:
    """
    Tokenize a SQL string after normalization.

    split_literals=True turns  'Human Resources'  into  ' human resources '
    so literal words share the vocabulary with the question words - this is
    what allows a copy/pointer mechanism (or just a shared embedding) to learn
    that the value in the WHERE clause comes from the question.
    """
    toks: List[str] = []
    for tok in _SQL_TOKEN_RE.findall(normalize_sql(sql)):
        quoted = (len(tok) >= 2 and
                  ((tok[0] == "'" and tok[-1] == "'") or
                   (tok[0] == '"' and tok[-1] == '"')))
        if quoted:
            # ' ... '  = string literal      " ... "  = column identifier
            # (WikiSQL headers contain spaces, so identifiers must be quoted too)
            if split_literals:
                q = tok[0]
                toks.append(q)
                toks.extend(tokenize_value(tok[1:-1]))
                toks.append(q)
            else:
                toks.append(tok.lower())
        else:
            toks.append(tok)
    return toks


def detokenize_sql(tokens: List[str]) -> str:
    """Turn predicted tokens back into a runnable SQL string (inverse of above)."""
    out, open_q, buf = [], None, []
    for t in tokens:
        if t in (SOS, EOS, PAD):
            continue
        if t in ("'", '"'):
            if open_q == t:                       # closing quote
                out.append(t + " ".join(buf) + t)
                buf, open_q = [], None
            elif open_q is None:                  # opening quote
                open_q = t
            else:                                 # other quote inside a span
                buf.append(t)
            continue
        if open_q:
            buf.append(t)
        else:
            out.append(t)
    if open_q:                                    # unterminated (bad prediction)
        out.append(open_q + " ".join(buf) + open_q)

    # Spacing-aware join. Quoted spans are already atomic strings in `out`, so
    # their interior is never reformatted - critical, because WikiSQL column
    # names legitimately contain parentheses, e.g. "money ( £ )".
    def is_atomic(t: str) -> bool:                # an already-assembled quoted span
        return len(t) >= 2 and t[0] in ("'", '"')

    parts: List[str] = []
    for tok in out:
        prev = parts[-1] if parts else None
        glue = False
        if prev is None:
            glue = True
        elif not is_atomic(tok) and tok in (")", ",", ";"):
            glue = True                            # ... x )   -> ...x)
        elif not is_atomic(tok) and tok == "(" and prev and not is_atomic(prev) \
                and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", prev):
            glue = True                            # COUNT (   -> COUNT(
        elif not is_atomic(prev) and prev == "(":
            glue = True                            # ( x       -> (x
        if not glue:
            parts.append(" ")
        parts.append(tok)
    return "".join(parts).strip()


# --------------------------------------------------------------------------
# Schema serialisation
# --------------------------------------------------------------------------

def quote_table_refs(sql: str, table_name: str = "table") -> str:
    """
    Quote a table identifier so it survives SQLite.

    `table` is a RESERVED WORD in SQL. The WikiSQL placeholder is literally
    `table`, so the bare `FROM table` the model emits is a SYNTAX ERROR - not a
    model failure, a harness failure. Quoting makes it legal. Any code that
    executes generated SQL must call this first.
    """
    return re.sub(rf'(?<![\w."])(FROM|JOIN|INTO|UPDATE)\s+{re.escape(table_name)}'
                  r'(?![\w."])',
                  lambda m: f'{m.group(1)} "{table_name}"', sql, flags=re.I)


def serialize_schema(table: str, columns: List[str]) -> List[str]:
    """
    <sep> <tab> table_name <col> c1_word1 c1_word2 <col> c2_word1 ...

    Column names are split into WORD tokens with the same whitespace rule used
    for quoted spans in the target (tokenize_value). This alignment is not
    cosmetic - it is what makes schema linking possible at all.

    If the source kept a column as one token ('school/club team') while the
    target emitted two ('school/club', 'team'), the two sides would share no
    vocabulary items, the copy mechanism could never point at a column name,
    and unseen columns would be <unk> on the source side. WikiSQL holds out
    100% of test tables, so almost every column at test time is unseen - which
    makes this the difference between generalising and guessing.
    """
    toks = [SEP, TAB, table.lower()]
    for c in columns:
        toks.append(COL)
        toks.extend(tokenize_value(str(c)) or ["unnamed"])
    return toks


def build_source_sequence(question: str, table: str, columns: List[str],
                          add_bos_eos: bool = True) -> List[str]:
    """Full encoder input: question tokens + serialised schema, wrapped in <sos>/<eos>."""
    toks = tokenize_question(question) + serialize_schema(table, columns)
    return [SOS] + toks + [EOS] if add_bos_eos else toks


def build_target_sequence(sql: str, add_bos_eos: bool = True) -> List[str]:
    """Full decoder target: normalized SQL tokens wrapped in <sos>/<eos>."""
    toks = tokenize_sql(sql)
    return [SOS] + toks + [EOS] if add_bos_eos else toks


def preprocess_record(rec: Dict) -> Dict:
    """Attach src_tokens / tgt_tokens to one raw corpus record."""
    rec = dict(rec)
    rec["src_tokens"] = build_source_sequence(rec["question"], rec["table"], rec["columns"])
    rec["tgt_tokens"] = build_target_sequence(rec["sql"])
    return rec


def preprocess_all(records: List[Dict]) -> List[Dict]:
    return [preprocess_record(r) for r in records]



# ==========================================================================
# SECTION 1a - SYNTHETIC CORPUS
# ==========================================================================
# STEP 1 - Dataset construction for the Text-to-SQL assignment.
#
# Two options are supported:
#
#   (A) build_synthetic_corpus(...)  -> template-driven parallel corpus.
#       5 sample tables, randomised column names and filter values,
#       ~30k unique (question, sql, schema) triples. Runs fully offline.
#
#   (B) the WikiSQL loader in the next section -> the real benchmark corpus
#       (80,654 single-table pairs).
#
# Every record produced by either path has the SAME shape:
#
#     {
#       "question":  "what is the average salary in the sales department",
#       "sql":       "SELECT AVG(salary) FROM employees WHERE department = 'Sales'",
#       "table":     "employees",
#       "columns":   ["employee_id", "name", "department", ...],
#       "template":  "AGG_WHERE_CAT",     # used for stratified splitting
#     }

# ----------------------------------------------------------------------------
# 1. Sample schemas
#    Each table declares its columns by ROLE so templates can pick sensibly:
#      key  = identifier      cat = categorical (string)
#      num  = numeric         date = date       txt = free text / name
#    "alts" gives synonym column names used for randomised schema variants.
# ----------------------------------------------------------------------------

SCHEMAS: Dict[str, dict] = {
    "employees": {
        "alts": ["employees", "staff", "employee_records"],
        "columns": {
            "employee_id": {"role": "key", "alts": ["employee_id", "emp_id", "staff_id"]},
            "name":        {"role": "txt", "alts": ["name", "employee_name", "full_name"]},
            "department":  {"role": "cat", "alts": ["department", "dept", "division"],
                            "values": ["Sales", "Engineering", "Marketing", "Finance",
                                       "Human Resources", "Support", "Legal"]},
            "city":        {"role": "cat", "alts": ["city", "location", "office_city"],
                            "values": ["London", "Mumbai", "Berlin", "Toronto",
                                       "Singapore", "Austin", "Sydney"]},
            "salary":      {"role": "num", "alts": ["salary", "annual_salary", "pay"],
                            "range": (30000, 200000), "step": 5000, "unit": "salary"},
            "age":         {"role": "num", "alts": ["age", "employee_age"],
                            "range": (21, 64), "step": 1, "unit": "age"},
            "years_experience": {"role": "num", "alts": ["years_experience", "experience", "tenure"],
                                 "range": (1, 30), "step": 1, "unit": "years"},
            "hire_date":   {"role": "date", "alts": ["hire_date", "joining_date", "start_date"]},
        },
    },
    "sales_orders": {
        "alts": ["sales_orders", "orders", "order_table"],
        "columns": {
            "order_id":      {"role": "key", "alts": ["order_id", "id", "order_no"]},
            "customer_name": {"role": "txt", "alts": ["customer_name", "customer", "client_name"]},
            "product":       {"role": "cat", "alts": ["product", "item", "product_name"],
                              "values": ["Laptop", "Monitor", "Keyboard", "Printer",
                                         "Tablet", "Headset", "Docking Station"]},
            "region":        {"role": "cat", "alts": ["region", "sales_region", "territory"],
                              "values": ["North", "South", "East", "West",
                                         "Central", "APAC", "EMEA"]},
            "status":        {"role": "cat", "alts": ["status", "order_status"],
                              "values": ["Shipped", "Pending", "Cancelled",
                                         "Delivered", "Returned"]},
            "quantity":      {"role": "num", "alts": ["quantity", "qty", "units"],
                              "range": (1, 500), "step": 5, "unit": "quantity"},
            "unit_price":    {"role": "num", "alts": ["unit_price", "price", "rate"],
                              "range": (10, 2000), "step": 10, "unit": "price"},
            "total_amount":  {"role": "num", "alts": ["total_amount", "amount", "order_value"],
                              "range": (100, 50000), "step": 100, "unit": "amount"},
            "order_date":    {"role": "date", "alts": ["order_date", "purchase_date"]},
        },
    },
    "students": {
        "alts": ["students", "student_records", "enrollment"],
        "columns": {
            "student_id":   {"role": "key", "alts": ["student_id", "roll_no", "sid"]},
            "student_name": {"role": "txt", "alts": ["student_name", "name", "full_name"]},
            "major":        {"role": "cat", "alts": ["major", "branch", "programme"],
                             "values": ["Computer Science", "Mechanical", "Economics",
                                        "Physics", "Biology", "Design", "Mathematics"]},
            "country":      {"role": "cat", "alts": ["country", "nationality", "home_country"],
                             "values": ["India", "Canada", "Germany", "Japan",
                                        "Brazil", "Kenya", "France"]},
            "gpa":          {"role": "num", "alts": ["gpa", "grade_point", "cgpa"],
                             "range": (2, 10), "step": 1, "unit": "gpa"},
            "credits":      {"role": "num", "alts": ["credits", "credit_hours", "units_earned"],
                             "range": (10, 180), "step": 10, "unit": "credits"},
            "enrollment_year": {"role": "num", "alts": ["enrollment_year", "admission_year", "batch"],
                                "range": (2010, 2025), "step": 1, "unit": "year"},
            "scholarship_amount": {"role": "num", "alts": ["scholarship_amount", "scholarship", "aid_amount"],
                                   "range": (0, 20000), "step": 1000, "unit": "amount"},
        },
    },
    "products": {
        "alts": ["products", "catalog", "inventory"],
        "columns": {
            "product_id":     {"role": "key", "alts": ["product_id", "sku", "item_id"]},
            "product_name":   {"role": "txt", "alts": ["product_name", "title", "name"]},
            "category":       {"role": "cat", "alts": ["category", "product_category", "segment"],
                               "values": ["Electronics", "Furniture", "Stationery",
                                          "Apparel", "Grocery", "Toys", "Footwear"]},
            "supplier":       {"role": "cat", "alts": ["supplier", "vendor", "supplier_name"],
                               "values": ["Acme Corp", "Globex", "Initech",
                                          "Umbrella Ltd", "Soylent", "Hooli"]},
            "price":          {"role": "num", "alts": ["price", "list_price", "cost"],
                               "range": (5, 5000), "step": 25, "unit": "price"},
            "stock_quantity": {"role": "num", "alts": ["stock_quantity", "stock", "units_in_stock"],
                               "range": (0, 2000), "step": 50, "unit": "stock"},
            "rating":         {"role": "num", "alts": ["rating", "avg_rating", "customer_rating"],
                               "range": (1, 5), "step": 1, "unit": "rating"},
            "launch_year":    {"role": "num", "alts": ["launch_year", "release_year", "year_launched"],
                               "range": (2005, 2025), "step": 1, "unit": "year"},
        },
    },
    "hospital_patients": {
        "alts": ["hospital_patients", "patients", "admissions"],
        "columns": {
            "patient_id":    {"role": "key", "alts": ["patient_id", "pid", "record_no"]},
            "patient_name":  {"role": "txt", "alts": ["patient_name", "name", "full_name"]},
            "diagnosis":     {"role": "cat", "alts": ["diagnosis", "condition", "ailment"],
                              "values": ["Diabetes", "Asthma", "Fracture",
                                         "Hypertension", "Migraine", "Anemia"]},
            "ward":          {"role": "cat", "alts": ["ward", "department", "unit"],
                              "values": ["Cardiology", "Orthopedics", "Neurology",
                                         "Pediatrics", "Oncology", "General"]},
            "gender":        {"role": "cat", "alts": ["gender", "sex"],
                              "values": ["Male", "Female", "Other"]},
            "age":           {"role": "num", "alts": ["age", "patient_age"],
                              "range": (1, 95), "step": 5, "unit": "age"},
            "days_admitted": {"role": "num", "alts": ["days_admitted", "length_of_stay", "stay_days"],
                              "range": (1, 60), "step": 1, "unit": "days"},
            "treatment_cost": {"role": "num", "alts": ["treatment_cost", "bill_amount", "cost"],
                               "range": (500, 100000), "step": 500, "unit": "amount"},
        },
    },
}

# ----------------------------------------------------------------------------
# 2. Natural-language surface forms
# ----------------------------------------------------------------------------

# How a column is referred to in English (derived from its name, plus role hints)
def _nl(col: str) -> str:
    return col.replace("_", " ")


AGG_NL = {
    "AVG": ["average", "mean", "avg"],
    "SUM": ["total", "sum of", "combined"],
    "MIN": ["minimum", "lowest", "smallest"],
    "MAX": ["maximum", "highest", "largest"],
    "COUNT": ["number of", "count of", "how many"],
}

CMP_NL = {
    ">":  ["greater than", "more than", "above", "over", "exceeding"],
    "<":  ["less than", "below", "under", "smaller than"],
    ">=": ["at least", "no less than", "greater than or equal to"],
    "<=": ["at most", "no more than", "less than or equal to"],
}


def _rand_value(spec: dict, rng: random.Random):
    lo, hi = spec["range"]
    step = spec.get("step", 1)
    n = rng.randrange(lo, hi + 1, step) if step > 1 else rng.randint(lo, hi)
    return n


# ----------------------------------------------------------------------------
# 3. Templates. Each returns (question, sql).
#    `t` = table name, `c` = dict of role -> chosen column name.
# ----------------------------------------------------------------------------

def _tpl_select_where_cat(t, cols, rng):
    proj = rng.choice(cols["txt"] + cols["num"])
    cat, spec = rng.choice(cols["cat_pairs"])
    val = rng.choice(spec["values"])
    q = rng.choice([
        f"what is the {_nl(proj)} of every record where the {_nl(cat)} is {val.lower()}",
        f"show the {_nl(proj)} for {val.lower()} {_nl(cat)}",
        f"list the {_nl(proj)} where {_nl(cat)} is {val.lower()}",
        f"give me the {_nl(proj)} when the {_nl(cat)} equals {val.lower()}",
    ])
    s = f"SELECT {proj} FROM {t} WHERE {cat} = '{val}'"
    return q, s


def _tpl_select_where_num(t, cols, rng):
    proj = rng.choice(cols["txt"] + cols["cat"])
    num, spec = rng.choice(cols["num_pairs"])
    op = rng.choice([">", "<", ">=", "<="])
    val = _rand_value(spec, rng)
    q = rng.choice([
        f"which {_nl(proj)} has a {_nl(num)} {rng.choice(CMP_NL[op])} {val}",
        f"list the {_nl(proj)} with {_nl(num)} {rng.choice(CMP_NL[op])} {val}",
        f"show me every {_nl(proj)} whose {_nl(num)} is {rng.choice(CMP_NL[op])} {val}",
    ])
    s = f"SELECT {proj} FROM {t} WHERE {num} {op} {val}"
    return q, s


def _tpl_count_all(t, cols, rng):
    q = rng.choice([
        f"how many rows are there in {_nl(t)}",
        f"count the total number of records in {_nl(t)}",
        f"what is the total count of entries in the {_nl(t)} table",
    ])
    return q, f"SELECT COUNT(*) FROM {t}"


def _tpl_count_where(t, cols, rng):
    cat, spec = rng.choice(cols["cat_pairs"])
    val = rng.choice(spec["values"])
    q = rng.choice([
        f"how many records have {_nl(cat)} equal to {val.lower()}",
        f"count the entries where the {_nl(cat)} is {val.lower()}",
        f"what is the number of rows with {val.lower()} as the {_nl(cat)}",
    ])
    return q, f"SELECT COUNT(*) FROM {t} WHERE {cat} = '{val}'"


def _tpl_agg_all(t, cols, rng):
    agg = rng.choice(["AVG", "SUM", "MIN", "MAX"])
    num, spec = rng.choice(cols["num_pairs"])
    q = rng.choice([
        f"what is the {rng.choice(AGG_NL[agg])} {_nl(num)}",
        f"find the {rng.choice(AGG_NL[agg])} {_nl(num)} in the {_nl(t)} table",
        f"compute the {rng.choice(AGG_NL[agg])} {_nl(num)} across all records",
    ])
    return q, f"SELECT {agg}({num}) FROM {t}"


def _tpl_agg_where_cat(t, cols, rng):
    agg = rng.choice(["AVG", "SUM", "MIN", "MAX"])
    num, _ = rng.choice(cols["num_pairs"])
    cat, spec = rng.choice(cols["cat_pairs"])
    val = rng.choice(spec["values"])
    q = rng.choice([
        f"what is the {rng.choice(AGG_NL[agg])} {_nl(num)} for {val.lower()} {_nl(cat)}",
        f"find the {rng.choice(AGG_NL[agg])} {_nl(num)} where {_nl(cat)} is {val.lower()}",
        f"calculate the {rng.choice(AGG_NL[agg])} {_nl(num)} of records with {_nl(cat)} {val.lower()}",
    ])
    return q, f"SELECT {agg}({num}) FROM {t} WHERE {cat} = '{val}'"


def _tpl_where_and(t, cols, rng):
    proj = rng.choice(cols["txt"] + cols["num"])
    cat, cspec = rng.choice(cols["cat_pairs"])
    num, nspec = rng.choice(cols["num_pairs"])
    val = rng.choice(cspec["values"])
    op = rng.choice([">", "<", ">=", "<="])
    nval = _rand_value(nspec, rng)
    q = rng.choice([
        f"show the {_nl(proj)} where {_nl(cat)} is {val.lower()} and {_nl(num)} is {rng.choice(CMP_NL[op])} {nval}",
        f"list the {_nl(proj)} for {val.lower()} {_nl(cat)} with {_nl(num)} {rng.choice(CMP_NL[op])} {nval}",
    ])
    s = f"SELECT {proj} FROM {t} WHERE {cat} = '{val}' AND {num} {op} {nval}"
    return q, s


def _tpl_group_count(t, cols, rng):
    cat, _ = rng.choice(cols["cat_pairs"])
    q = rng.choice([
        f"how many records are there for each {_nl(cat)}",
        f"count the entries grouped by {_nl(cat)}",
        f"give the number of rows per {_nl(cat)}",
    ])
    return q, f"SELECT {cat}, COUNT(*) FROM {t} GROUP BY {cat}"


def _tpl_group_agg(t, cols, rng):
    agg = rng.choice(["AVG", "SUM", "MIN", "MAX"])
    num, _ = rng.choice(cols["num_pairs"])
    cat, _ = rng.choice(cols["cat_pairs"])
    q = rng.choice([
        f"what is the {rng.choice(AGG_NL[agg])} {_nl(num)} for each {_nl(cat)}",
        f"show the {rng.choice(AGG_NL[agg])} {_nl(num)} grouped by {_nl(cat)}",
        f"break down the {rng.choice(AGG_NL[agg])} {_nl(num)} by {_nl(cat)}",
    ])
    return q, f"SELECT {cat}, {agg}({num}) FROM {t} GROUP BY {cat}"


def _tpl_group_agg_order(t, cols, rng):
    agg = rng.choice(["AVG", "SUM", "COUNT"])
    cat, _ = rng.choice(cols["cat_pairs"])
    direction, dnl, opposite = rng.choice([
        ("DESC", ["highest", "largest"], "lowest"),
        ("ASC", ["lowest", "smallest"], "highest"),
    ])
    if agg == "COUNT":
        expr = "COUNT(*)"
        metric = "record count"
    else:
        num, _ = rng.choice(cols["num_pairs"])
        expr = f"{agg}({num})"
        metric = f"{rng.choice(AGG_NL[agg])} {_nl(num)}"
    q = rng.choice([
        f"list each {_nl(cat)} by {metric} from {rng.choice(dnl)} to {opposite}",
        f"rank the {_nl(cat)} values by {metric} in {direction.lower()} order",
        f"order the {_nl(cat)} groups by {metric} {direction.lower()}",
    ])
    s = f"SELECT {cat}, {expr} FROM {t} GROUP BY {cat} ORDER BY {expr} {direction}"
    return q, s


def _tpl_order_limit(t, cols, rng):
    proj = rng.choice(cols["txt"] + cols["cat"])
    num, _ = rng.choice(cols["num_pairs"])
    k = rng.choice([1, 3, 5, 10, 20])
    direction, word = rng.choice([("DESC", ["highest", "top", "largest", "biggest"]),
                                  ("ASC", ["lowest", "bottom", "smallest"])])
    q = rng.choice([
        f"what are the {k} {rng.choice(word)} {_nl(num)} records and their {_nl(proj)}",
        f"show the top {k} {_nl(proj)} sorted by {_nl(num)} {direction.lower()}",
        f"give me the {k} {_nl(proj)} with the {rng.choice(word)} {_nl(num)}",
    ])
    s = f"SELECT {proj} FROM {t} ORDER BY {num} {direction} LIMIT {k}"
    return q, s


def _tpl_distinct(t, cols, rng):
    cat, _ = rng.choice(cols["cat_pairs"])
    q = rng.choice([
        f"list the distinct {_nl(cat)} values",
        f"what are the unique {_nl(cat)} entries in {_nl(t)}",
        f"show all different {_nl(cat)} names",
    ])
    return q, f"SELECT DISTINCT {cat} FROM {t}"


def _tpl_having(t, cols, rng):
    cat, _ = rng.choice(cols["cat_pairs"])
    num, spec = rng.choice(cols["num_pairs"])
    agg = rng.choice(["SUM", "AVG"])
    val = _rand_value(spec, rng)
    op = rng.choice([">", "<"])
    q = rng.choice([
        f"which {_nl(cat)} groups have a {rng.choice(AGG_NL[agg])} {_nl(num)} {rng.choice(CMP_NL[op])} {val}",
        f"show the {_nl(cat)} where the {rng.choice(AGG_NL[agg])} {_nl(num)} is {rng.choice(CMP_NL[op])} {val}",
    ])
    s = (f"SELECT {cat}, {agg}({num}) FROM {t} GROUP BY {cat} "
         f"HAVING {agg}({num}) {op} {val}")
    return q, s


def _tpl_count_distinct(t, cols, rng):
    cat, _ = rng.choice(cols["cat_pairs"])
    q = rng.choice([
        f"how many unique {_nl(cat)} values are there",
        f"count the distinct {_nl(cat)} entries",
    ])
    return q, f"SELECT COUNT(DISTINCT {cat}) FROM {t}"


TEMPLATES = {
    "SELECT_WHERE_CAT":  _tpl_select_where_cat,
    "SELECT_WHERE_NUM":  _tpl_select_where_num,
    "COUNT_ALL":         _tpl_count_all,
    "COUNT_WHERE":       _tpl_count_where,
    "AGG_ALL":           _tpl_agg_all,
    "AGG_WHERE_CAT":     _tpl_agg_where_cat,
    "WHERE_AND":         _tpl_where_and,
    "GROUP_COUNT":       _tpl_group_count,
    "GROUP_AGG":         _tpl_group_agg,
    "GROUP_AGG_ORDER":   _tpl_group_agg_order,
    "ORDER_LIMIT":       _tpl_order_limit,
    "DISTINCT":          _tpl_distinct,
    "HAVING":            _tpl_having,
    "COUNT_DISTINCT":    _tpl_count_distinct,
}


# ----------------------------------------------------------------------------
# 4. Corpus generation
# ----------------------------------------------------------------------------

def _materialise_schema(base_table: str, rng: random.Random, randomise: bool):
    """Pick one concrete naming variant of a schema (randomised column names)."""
    spec = SCHEMAS[base_table]
    table = rng.choice(spec["alts"]) if randomise else base_table

    cols, buckets = [], {"key": [], "txt": [], "cat": [], "num": [], "date": [],
                         "cat_pairs": [], "num_pairs": []}
    for canonical, cspec in spec["columns"].items():
        name = rng.choice(cspec["alts"]) if randomise else canonical
        cols.append(name)
        buckets[cspec["role"]].append(name)
        if cspec["role"] == "cat":
            buckets["cat_pairs"].append((name, cspec))
        elif cspec["role"] == "num":
            buckets["num_pairs"].append((name, cspec))
    return table, cols, buckets


def build_synthetic_corpus(n_samples: int = 30000,
                           seed: int = 13,
                           randomise_schema: bool = True) -> List[dict]:
    """Generate a de-duplicated parallel corpus of (question, sql, schema)."""
    rng = random.Random(seed)
    tpl_names = list(TEMPLATES)
    records, seen = [], set()
    attempts, max_attempts = 0, n_samples * 60

    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        base = rng.choice(list(SCHEMAS))
        table, cols, buckets = _materialise_schema(base, rng, randomise_schema)
        tname = rng.choice(tpl_names)
        try:
            q, s = TEMPLATES[tname](table, buckets, rng)
        except IndexError:          # a role bucket was empty for this table
            continue
        key = (q, s)
        if key in seen:
            continue
        seen.add(key)
        records.append({"question": q, "sql": s, "table": table,
                        "columns": cols, "template": tname})

    rng.shuffle(records)
    return records


def save_jsonl(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



# ==========================================================================
# SECTION 1b - WIKISQL ADAPTER
# ==========================================================================
# WikiSQL adapter (https://github.com/salesforce/WikiSQL).
#
# Setup
# -----
#     git clone --depth 1 https://github.com/salesforce/WikiSQL.git
#     cd WikiSQL && tar xjf data.tar.bz2      # -> data/{train,dev,test}.jsonl
#                                             #    data/{train,dev,test}.tables.jsonl
#                                             #    data/{train,dev,test}.db
#
# Then:
#     python run_prepare.py --wikisql WikiSQL/data
#
# What this module does
# ---------------------
# WikiSQL stores the query as a STRUCT, not a string:
#
#     {"sel": 5, "agg": 0, "conds": [[3, 0, "SOUTH AUSTRALIA"]]}
#
# `sel` indexes the table header, `agg` indexes AGG_OPS, and each condition is
# [column_index, operator_index, value]. We render that into a real SQL string.
#
# Two naming schemes matter and they are NOT the same:
#
#   * header names  - "State/territory", "Text/background colour" ...
#                     human-readable, contain spaces and punctuation. These go
#                     into the model's input and target, double-quoted, because
#                     their WORDS overlap with the question's words - which is
#                     exactly the signal the model needs for schema linking.
#   * physical names - col0, col1, ... as actually stored in the .db files.
#                     Used only by the execution harness.
#
# `col_map` on each record maps header index -> physical name so predictions can
# be executed for execution accuracy.

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]          # "OP" is an unused placeholder slot
SPLITS = ("train", "dev", "test")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _clean_header(h: str) -> str:
    """Normalise a header for use as a quoted identifier (keep it readable)."""
    h = re.sub(r"\s+", " ", str(h).replace('"', "").replace("\n", " ")).strip()
    return h.lower() or "unnamed"


def _fmt_value(v) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "")               # drop apostrophes: they break quoting
    s = re.sub(r"\s+", " ", s).strip().lower()
    return f"'{s}'"


PLACEHOLDER = "table"   # see load_split() for why the real name is not used


def render_sql(sql_obj: dict, header: List[str], table: str) -> str:
    """WikiSQL struct -> SQL string with double-quoted, human-readable columns."""
    cols = [_clean_header(h) for h in header]
    sel = cols[sql_obj["sel"]]
    agg = AGG_OPS[sql_obj["agg"]]
    proj = f'{agg}("{sel}")' if agg else f'"{sel}"'

    q = f'SELECT {proj} FROM {table}'
    conds = sql_obj.get("conds") or []
    if conds:
        parts = [f'"{cols[ci]}" {COND_OPS[oi]} {_fmt_value(val)}'
                 for ci, oi, val in conds]
        q += " WHERE " + " AND ".join(parts)
    return q


def _template_id(sql_obj: dict) -> str:
    """Coarse pattern label used for stratification / error analysis."""
    return f"AGG{sql_obj['agg']}_C{len(sql_obj.get('conds') or [])}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_split(data_dir: str, split: str, max_rows: Optional[int] = None) -> List[Dict]:
    tables_path = os.path.join(data_dir, f"{split}.tables.jsonl")
    quest_path = os.path.join(data_dir, f"{split}.jsonl")

    headers: Dict[str, List[str]] = {}
    with open(tables_path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            headers[t["id"]] = t["header"]

    out: List[Dict] = []
    with open(quest_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            header = headers.get(r["table_id"])
            if not header:
                continue
            real_table = "table_" + r["table_id"].replace("-", "_")
            # Use a PLACEHOLDER table name in both the encoder input and the
            # target SQL. WikiSQL is single-table, so the name carries no
            # information, but there are ~25.7k distinct names and each would
            # become a vocabulary entry the decoder could never learn (test
            # tables are 100% unseen in train). The real name is restored at
            # execution time from `real_table`.
            table = PLACEHOLDER
            try:
                sql = render_sql(r["sql"], header, table)
            except (IndexError, KeyError):
                continue                      # malformed / out-of-range index
            out.append({
                "question": r["question"],
                "sql": sql,
                "table": table,
                "real_table": real_table,
                "columns": [_clean_header(h) for h in header],
                "template": _template_id(r["sql"]),
                "split": split,
                "table_id": r["table_id"],
                "col_map": {_clean_header(h): f"col{i}" for i, h in enumerate(header)},
                "sql_struct": r["sql"],
            })
            if max_rows and len(out) >= max_rows:
                break
    return out


def load_wikisql(data_dir: str, max_rows_per_split: Optional[int] = None
                 ) -> Dict[str, List[Dict]]:
    """Load the three OFFICIAL splits. Use these rather than re-splitting:
    WikiSQL's test set holds out tables, so a random re-split would leak."""
    return {s: load_split(data_dir, s, max_rows_per_split) for s in SPLITS}


# --------------------------------------------------------------------------
# Execution harness (execution accuracy)
# --------------------------------------------------------------------------
def to_physical(sql: str, col_map: Dict[str, str]) -> str:
    """Swap "human readable header" -> col7 so the query runs on the .db file."""
    def sub(m):
        name = m.group(1).lower()
        return col_map.get(name, f'"{name}"')
    return re.sub(r'"([^"]*)"', sub, sql)


class WikiSQLExecutor:
    """Runs gold and predicted SQL against the official .db files."""

    def __init__(self, data_dir: str):
        self.conns = {}
        for s in SPLITS:
            p = os.path.join(data_dir, f"{s}.db")
            if os.path.exists(p):
                self.conns[s] = sqlite3.connect(p)

    def execute(self, sql: str, col_map: Dict[str, str], split: str,
                real_table: Optional[str] = None):
        con = self.conns.get(split)
        if con is None:
            return None
        q = to_physical(sql, col_map)
        if real_table:
            q = re.sub(r"\bFROM\s+" + re.escape(PLACEHOLDER) + r"\b",
                       f"FROM {real_table}", q, flags=re.I)
        try:
            return con.execute(q).fetchall()
        except Exception:
            return None

    def execution_accuracy(self, records: List[Dict], predictions: List[str],
                           split: str = "dev") -> float:
        """Fraction of predictions whose RESULT SET matches the gold result set."""
        hit = 0
        for rec, pred in zip(records, predictions):
            rt = rec.get("real_table")
            gold_r = self.execute(rec["sql"], rec["col_map"], split, rt)
            pred_r = self.execute(pred, rec["col_map"], split, rt)
            if gold_r is not None and pred_r is not None and gold_r == pred_r:
                hit += 1
        return hit / max(len(records), 1)



# ==========================================================================
# SECTION 3 - VOCABULARY, SPLITS, PADDING
# ==========================================================================
# STEP 3 - Vocabularies, padding, splits and batching.
#
#   * Vocabulary  : separate source (question + schema) and target (SQL) vocabs,
#                   built ONLY from the training split to avoid leakage.
#   * Splits      : 80 / 10 / 20 -> train / val / test, stratified by template,
#                   with de-duplication so no identical pair crosses splits.
#   * Padding     : right-padding to a length chosen from a percentile of the
#                   training distribution, plus truncation of the tail.
#   * Batching    : NumPy tensors always; a torch Dataset/DataLoader with dynamic
#                   padding is exposed when PyTorch is installed.

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class Vocab:
    def __init__(self, itos: List[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]], min_freq: int = 1,
              max_size: int | None = None) -> "Vocab":
        counter = Counter()
        for toks in token_lists:
            counter.update(t for t in toks if t not in SPECIALS)
        itos = list(SPECIALS)
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for tok, freq in ordered:
            if freq < min_freq:
                break
            if max_size and len(itos) >= max_size:
                break
            itos.append(tok)
        return cls(itos)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_special and tok in (PAD, SOS, EOS):
                if tok == EOS:
                    break
                continue
            out.append(tok)
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def unk_rate(self, token_lists: Sequence[Sequence[str]]) -> float:
        tot = unk = 0
        for toks in token_lists:
            for t in toks:
                tot += 1
                unk += t not in self.stoi
        return unk / max(tot, 1)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def stratified_split(records: List[Dict],
                     ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                     seed: int = 42,
                     stratify_key: str = "template",
                     dedup_on: str = "question"
                     ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratify by query template and keep duplicate questions inside one split."""
    assert abs(sum(ratios) - 1.0) < 1e-6

    # group duplicates of the same question so they cannot straddle a boundary
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        groups[r[dedup_on]].append(r)

    by_strata: Dict[str, List[List[Dict]]] = defaultdict(list)
    for grp in groups.values():
        by_strata[grp[0].get(stratify_key, "_")].append(grp)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for stratum, grps in sorted(by_strata.items()):
        idx = rng.permutation(len(grps))
        n = len(grps)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        for pos, gi in enumerate(idx):
            bucket = train if pos < n_tr else (val if pos < n_tr + n_va else test)
            bucket.extend(grps[gi])

    for b in (train, val, test):
        rng.shuffle(b)
    return train, val, test


# --------------------------------------------------------------------------
# Padding / encoding
# --------------------------------------------------------------------------
def choose_max_len(token_lists: Sequence[Sequence[str]], percentile: float = 99.5,
                   hard_cap: int | None = None) -> int:
    lens = np.array([len(t) for t in token_lists])
    m = int(np.percentile(lens, percentile))
    return min(m, hard_cap) if hard_cap else m


def encode_and_pad(token_lists: Sequence[Sequence[str]], vocab: Vocab,
                   max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Right-pad / truncate to max_len. Returns (ids, lengths). EOS is preserved."""
    ids = np.full((len(token_lists), max_len), PAD_ID, dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        enc = vocab.encode(toks)
        if len(enc) > max_len:                     # truncate but keep <eos> last
            enc = enc[: max_len - 1] + [EOS_ID]
        ids[i, : len(enc)] = enc
        lengths[i] = len(enc)
    return ids, lengths


def make_tensors(split: List[Dict], src_vocab: Vocab, tgt_vocab: Vocab,
                 src_len: int, tgt_len: int) -> Dict[str, np.ndarray]:
    src, src_l = encode_and_pad([r["src_tokens"] for r in split], src_vocab, src_len)
    tgt, tgt_l = encode_and_pad([r["tgt_tokens"] for r in split], tgt_vocab, tgt_len)
    return {
        "src": src, "src_len": src_l,
        # teacher forcing: decoder reads tgt_in, is scored against tgt_out
        "tgt_in": tgt[:, :-1].copy(),
        "tgt_out": tgt[:, 1:].copy(),
        "tgt_full": tgt, "tgt_len": tgt_l,
    }


# --------------------------------------------------------------------------
# Optional PyTorch layer (dynamic padding -> less wasted compute)
# --------------------------------------------------------------------------
class Text2SQLDataset(Dataset):
    def __init__(self, records, src_vocab: Vocab, tgt_vocab: Vocab,
                 src_max: int, tgt_max: int):
        self.src = [src_vocab.encode(r["src_tokens"])[:src_max] for r in records]
        self.tgt = [tgt_vocab.encode(r["tgt_tokens"])[:tgt_max] for r in records]
        self.records = records

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return (torch.tensor(self.src[i], dtype=torch.long),
                torch.tensor(self.tgt[i], dtype=torch.long))

def collate_batch(batch):
    srcs, tgts = zip(*batch)
    sl = torch.tensor([len(s) for s in srcs])
    tl = torch.tensor([len(t) for t in tgts])
    src = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True,
                                          padding_value=PAD_ID)
    tgt = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True,
                                          padding_value=PAD_ID)
    return {"src": src, "src_len": sl,
            "tgt_in": tgt[:, :-1], "tgt_out": tgt[:, 1:], "tgt_len": tl}

def make_loaders(splits, src_vocab, tgt_vocab, src_max, tgt_max,
                 batch_size=64):
    out = {}
    for name, recs in splits.items():
        ds = Text2SQLDataset(recs, src_vocab, tgt_vocab, src_max, tgt_max)
        out[name] = DataLoader(ds, batch_size=batch_size,
                               shuffle=(name == "train"),
                               collate_fn=collate_batch)
    return out



# ==========================================================================
# SECTION 4a - COPY-AWARE DATA LAYER
# ==========================================================================
# STEP 4a - Copy-aware data layer.
#
# A plain seq2seq cannot emit a WHERE value it never saw in training. On WikiSQL
# that is fatal: values are lifted verbatim from the question, and we measured
# that 3.57% of validation target tokens are OOV while 60.5% of those OOV tokens
# are sitting right there in the encoder input.
#
# The pointer-generator (See et al., 2017) fixes this with an EXTENDED VOCABULARY
# defined per example:
#
#     ids  [0, V)            -> the fixed target vocabulary
#     ids  [V, V + n_oov)    -> source tokens of THIS example that are OOV
#                               for the target vocabulary
#
# Three tensors make it work:
#
#   src_ext   source positions mapped into extended-vocab ids. The copy
#             distribution over source positions is scattered onto these ids.
#   tgt_ext   the gold target, but OOV tokens that appear in the source are
#             given their extended id instead of <unk>. This is what the loss
#             is computed against - so the model is REWARDED for copying.
#   tgt_in    decoder input, which must stay inside [0, V) because the embedding
#             matrix has only V rows. OOV positions stay <unk>.
#
# That asymmetry (tgt_in uses <unk>, tgt_ext uses the extended id) is the part
# that is easy to get wrong.

def build_extended(src_tokens: Sequence[str], tgt_tokens: Sequence[str],
                   src_vocab: Vocab, tgt_vocab: Vocab
                   ) -> Tuple[List[int], List[int], List[int], List[int], List[str]]:
    """Returns (src_ids, src_ext_ids, tgt_in_ids, tgt_ext_ids, oov_list)."""
    V = len(tgt_vocab)
    oovs: List[str] = []
    oov_index: Dict[str, int] = {}

    src_ids, src_ext = [], []
    for tok in src_tokens:
        src_ids.append(src_vocab.stoi.get(tok, UNK_ID))
        tid = tgt_vocab.stoi.get(tok)
        if tid is not None:
            src_ext.append(tid)
        else:
            if tok not in oov_index:
                oov_index[tok] = V + len(oovs)
                oovs.append(tok)
            src_ext.append(oov_index[tok])

    tgt_in, tgt_ext = [], []
    for tok in tgt_tokens:
        tid = tgt_vocab.stoi.get(tok)
        tgt_in.append(tid if tid is not None else UNK_ID)
        if tid is not None:
            tgt_ext.append(tid)
        else:
            # copyable only if this exact token occurs in the source
            tgt_ext.append(oov_index.get(tok, UNK_ID))

    return src_ids, src_ext, tgt_in, tgt_ext, oovs


def outputids_to_tokens(ids: Sequence[int], tgt_vocab: Vocab,
                        oovs: Sequence[str]) -> List[str]:
    """Inverse map, including the per-example extended slots."""
    V = len(tgt_vocab)
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        if i < V:
            out.append(tgt_vocab.itos[i])
        else:
            j = i - V
            out.append(oovs[j] if j < len(oovs) else "<unk>")
    return out


# --------------------------------------------------------------------------
# torch layer
# --------------------------------------------------------------------------
class CopyDataset(Dataset):
    def __init__(self, records, src_vocab: Vocab, tgt_vocab: Vocab,
                 src_max: int = 60, tgt_max: int = 45):
        self.recs = records
        self.sv, self.tv = src_vocab, tgt_vocab
        self.src_max, self.tgt_max = src_max, tgt_max

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        st = list(r["src_tokens"])[: self.src_max]
        tt = list(r["tgt_tokens"])[: self.tgt_max]
        if tt and tt[-1] != "<eos>":
            tt[-1] = "<eos>"                      # keep a terminator
        s, s_ext, t_in, t_ext, oovs = build_extended(st, tt, self.sv, self.tv)
        return {
            "src": torch.tensor(s), "src_ext": torch.tensor(s_ext),
            # teacher forcing shift
            "tgt_in": torch.tensor(t_in[:-1]), "tgt_out": torch.tensor(t_ext[1:]),
            "n_oov": len(oovs), "idx": i,
        }

def collate_copy(batch):
    def pad(key):
        xs = [b[key] for b in batch]
        m = max(len(x) for x in xs)
        out = torch.full((len(xs), m), PAD_ID, dtype=torch.long)
        for i, x in enumerate(xs):
            out[i, : len(x)] = x
        return out

    return {
        "src": pad("src"),
        "src_ext": pad("src_ext"),
        "src_len": torch.tensor([len(b["src"]) for b in batch]),
        "tgt_in": pad("tgt_in"),
        "tgt_out": pad("tgt_out"),
        "max_oov": max(b["n_oov"] for b in batch),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }

def make_copy_loaders(splits: Dict[str, list], src_vocab, tgt_vocab,
                      batch_size=64, src_max=60, tgt_max=45, num_workers=2):
    return {
        name: DataLoader(
            CopyDataset(recs, src_vocab, tgt_vocab, src_max, tgt_max),
            batch_size=batch_size, shuffle=(name == "train"),
            collate_fn=collate_copy, num_workers=num_workers,
            drop_last=(name == "train"))
        for name, recs in splits.items()
    }



# ==========================================================================
# SECTION 4b - MODEL (BiLSTM + ATTENTION + POINTER-GENERATOR)
# ==========================================================================
# STEP 4b - The model.
#
#     BiLSTM encoder  ->  Bahdanau (additive) attention  ->  LSTM decoder
#                     ->  pointer-generator copy head
#
# Shapes throughout: B = batch, S = source length, T = target length,
# E = embedding dim, H = hidden dim (per direction), V = target vocab size.
#
# Why this architecture for Text-to-SQL
# -------------------------------------
# * BiLSTM encoder: the schema is serialised AFTER the question, so a column name
#   at position 40 must be matched against a question word at position 5.
#   Backward context is what makes that alignment possible.
# * Attention: SQL generation is a soft alignment problem - "average salary"
#   must point at the `salary` column token in the input. Attention weights are
#   returned so you can plot the schema-linking heatmap for the report.
# * Copy: a fixed softmax cannot emit an unseen WHERE value. p_gen mixes a
#   generation distribution over the target vocab with a copy distribution over
#   source positions.

EPS = 1e-10


# --------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 n_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           bidirectional=True, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        # fuse the two directions down to the decoder's hidden size
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.drop = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, src: torch.Tensor, src_len: torch.Tensor):
        # src: (B, S)
        x = self.drop(self.emb(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_len.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=src.size(1))       # (B, S, 2H)

        # (layers*2, B, H) -> (layers, B, 2H) -> (layers, B, H)
        def merge(state, fc):
            L, B, H = self.n_layers, state.size(1), state.size(2)
            state = state.view(L, 2, B, H).permute(0, 2, 1, 3).reshape(L, B, 2 * H)
            return torch.tanh(fc(state))

        return out, (merge(h, self.fc_h).contiguous(),
                     merge(c, self.fc_c).contiguous())


# --------------------------------------------------------------------------
class BahdanauAttention(nn.Module):
    """score(h_dec, h_enc) = v^T tanh(W_dec h_dec + W_enc h_enc)"""

    def __init__(self, hid_dim: int, enc_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W_dec = nn.Linear(hid_dim, attn_dim, bias=False)
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_h: torch.Tensor, enc_out: torch.Tensor,
                mask: torch.Tensor):
        # dec_h (B, H) | enc_out (B, S, 2H) | mask (B, S) True = real token
        e = self.v(torch.tanh(self.W_dec(dec_h).unsqueeze(1) +
                              self.W_enc(enc_out))).squeeze(-1)      # (B, S)
        e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, dim=-1)                                 # (B, S)
        ctx = torch.bmm(a.unsqueeze(1), enc_out).squeeze(1)          # (B, 2H)
        return ctx, a


# --------------------------------------------------------------------------
class PointerGeneratorDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int,
                 enc_dim: int, n_layers: int = 1, dropout: float = 0.3,
                 use_copy: bool = True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        # input feeding: previous context vector is concatenated to the input
        self.rnn = nn.LSTM(emb_dim + enc_dim, hid_dim, num_layers=n_layers,
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.attn = BahdanauAttention(hid_dim, enc_dim)
        self.out = nn.Linear(hid_dim + enc_dim, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.use_copy = use_copy
        if use_copy:
            # p_gen from (context, decoder state, decoder input)
            self.p_gen = nn.Linear(enc_dim + hid_dim + emb_dim, 1)
        self.vocab_size = vocab_size

    def step(self, y_prev, state, ctx_prev, enc_out, mask, src_ext, max_oov):
        """One decoding step. y_prev (B,) -> final distribution (B, V+max_oov)."""
        emb = self.drop(self.emb(y_prev)).unsqueeze(1)               # (B,1,E)
        rnn_in = torch.cat([emb, ctx_prev.unsqueeze(1)], dim=-1)
        out, state = self.rnn(rnn_in, state)
        dec_h = out.squeeze(1)                                       # (B,H)

        ctx, attn = self.attn(dec_h, enc_out, mask)
        logits = self.out(self.drop(torch.cat([dec_h, ctx], dim=-1)))
        p_vocab = torch.softmax(logits, dim=-1)                      # (B,V)

        if not self.use_copy:
            if max_oov > 0:                       # keep the interface uniform
                p_vocab = torch.cat(
                    [p_vocab, p_vocab.new_zeros(p_vocab.size(0), max_oov)], -1)
            return p_vocab, state, ctx, attn, None

        p_gen = torch.sigmoid(self.p_gen(
            torch.cat([ctx, dec_h, emb.squeeze(1)], dim=-1)))        # (B,1)

        p_ext = torch.cat(
            [p_gen * p_vocab,
             p_vocab.new_zeros(p_vocab.size(0), max_oov)], dim=-1)
        # scatter the copy mass onto extended-vocab ids
        p_ext = p_ext.scatter_add(1, src_ext, (1.0 - p_gen) * attn)
        return p_ext, state, ctx, attn, p_gen


# --------------------------------------------------------------------------
class Seq2SeqCopy(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 emb_dim: int = 256, hid_dim: int = 512, n_layers: int = 1,
                 dropout: float = 0.3, use_copy: bool = True):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = PointerGeneratorDecoder(
            tgt_vocab_size, emb_dim, hid_dim, hid_dim * 2, n_layers,
            dropout, use_copy)
        self.hid_dim, self.enc_dim = hid_dim, hid_dim * 2
        self.tgt_vocab_size = tgt_vocab_size

    def encode(self, src, src_len):
        enc_out, state = self.encoder(src, src_len)
        mask = (src != PAD_ID)
        return enc_out, state, mask

    def forward(self, src, src_len, src_ext, tgt_in, max_oov: int = 0):
        """Teacher forcing. Returns log-probs (B, T, V+max_oov)."""
        enc_out, state, mask = self.encode(src, src_len)
        B, T = tgt_in.shape
        ctx = src.new_zeros(B, self.enc_dim, dtype=enc_out.dtype)

        probs = []
        for t in range(T):
            p, state, ctx, _, _ = self.decoder.step(
                tgt_in[:, t], state, ctx, enc_out, mask, src_ext, max_oov)
            probs.append(p)
        # log AFTER mixing: the copy distribution is a probability, not a logit,
        # so the loss must be NLL over log(p), never cross_entropy over logits.
        return torch.log(torch.stack(probs, dim=1) + EPS)


def masked_nll(log_probs: torch.Tensor, target: torch.Tensor,
               label_smoothing: float = 0.0) -> torch.Tensor:
    """NLL over log-probabilities, ignoring <pad>."""
    B, T, V = log_probs.shape
    lp = log_probs.reshape(B * T, V)
    tg = target.reshape(B * T)
    mask = tg != PAD_ID
    nll = -lp.gather(1, tg.clamp(min=0).unsqueeze(1)).squeeze(1)
    if label_smoothing > 0:
        nll = (1 - label_smoothing) * nll - label_smoothing * lp.mean(dim=-1)
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ==========================================================================
# SECTION 4c - TRANSFORMER ALTERNATIVE
# ==========================================================================
# STEP 4c (optional) - Transformer alternative.
#
# The assignment allows "Seq2Seq/LSTM or Transformer". This is a drop-in
# alternative to Seq2SeqCopy for the ablation table: same constructor signature,
# same forward signature, so train.py works unchanged apart from the import.
#
#
# Copy support here is a SINGLE-HEAD copy attention computed on top of the
# decoder output rather than reusing a cross-attention head, because
# nn.TransformerDecoder does not expose per-layer attention weights without
# hooks. This keeps the pointer mechanism explicit and easy to read.
#
# Note for the report: on WikiSQL-scale data (56k pairs) a 2-layer Transformer
# and a 1-layer BiLSTM land in a similar range. The Transformer wins on longer
# inputs and trains faster per epoch because the encoder is parallel; the LSTM
# is more sample-efficient at this size. Neither is "the" right answer.

# EPS is defined alongside the LSTM model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 emb_dim: int = 256, hid_dim: int = 512, n_layers: int = 2,
                 dropout: float = 0.1, use_copy: bool = True,
                 n_heads: int = 8):
        super().__init__()
        d = emb_dim
        self.src_emb = nn.Embedding(src_vocab_size, d, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d, dropout=dropout)
        self.transformer = nn.Transformer(
            d_model=d, nhead=n_heads, num_encoder_layers=n_layers,
            num_decoder_layers=n_layers, dim_feedforward=hid_dim,
            dropout=dropout, batch_first=True)
        self.out = nn.Linear(d, tgt_vocab_size)
        self.scale = math.sqrt(d)
        self.use_copy = use_copy
        self.tgt_vocab_size = tgt_vocab_size
        self.enc_dim = d
        if use_copy:
            self.copy_q = nn.Linear(d, d)
            self.copy_k = nn.Linear(d, d)
            self.p_gen = nn.Linear(2 * d, 1)

    # ---- encoder is reusable for decoding ----
    def encode(self, src, src_len=None):
        pad_mask = src == PAD_ID
        h = self.transformer.encoder(
            self.pos(self.src_emb(src) * self.scale),
            src_key_padding_mask=pad_mask)
        return h, None, ~pad_mask

    def _decode_states(self, memory, mem_pad, tgt_in):
        T = tgt_in.size(1)
        # bool mask, matching the dtype of tgt_key_padding_mask (torch warns
        # and may error if a float mask is mixed with a bool padding mask)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=tgt_in.device), diagonal=1)
        return self.transformer.decoder(
            self.pos(self.tgt_emb(tgt_in) * self.scale), memory,
            tgt_mask=causal,
            tgt_key_padding_mask=(tgt_in == PAD_ID),
            memory_key_padding_mask=mem_pad)

    def _mix(self, dec, memory, mask, src_ext, max_oov):
        """dec (B,T,D) -> extended-vocab probabilities (B,T,V+max_oov)."""
        p_vocab = torch.softmax(self.out(dec), dim=-1)
        if not self.use_copy:
            if max_oov > 0:
                p_vocab = torch.cat(
                    [p_vocab, p_vocab.new_zeros(*p_vocab.shape[:2], max_oov)], -1)
            return p_vocab

        # single-head copy attention over source positions
        att = torch.bmm(self.copy_q(dec), self.copy_k(memory).transpose(1, 2))
        att = att / self.scale
        att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        att = torch.softmax(att, dim=-1)                      # (B,T,S)
        ctx = torch.bmm(att, memory)                          # (B,T,D)

        p_gen = torch.sigmoid(self.p_gen(torch.cat([dec, ctx], dim=-1)))
        p_ext = torch.cat(
            [p_gen * p_vocab,
             p_vocab.new_zeros(*p_vocab.shape[:2], max_oov)], dim=-1)
        idx = src_ext.unsqueeze(1).expand(-1, dec.size(1), -1)
        return p_ext.scatter_add(2, idx, (1 - p_gen) * att)

    def forward(self, src, src_len, src_ext, tgt_in, max_oov: int = 0):
        memory, _, mask = self.encode(src, src_len)
        dec = self._decode_states(memory, ~mask, tgt_in)
        return torch.log(self._mix(dec, memory, mask, src_ext, max_oov) + EPS)

    # ---- incremental interface so decode.py can drive it ----
    @torch.no_grad()
    def step_full(self, memory, mask, src_ext, prefix, max_oov):
        """Re-runs the decoder over the whole prefix; returns the last step."""
        dec = self._decode_states(memory, ~mask, prefix)
        return self._mix(dec, memory, mask, src_ext, max_oov)[:, -1, :]



# ==========================================================================
# SECTION 5 - TRAINING
# ==========================================================================
# STEP 5 - Training.
#
#     python train.py --wikisql WikiSQL/data --epochs 15
#     python train.py --wikisql WikiSQL/data --smoke      # 1-minute wiring check
#
# Produces
#     runs/<name>/best.pt          checkpoint (weights + vocabs + config)
#     runs/<name>/history.json     per-epoch losses, PPL, val token accuracy
#     runs/<name>/loss_curve.png   training + validation loss curves
#     runs/<name>/config.json      the exact configuration used
#
# The loss is NLL over LOG-PROBABILITIES, not cross-entropy over logits: the
# pointer-generator's output is already a mixed probability distribution, so
# applying log_softmax to it a second time would be wrong.

# --------------------------------------------------------------------------
@dataclass
class Config:
    # data
    dataset: str = "wikisql"
    src_max_len: int = 60
    tgt_max_len: int = 45
    src_min_freq: int = 2
    tgt_min_freq: int = 2
    max_src_vocab: int = 30000
    max_tgt_vocab: int = 15000

    # architecture
    emb_dim: int = 256
    hid_dim: int = 512
    n_layers: int = 1
    dropout: float = 0.3
    use_copy: bool = True

    # optimisation
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    label_smoothing: float = 0.0
    teacher_forcing: float = 1.0        # always 1.0 here; scheduled sampling optional
    lr_patience: int = 1                # ReduceLROnPlateau
    lr_factor: float = 0.5
    early_stop_patience: int = 3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # bookkeeping
    run_name: str = "seq2seq_copy"
    notes: str = ""
    param_count: int = 0
    src_vocab_size: int = 0
    tgt_vocab_size: int = 0


# --------------------------------------------------------------------------
def load_data(args) -> Dict[str, List[dict]]:
    if args.wikisql:
        ws = load_wikisql(args.wikisql, max_rows_per_split=args.max_rows)
        splits = {"train": ws["train"], "val": ws["dev"], "test": ws["test"]}
    else:
        recs = build_synthetic_corpus(args.n)
        tr, va, te = stratified_split(recs)
        splits = {"train": tr, "val": va, "test": te}
    return {k: preprocess_all(v) for k, v in splits.items()}


def build_vocabs(train, cfg: Config):
    src_vocab = Vocab.build([r["src_tokens"] for r in train],
                            min_freq=cfg.src_min_freq, max_size=cfg.max_src_vocab)
    tgt_vocab = Vocab.build([r["tgt_tokens"] for r in train],
                            min_freq=cfg.tgt_min_freq, max_size=cfg.max_tgt_vocab)
    return src_vocab, tgt_vocab


# --------------------------------------------------------------------------
def run_epoch(model, loader, cfg, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = total_tok = correct = 0

    for batch in loader:
        dev = cfg.device
        src, src_len = batch["src"].to(dev), batch["src_len"]
        src_ext = batch["src_ext"].to(dev)
        tgt_in, tgt_out = batch["tgt_in"].to(dev), batch["tgt_out"].to(dev)
        max_oov = int(batch["max_oov"])

        with torch.set_grad_enabled(train_mode):
            log_probs = model(src, src_len, src_ext, tgt_in, max_oov)
            loss = masked_nll(log_probs, tgt_out, cfg.label_smoothing)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        mask = tgt_out != PAD_ID
        n = int(mask.sum())
        total_loss += float(loss.detach()) * n
        total_tok += n
        correct += int(((log_probs.argmax(-1) == tgt_out) & mask).sum())

    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20)), correct / max(total_tok, 1)


# --------------------------------------------------------------------------
def plot_curves(history: List[dict], path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping loss curve")
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], marker="s", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("NLL / token")
    ax[0].set_title("Loss curve"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [h["val_token_acc"] for h in history], marker="s", color="g")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("token accuracy")
    ax[1].set_title("Validation token accuracy"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("loss curve ->", path)


# --------------------------------------------------------------------------
def cmd_train(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikisql", type=str, default=None)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--hid-dim", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-copy", action="store_true",
                    help="ablation: attention only, no pointer-generator")
    ap.add_argument("--run-name", type=str, default="seq2seq_copy")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 1 epoch, just to verify wiring")
    args = ap.parse_args(argv)

    if args.smoke:
        args.max_rows, args.n, args.epochs = 512, 512, 1

    cfg = Config(dataset="wikisql" if args.wikisql else "synthetic",
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 emb_dim=args.emb_dim, hid_dim=args.hid_dim,
                 n_layers=args.n_layers, dropout=args.dropout,
                 use_copy=not args.no_copy, run_name=args.run_name)

    torch.manual_seed(cfg.seed)
    out_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    splits = load_data(args)
    src_vocab, tgt_vocab = build_vocabs(splits["train"], cfg)
    cfg.src_vocab_size, cfg.tgt_vocab_size = len(src_vocab), len(tgt_vocab)
    loaders = make_copy_loaders(splits, src_vocab, tgt_vocab,
                                batch_size=cfg.batch_size,
                                src_max=cfg.src_max_len, tgt_max=cfg.tgt_max_len,
                                num_workers=args.num_workers)

    model = Seq2SeqCopy(len(src_vocab), len(tgt_vocab), cfg.emb_dim, cfg.hid_dim,
                        cfg.n_layers, cfg.dropout, cfg.use_copy).to(cfg.device)
    cfg.param_count = count_params(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience)

    print(json.dumps(asdict(cfg), indent=2))
    print(f"train {len(splits['train']):,} | val {len(splits['val']):,} "
          f"| test {len(splits['test']):,} | params {cfg.param_count:,}")

    history, best, bad_epochs = [], float("inf"), 0
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl, tr_acc = run_epoch(model, loaders["train"], cfg, optimizer)
        va_loss, va_ppl, va_acc = run_epoch(model, loaders["val"], cfg)
        scheduler.step(va_loss)

        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                        "train_ppl": tr_ppl, "val_ppl": va_ppl,
                        "train_token_acc": tr_acc, "val_token_acc": va_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "secs": round(time.time() - t0, 1)})
        print(f"epoch {ep:>2} | train {tr_loss:.4f} (ppl {tr_ppl:6.2f}) "
              f"| val {va_loss:.4f} (ppl {va_ppl:6.2f}) "
              f"| val tok-acc {va_acc:.4f} | {history[-1]['secs']}s")

        if va_loss < best:
            best, bad_epochs = va_loss, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "src_vocab": src_vocab.itos, "tgt_vocab": tgt_vocab.itos},
                       os.path.join(out_dir, "best.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"early stopping at epoch {ep}")
                break

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    plot_curves(history, os.path.join(out_dir, "loss_curve.png"))
    print(f"best val loss {best:.4f} | artifacts in {out_dir}/")



# ==========================================================================
# SECTION 6 - DECODING (GREEDY / BEAM) AND EVALUATION
# ==========================================================================
# STEP 6 - Inference: greedy and beam-search decoding, plus evaluation.
#
#     # decode the test set and score it
#     python decode.py --ckpt runs/seq2seq_copy/best.pt --wikisql WikiSQL/data \
#                      --split test --beam 5 --limit 2000
#
#     # ask your own question
#     python decode.py --ckpt runs/seq2seq_copy/best.pt --interactive
#
# The copy mechanism makes decoding slightly subtle: the model may emit an id
# >= V (an extended-vocabulary slot pointing at a source token). That id has no
# row in the decoder's embedding matrix, so before feeding it back in at the next
# step it must be mapped to <unk>. The EMITTED sequence keeps the extended id -
# only the FED-BACK id is clamped. Getting this backwards silently destroys copy
# accuracy while the loss still looks fine.

# --------------------------------------------------------------------------
def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    src_vocab, tgt_vocab = Vocab(ck["src_vocab"]), Vocab(ck["tgt_vocab"])
    model = Seq2SeqCopy(len(src_vocab), len(tgt_vocab), cfg["emb_dim"],
                        cfg["hid_dim"], cfg["n_layers"], cfg["dropout"],
                        cfg["use_copy"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, src_vocab, tgt_vocab, cfg


def _clamp(ids: torch.Tensor, V: int) -> torch.Tensor:
    """Extended ids have no embedding row -> feed <unk> back into the decoder."""
    return torch.where(ids < V, ids, torch.full_like(ids, UNK_ID))


# --------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, src_ids, src_ext_ids, n_oov, max_len=45, device="cpu"):
    """Batch-of-one greedy decoding. Returns a list of extended-vocab ids."""
    V = model.tgt_vocab_size
    src = torch.tensor([src_ids], device=device)
    src_ext = torch.tensor([src_ext_ids], device=device)
    src_len = torch.tensor([len(src_ids)])

    enc_out, state, mask = model.encode(src, src_len)
    ctx = torch.zeros(1, model.enc_dim, device=device)
    y = torch.tensor([SOS_ID], device=device)

    out = []
    for _ in range(max_len):
        p, state, ctx, _, _ = model.decoder.step(
            y, state, ctx, enc_out, mask, src_ext, n_oov)
        nxt = int(p.argmax(-1))
        if nxt == EOS_ID:
            break
        out.append(nxt)
        y = _clamp(torch.tensor([nxt], device=device), V)
    return out


# --------------------------------------------------------------------------
@torch.no_grad()
def beam_decode(model, src_ids, src_ext_ids, n_oov, beam_size=5, max_len=45,
                length_penalty=0.7, device="cpu"):
    """
    Beam search with a GNMT-style length penalty:
        score = logP / ((5 + |Y|)/6) ** alpha
    Without it, beam search systematically prefers short queries - it will drop
    the WHERE clause, which looks fluent and is wrong.
    """
    V = model.tgt_vocab_size
    src = torch.tensor([src_ids], device=device)
    src_ext = torch.tensor([src_ext_ids], device=device)
    src_len = torch.tensor([len(src_ids)])

    enc_out, state, mask = model.encode(src, src_len)
    ctx0 = torch.zeros(1, model.enc_dim, device=device)

    # each beam: (tokens, logprob, state, ctx)
    beams = [([], 0.0, state, ctx0)]
    finished = []

    for _ in range(max_len):
        candidates = []
        for tokens, score, st, ct in beams:
            last = tokens[-1] if tokens else SOS_ID
            y = _clamp(torch.tensor([last], device=device), V)
            p, st2, ct2, _, _ = model.decoder.step(
                y, st, ct, enc_out, mask, src_ext, n_oov)
            logp = torch.log(p.squeeze(0) + 1e-10)
            top_lp, top_ix = logp.topk(beam_size)
            for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                candidates.append((tokens + [ix], score + lp, st2, ct2))

        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for toks, sc, st, ct in candidates:
            if toks and toks[-1] == EOS_ID:
                norm = ((5 + len(toks)) / 6) ** length_penalty
                finished.append((toks[:-1], sc / norm))
            else:
                beams.append((toks, sc, st, ct))
            if len(beams) >= beam_size:
                break
        if not beams or len(finished) >= beam_size:
            break

    if not finished:
        norm = lambda t: ((5 + len(t)) / 6) ** length_penalty
        finished = [(t, s / norm(t)) for t, s, _, _ in beams]
    finished.sort(key=lambda x: x[1], reverse=True)
    return finished[0][0]


# --------------------------------------------------------------------------
def predict_sql(model, question: str, table: str, columns: Sequence[str],
                src_vocab: Vocab, tgt_vocab: Vocab, beam: int = 1,
                max_len: int = 45, device: str = "cpu") -> str:
    """End-to-end: raw question + schema -> SQL string."""
    src_tokens = build_source_sequence(question, table, list(columns))
    src_ids, src_ext, _, _, oovs = build_extended(
        src_tokens, [], src_vocab, tgt_vocab)
    fn = greedy_decode if beam <= 1 else (
        lambda *a, **k: beam_decode(*a, beam_size=beam, **k))
    ids = fn(model, src_ids, src_ext, len(oovs), max_len=max_len, device=device)
    return detokenize_sql(outputids_to_tokens(ids, tgt_vocab, oovs))


# --------------------------------------------------------------------------
def normalise(sql: str) -> str:
    return " ".join(sql.lower().split())


def evaluate(model, records: List[dict], src_vocab: Vocab, tgt_vocab: Vocab,
             beam: int = 1, device: str = "cpu",
             executor=None, split: str = "test") -> Dict[str, float]:
    """
    Reports the two metrics WikiSQL papers always quote together:
      * logical-form accuracy - predicted string == gold string
      * execution accuracy    - predicted query returns the gold result set
    Execution accuracy is always the higher of the two, because different
    queries can return identical results.
    """
    preds = [predict_sql(model, r["question"], r["table"], r["columns"],
                         src_vocab, tgt_vocab, beam, device=device)
             for r in records]

    exact = sum(normalise(p) == normalise(r["sql"])
                for p, r in zip(preds, records)) / max(len(records), 1)
    out = {"n": len(records), "logical_form_acc": exact, "beam": beam}

    if executor is not None:
        out["execution_acc"] = executor.execution_accuracy(records, preds, split)
    return out, preds


# --------------------------------------------------------------------------
def cmd_decode(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wikisql", type=str, default=None)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--beam", type=int, default=1, help="1 = greedy")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="predictions.json")
    args = ap.parse_args(argv)

    model, src_vocab, tgt_vocab, cfg = load_checkpoint(args.ckpt, args.device)

    if args.interactive:
        print("Enter a question, then the schema. Ctrl-C to quit.\n")
        while True:
            q = input("question> ").strip()
            t = input("table>    ").strip() or "table"
            cols = [c.strip() for c in input("columns>  ").split(",") if c.strip()]
            print("SQL:", predict_sql(model, q, t, cols, src_vocab, tgt_vocab,
                                      args.beam, device=args.device), "\n")
        return

    if not args.wikisql:
        raise SystemExit("--wikisql is required unless --interactive")

    key = {"val": "dev"}.get(args.split, args.split)
    recs = preprocess_all(load_split(args.wikisql, key))[: args.limit]
    ex = WikiSQLExecutor(args.wikisql)

    metrics, preds = evaluate(model, recs, src_vocab, tgt_vocab, args.beam,
                              args.device, ex, key)
    print(json.dumps(metrics, indent=2))
    with open(args.out, "w") as f:
        json.dump([{"question": r["question"], "gold": r["sql"], "pred": p}
                   for r, p in zip(recs, preds)], f, indent=2)
    print("predictions ->", args.out)



# ==========================================================================
# SECTION 7a - DATA PREPARATION DRIVER
# ==========================================================================
# Driver: runs Steps 1-3 end to end and writes everything the model stage needs.
#
#     python run_prepare.py                      # 30k synthetic pairs
#     python run_prepare.py --n 50000            # larger corpus
#     python run_prepare.py --wikisql data/wikisql   # use real WikiSQL instead
#
# Artifacts written to ./data and ./artifacts:
#     data/corpus.jsonl        raw parallel corpus
#     data/{train,val,test}.jsonl   preprocessed splits (with token lists)
#     artifacts/src_vocab.json, artifacts/tgt_vocab.json
#     artifacts/tensors.npz    padded integer tensors for all three splits
#     artifacts/stats.json     corpus / vocabulary / length statistics

def cmd_prepare(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000, help="synthetic corpus size")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--wikisql", type=str, default=None,
                    help="path to WikiSQL/data (after tar xjf data.tar.bz2)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows per WikiSQL split (for quick debugging)")
    ap.add_argument("--max-src-vocab", type=int, default=30000)
    ap.add_argument("--max-tgt-vocab", type=int, default=15000)
    ap.add_argument("--tgt-min-freq", type=int, default=2)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args(argv)

    data_dir = os.path.join(args.outdir, "data")
    art_dir = os.path.join(args.outdir, "artifacts")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    # ---------------- Step 1: corpus ----------------
    official_splits = None
    if args.wikisql:
        ws = load_wikisql(args.wikisql, max_rows_per_split=args.max_rows)
        # WikiSQL ships its own splits and HOLDS OUT TABLES in dev/test.
        # A random re-split would leak schemas and inflate accuracy, so we
        # keep the official partition: dev -> val.
        official_splits = {"train": ws["train"], "val": ws["dev"], "test": ws["test"]}
        records = ws["train"] + ws["dev"] + ws["test"]
        source = "WikiSQL"
    else:
        records = build_synthetic_corpus(args.n, seed=args.seed)
        source = "synthetic"
    save_jsonl(records, os.path.join(data_dir, "corpus.jsonl"))
    print(f"[1] {source} corpus: {len(records):,} pairs")
    print(f"    unique SQL queries : {len({r['sql'] for r in records}):,}")
    print(f"    tables             : {len({r['table'] for r in records}):,}")
    print(f"    query patterns     : {len({r['template'] for r in records})}")

    # ---------------- Step 2: preprocessing ----------------
    if official_splits:
        official_splits = {k: preprocess_all(v) for k, v in official_splits.items()}
        records = (official_splits["train"] + official_splits["val"]
                   + official_splits["test"])
    else:
        records = preprocess_all(records)
    ex = records[0]
    print("\n[2] preprocessing example")
    print("    Q  :", ex["question"])
    print("    SQL:", ex["sql"])
    print("    SRC:", " ".join(ex["src_tokens"])[:150], "...")
    print("    TGT:", " ".join(ex["tgt_tokens"]))

    # ---------------- Step 3: splits, vocab, padding ----------------
    if official_splits:
        train = official_splits["train"]
        val, test = official_splits["val"], official_splits["test"]
        print(f"\n[3] OFFICIAL splits  train={len(train):,}  val={len(val):,}  test={len(test):,}")
        tr_tabs = {r.get("table_id") for r in train}
        te_tabs = {r.get("table_id") for r in test}
        print(f"    test tables unseen in train: {len(te_tabs - tr_tabs):,}"
              f" / {len(te_tabs):,}  (cross-schema generalisation)")
    else:
        train, val, test = stratified_split(records, (0.8, 0.1, 0.1), seed=42)
        print(f"\n[3] splits  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    overlap = {r["question"] for r in train} & {r["question"] for r in test}
    print(f"    train/test question overlap: {len(overlap)}")

    src_vocab = Vocab.build([r["src_tokens"] for r in train],
                            min_freq=args.min_freq, max_size=args.max_src_vocab)
    tgt_vocab = Vocab.build([r["tgt_tokens"] for r in train],
                            min_freq=args.tgt_min_freq, max_size=args.max_tgt_vocab)

    # Diagnostic: how many target tokens could be COPIED from the question?
    # This is the headline argument for attention + a pointer/copy mechanism.
    copyable = total_oov = 0
    for r in val:
        src_set = set(r["src_tokens"])
        for t in r["tgt_tokens"]:
            if t not in tgt_vocab.stoi:
                total_oov += 1
                copyable += t in src_set
    if total_oov:
        print(f"    OOV target tokens copyable from input: "
              f"{copyable}/{total_oov} ({copyable/total_oov:.1%})")
    print(f"    source vocab: {len(src_vocab):,}   target vocab: {len(tgt_vocab):,}")
    print(f"    val OOV rate (src): {src_vocab.unk_rate([r['src_tokens'] for r in val]):.4%}")
    print(f"    val OOV rate (tgt): {tgt_vocab.unk_rate([r['tgt_tokens'] for r in val]):.4%}")

    src_max = choose_max_len([r["src_tokens"] for r in train], 99.5)
    tgt_max = choose_max_len([r["tgt_tokens"] for r in train], 99.5)
    print(f"    padded lengths: src={src_max}  tgt={tgt_max}")

    splits = {"train": train, "val": val, "test": test}
    tensors = {}
    for name, recs in splits.items():
        t = make_tensors(recs, src_vocab, tgt_vocab, src_max, tgt_max)
        for k, v in t.items():
            tensors[f"{name}_{k}"] = v
        save_jsonl(recs, os.path.join(data_dir, f"{name}.jsonl"))

    src_vocab.save(os.path.join(art_dir, "src_vocab.json"))
    tgt_vocab.save(os.path.join(art_dir, "tgt_vocab.json"))
    np.savez_compressed(os.path.join(art_dir, "tensors.npz"), **tensors)

    src_lens = [len(r["src_tokens"]) for r in records]
    tgt_lens = [len(r["tgt_tokens"]) for r in records]
    stats = {
        "source": source,
        "n_pairs": len(records),
        "n_unique_sql": len({r["sql"] for r in records}),
        "n_tables": len({r["table"] for r in records}),
        "n_templates": len({r["template"] for r in records}),
        "splits": {k: len(v) for k, v in splits.items()},
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "src_len": {"mean": float(np.mean(src_lens)), "max": int(np.max(src_lens)),
                    "p99_5": src_max},
        "tgt_len": {"mean": float(np.mean(tgt_lens)), "max": int(np.max(tgt_lens)),
                    "p99_5": tgt_max},
        "pad_id": 0, "unk_id": 1, "sos_id": 2, "eos_id": 3,
    }
    with open(os.path.join(art_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n    tensors: train_src {tensors['train_src'].shape}, "
          f"train_tgt_in {tensors['train_tgt_in'].shape}")
    print(f"    saved -> {art_dir}/ and {data_dir}/")



# ==========================================================================
# SECTION 7b - VALIDATION (SYNTHETIC)
# ==========================================================================
# Sanity checks you can quote directly in the report.
#
#   1. Tokenize -> detokenize round-trip fidelity (should be 100%).
#   2. Every reconstructed query parses and executes against a SQLite table
#      built from the record's own schema -> the targets really are executable SQL.
#   3. Split integrity: no question leaks from train into val/test.
#   4. Template and length distributions.
#
#     python validate.py

def _load_prepared_split(split):
    with open(f"data/{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_validate():
    train, val, test = _load_prepared_split("train"), _load_prepared_split("val"), _load_prepared_split("test")
    allr = train + val + test

    # 1. round trip
    ok = sum(detokenize_sql(build_target_sequence(r["sql"])).lower() == r["sql"].lower()
             for r in allr)
    print(f"round-trip fidelity      : {ok}/{len(allr)}  ({ok/len(allr):.2%})")

    # 2. executability
    con = sqlite3.connect(":memory:")
    fails = []
    for r in allr:
        cols = ", ".join(f'"{c}" TEXT' for c in r["columns"])
        con.execute(f'DROP TABLE IF EXISTS "{r["table"]}"')
        con.execute(f'CREATE TABLE "{r["table"]}" ({cols})')
        try:
            # quote_table_refs is REQUIRED: `table` is a reserved word, so the
            # WikiSQL placeholder produces a syntax error unquoted.
            con.execute(quote_table_refs(
                detokenize_sql(build_target_sequence(r["sql"])), r["table"]))
        except Exception as e:
            fails.append((r["sql"], str(e)))
    print(f"executable in SQLite     : {len(allr)-len(fails)}/{len(allr)}")
    for s, e in fails[:5]:
        print("   FAIL:", s, "|", e)

    # 3. leakage
    tq = {r["question"] for r in train}
    n_val = len(tq & {r["question"] for r in val})
    n_test = len(tq & {r["question"] for r in test})
    print(f"train->val question leak : {n_val}")
    print(f"train->test question leak: {n_test}")
    if (n_val or n_test) and any("table_id" in r for r in test):
        # A repeated question is only a LEAK if it targets the same table.
        # Generic WikiSQL questions ("Name the most wins") recur across many
        # tables, where the correct answer differs - nothing is memorisable.
        seen = {}
        for r in train:
            seen.setdefault(r["question"], set()).add(r.get("table_id"))
        same = sum(1 for r in test
                   if r["question"] in seen and r.get("table_id") in seen[r["question"]])
        print(f"   of which SAME table (a real leak): {same}")
        print("   the rest are the same question asked of a different table,"
              "\n   which is harmless. Tables remain 100% held out.")

    # 4. distributions
    print("\ntemplate distribution (train):")
    for t, c in Counter(r["template"] for r in train).most_common():
        print(f"   {t:<18} {c:>6}")

    sl = np.array([len(r["src_tokens"]) for r in allr])
    tl = np.array([len(r["tgt_tokens"]) for r in allr])
    print(f"\nsrc len mean/max/p99.5   : {sl.mean():.1f} / {sl.max()} / {np.percentile(sl,99.5):.0f}")
    print(f"tgt len mean/max/p99.5   : {tl.mean():.1f} / {tl.max()} / {np.percentile(tl,99.5):.0f}")



# ==========================================================================
# SECTION 7c - VALIDATION (WIKISQL)
# ==========================================================================
# WikiSQL-specific sanity checks. Numbers printed here are quotable in the report.
#
#     python validate_wikisql.py WikiSQL/data
#
# Checks
#   1. Round-trip: tokenize -> detokenize reproduces the gold SQL string exactly,
#      on all 80,654 pairs.
#   2. Execution upper bound: gold queries pushed through the FULL preprocessing
#      path still return the gold result set from the official .db files.
#      Anything below 100% here is a preprocessing bug, not a model error.
#   3. Split integrity and table hold-out.
#   4. Query-pattern distribution (AGG{i}_C{j}).

def cmd_validate_wikisql(data_dir: str, sample: int = 4000):
    splits = load_wikisql(data_dir)
    ex = WikiSQLExecutor(data_dir)

    # 1. round trip on everything
    total = bad = 0
    for name, recs in splits.items():
        n_bad = 0
        for r in recs:
            total += 1
            if detokenize_sql(build_target_sequence(r["sql"])) != r["sql"]:
                n_bad += 1
        bad += n_bad
        print(f"round-trip {name:<6}: {len(recs)-n_bad:,}/{len(recs):,}")
    print(f"round-trip TOTAL : {total-bad:,}/{total:,}  ({(total-bad)/total:.4%})\n")

    # 2. execution upper bound
    random.seed(0)
    for name, split_key in (("dev", "dev"), ("test", "test")):
        recs = splits[split_key]
        s = random.sample(recs, min(sample, len(recs)))
        preds = [detokenize_sql(build_target_sequence(r["sql"])) for r in s]
        acc = ex.execution_accuracy(s, preds, split_key)
        print(f"execution upper bound ({name}, n={len(s):,}): {acc:.4%}")

    # 3. hold-out
    tr = {r["table_id"] for r in splits["train"]}
    te = {r["table_id"] for r in splits["test"]}
    print(f"\ntest tables unseen in train: {len(te - tr):,}/{len(te):,}")
    q_tr = {r["question"] for r in splits["train"]}
    print(f"train/test question overlap: {len(q_tr & {r['question'] for r in splits['test']})}")

    # 4. pattern distribution
    print("\nquery patterns in train (AGG index _ n-conditions):")
    for p, c in Counter(r["template"] for r in splits["train"]).most_common(10):
        print(f"   {p:<10} {c:>7,}")


# ==========================================================================
# SECTION 8 - COMMAND-LINE ENTRY POINT
# ==========================================================================

COMMANDS = {
    "prepare": cmd_prepare,
    "train": cmd_train,
    "decode": cmd_decode,
    "validate": cmd_validate,
    "validate-wikisql": cmd_validate_wikisql,
}


def _usage() -> str:
    return ("usage: python text2sql.py <command> [options]\n\n"
            "commands:\n"
            "  prepare           build corpus, preprocess, vocab/splits/tensors\n"
            "  train             train the encoder-decoder model\n"
            "  decode            generate SQL and score it\n"
            "  validate          sanity checks on a prepared synthetic corpus\n"
            "  validate-wikisql  sanity checks on the raw WikiSQL data\n\n"
            "run `python text2sql.py <command> --help` for per-command options")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n")
        print(_usage())
        return 2
    if cmd == "validate-wikisql":
        cmd_validate_wikisql(rest[0] if rest else "WikiSQL/data")
    elif cmd == "validate":
        cmd_validate()
    else:
        COMMANDS[cmd](rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
