"""
STEP 7 - Streamlit web application.

Live Text-to-SQL generation with per-query assessment history.

The user enters a question and, when evaluating the model, an Expected SQL
(Ground Truth) for that question. Each generation stores:
  - question
  - raw model SQL
  - final generated SQL
  - expected SQL
  - exact-match accuracy
  - component-wise accuracy
  - execution accuracy
  - BLEU-4
  - execution details / result row count

The full assessment history can be downloaded as CSV or XLSX.
"""

from __future__ import annotations

import difflib
import glob
import io
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime

import math
import pandas as pd
import streamlit as st
import torch

from text2sql import (
    load_checkpoint,
    predict_sql_with_values,
    quote_table_refs,
)

st.set_page_config(page_title="Text-to-SQL", page_icon="🗄️", layout="wide")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TABLE = "table"


# ============================================================================
# CHECKPOINT / MODEL
# ============================================================================
def resolve_ckpt(path: str) -> str | None:
    if not path:
        return None
    for cand in (path, os.path.join(os.getcwd(), path), os.path.join(APP_DIR, path)):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


@st.cache_data(show_spinner=False)
def discover_ckpts() -> list[str]:
    found = []
    for root in {os.getcwd(), APP_DIR}:
        for pat in ("runs/*/*.pt", "*.pt", "runs/*.pt"):
            found += glob.glob(os.path.join(root, pat))
    return sorted({os.path.abspath(f) for f in found})


@st.cache_resource(show_spinner="Loading model…")
def get_model(ckpt_path: str, device: str):
    return load_checkpoint(ckpt_path, device)


# ============================================================================
# CSV / SQLITE EXECUTION
# ============================================================================
def norm_ws(s: str) -> str:
    return " ".join((s or "").split())


def _sqlite_type(dtype) -> str:
    k = getattr(dtype, "kind", "O")
    if k in "iu":
        return "INTEGER"
    if k == "f":
        return "REAL"
    if k == "b":
        return "INTEGER"
    return "TEXT"


def create_table(con: sqlite3.Connection, df: pd.DataFrame, table_name: str,
                 nocase: bool = True) -> None:
    cols = []
    for name, dtype in zip(df.columns, df.dtypes):
        t = _sqlite_type(dtype)
        collate = " COLLATE NOCASE" if (nocase and t == "TEXT") else ""
        cols.append(f'"{name}" {t}{collate}')
    con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    con.execute(f'CREATE TABLE "{table_name}" ({", ".join(cols)})')


def run_sql(df: pd.DataFrame | None, sql: str, table_name: str = MODEL_TABLE,
            nocase: bool = True):
    if df is None:
        return None, "No CSV data is loaded."
    con = sqlite3.connect(":memory:")
    try:
        create_table(con, df, table_name, nocase)
        df.to_sql(table_name, con, index=False, if_exists="append")
        for alias in ("data", "df", "t"):
            if alias != table_name:
                try:
                    con.execute(f'CREATE VIEW "{alias}" AS SELECT * FROM "{table_name}"')
                except sqlite3.Error:
                    pass
        return pd.read_sql_query(quote_table_refs(sql, table_name), con), None
    except Exception as e:
        return None, str(e)
    finally:
        con.close()


def validate_sql(sql: str, columns) -> list[str]:
    msgs, s = [], (sql or "").strip()
    if not s:
        return ["Query is empty."]
    if s.count("'") % 2:
        msgs.append("Odd number of single quotes - a string literal looks unclosed.")
    if s.count('"') % 2:
        msgs.append("Odd number of double quotes - an identifier looks unclosed.")
    if s.count("(") != s.count(")"):
        msgs.append("Unbalanced parentheses.")
    if not re.match(r"^\s*(SELECT|WITH)\b", s, re.I):
        msgs.append("Only SELECT queries can be run here.")
    known = {str(c).lower() for c in (columns or [])}
    if known:
        used = {m.lower() for m in re.findall(r'"([^"]+)"', s)}
        unknown = sorted(u for u in used if u not in known and u != MODEL_TABLE)
        if unknown:
            msgs.append("Not a column in this table: " + ", ".join(f'"{u}"' for u in unknown))
    return msgs


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("_", " ")).strip().lower()
                  for c in df.columns]
    return df


_NUM_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[a-zA-Z/%]+\s*$")


def strip_units(df: pd.DataFrame, min_ratio: float = 0.8) -> tuple[pd.DataFrame, list[str]]:
    df, converted = df.copy(), []
    for col in df.columns:
        if df[col].dtype.kind in "ifb":
            continue
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        hits = vals.str.match(_NUM_UNIT)
        if hits.mean() >= min_ratio:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.extract(_NUM_UNIT, expand=False),
                errors="coerce",
            )
            converted.append(col)
    return df, converted


# ============================================================================
# LIVE ASSESSMENT METRICS
# ============================================================================
def _sql_norm(sql: str) -> str:
    s = quote_table_refs(sql or "", MODEL_TABLE)
    s = re.sub(r";+$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip().lower()


def _extract_clause(sql: str, keyword: str, end_keywords=()) -> str:
    s = _sql_norm(sql)
    m = re.search(rf"\b{re.escape(keyword.lower())}\b", s)
    if not m:
        return ""
    end = len(s)
    for k in end_keywords:
        km = re.search(rf"\b{re.escape(k.lower())}\b", s[m.end():])
        if km:
            end = min(end, m.end() + km.start())
    return s[m.start():end].strip()


def _split_and(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"\s+and\s+", text, flags=re.I) if x.strip()]


def _component_signature(sql: str) -> dict[str, str]:
    s = _sql_norm(sql)
    select = _extract_clause(s, "select", ("from",))
    from_clause = _extract_clause(s, "from", ("where", "group by", "having", "order by", "limit"))
    where = _extract_clause(s, "where", ("group by", "having", "order by", "limit"))
    group_by = _extract_clause(s, "group by", ("having", "order by", "limit"))
    order_by = _extract_clause(s, "order by", ("limit",))
    limit = _extract_clause(s, "limit", ())
    aggs = sorted(set(re.findall(r"\b(count|avg|sum|min|max)\s*\(", select, flags=re.I)))
    where_conds = sorted(_split_and(where[5:].strip()) if where.startswith("where") else [])
    return {
        "SELECT": select,
        "FROM": from_clause,
        "AGGREGATION": ",".join(aggs),
        "WHERE": " AND ".join(where_conds),
        "GROUP BY": group_by,
        "ORDER BY": order_by,
        "LIMIT": limit,
    }


def component_accuracy(pred: str, gold: str) -> tuple[float, dict[str, float]]:
    p = _component_signature(pred)
    g = _component_signature(gold)
    scores = {k: float(p[k] == g[k]) for k in p}
    return sum(scores.values()) / max(len(scores), 1), scores


def _bleu4(reference: str, hypothesis: str) -> float:
    ref = re.findall(r'"[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z_0-9]*|\S', _sql_norm(reference))
    hyp = re.findall(r'"[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z_0-9]*|\S', _sql_norm(hypothesis))
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp)-n+1))
        ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        if not hyp_ngrams:
            precisions.append(1.0 if len(hyp) < n else 0.0)
            continue
        clipped = sum(min(c, ref_ngrams[ng]) for ng, c in hyp_ngrams.items())
        precisions.append((clipped + 1.0) / (sum(hyp_ngrams.values()) + 1.0))
    geo = math.exp(sum(math.log(max(p, 1e-12)) for p in precisions) / 4.0)
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / len(hyp))
    return float(geo * bp)


def _execution_match(df: pd.DataFrame | None, pred: str, gold: str,
                     nocase: bool) -> tuple[bool | None, str, int | None]:
    if df is None:
        return None, "CSV required for execution accuracy.", None
    pred_df, pred_err = run_sql(df, pred, MODEL_TABLE, nocase)
    gold_df, gold_err = run_sql(df, gold, MODEL_TABLE, nocase)
    if pred_err:
        return False, f"Generated SQL error: {pred_err}", None
    if gold_err:
        return False, f"Expected SQL error: {gold_err}", None
    try:
        same = pred_df.reset_index(drop=True).equals(gold_df.reset_index(drop=True))
        return bool(same), "", int(len(pred_df))
    except Exception as e:
        return False, str(e), None


def assess_live_query(question: str, predicted_sql: str, expected_sql: str,
                      df: pd.DataFrame | None, nocase: bool) -> dict:
    expected_sql = (expected_sql or "").strip()
    pred_sql = (predicted_sql or "").strip()
    exact = bool(expected_sql) and (_sql_norm(pred_sql) == _sql_norm(expected_sql))
    comp_mean, comp = component_accuracy(pred_sql, expected_sql) if expected_sql else (None, {})
    exec_ok, exec_note, generated_rows = _execution_match(
        df, pred_sql, expected_sql, nocase) if expected_sql else (None, "Expected SQL not supplied.", None)
    bleu = _bleu4(expected_sql, pred_sql) if expected_sql else None
    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Question": question,
        "Raw Model SQL": st.session_state.get("raw_sql", ""),
        "Generated SQL": pred_sql,
        "Expected SQL": expected_sql,
        "Exact-Match Accuracy": exact if expected_sql else None,
        "Component-wise Accuracy": comp_mean,
        "Execution Accuracy": exec_ok,
        "BLEU-4": bleu,
        "SELECT Acc.": comp.get("SELECT"),
        "FROM Acc.": comp.get("FROM"),
        "Aggregation Acc.": comp.get("AGGREGATION"),
        "WHERE Acc.": comp.get("WHERE"),
        "GROUP BY Acc.": comp.get("GROUP BY"),
        "ORDER BY Acc.": comp.get("ORDER BY"),
        "LIMIT Acc.": comp.get("LIMIT"),
        "Generated Rows": generated_rows,
        "Execution Note": exec_note,
        "Executed SQL": "",
        "Executed Rows": None,
    }


def history_csv_bytes(history: list[dict]) -> bytes:
    return pd.DataFrame(history).to_csv(index=False).encode("utf-8")


def history_xlsx_bytes(history: list[dict]) -> bytes | None:
    out = io.BytesIO()
    try:
        engine = None
        for candidate in ("openpyxl", "xlsxwriter"):
            try:
                __import__(candidate)
                engine = candidate
                break
            except ImportError:
                continue
        if engine is None:
            return None
        with pd.ExcelWriter(out, engine=engine) as writer:
            df_hist = pd.DataFrame(history)
            df_hist.to_excel(writer, sheet_name="Assessment History", index=False)
            if not df_hist.empty:
                summary = pd.DataFrame({
                    "Metric": [
                        "Exact-Match Accuracy",
                        "Component-wise Accuracy",
                        "Execution Accuracy",
                        "BLEU-4",
                        "Queries Assessed",
                    ],
                    "Value": [
                        pd.to_numeric(df_hist["Exact-Match Accuracy"], errors="coerce").mean(),
                        pd.to_numeric(df_hist["Component-wise Accuracy"], errors="coerce").mean(),
                        pd.to_numeric(df_hist["Execution Accuracy"], errors="coerce").mean(),
                        pd.to_numeric(df_hist["BLEU-4"], errors="coerce").mean(),
                        len(df_hist),
                    ],
                })
                summary.to_excel(writer, sheet_name="Summary", index=False)
        return out.getvalue()
    except Exception:
        return None


# ============================================================================
# UI
# ============================================================================
st.title("🗄️ Text-to-SQL")
st.caption("Pipeline: decoder-level SQL constraints v5")
st.caption("Each generated query can be assessed live against a supplied ground-truth SQL.")

with st.sidebar:
    st.header("Model")
    found = discover_ckpts()
    if found:
        labels = [os.path.relpath(f, os.getcwd()) if f.startswith(os.getcwd()) else f
                  for f in found]
        pick = st.selectbox("Detected checkpoints", labels + ["Other path…"])
        ckpt_raw = (st.text_input("Checkpoint path", value="runs/seq2seq_copy/best.pt")
                    if pick == "Other path…" else found[labels.index(pick)])
    else:
        ckpt_raw = st.text_input("Checkpoint path", value="runs/seq2seq_copy/best.pt")
    device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    st.divider()
    st.header("Decoding")
    strategy = st.radio("Strategy", ["Greedy", "Beam search"], horizontal=True)
    beam = st.slider("Beam size", 2, 10, 5) if strategy == "Beam search" else 1
    max_len = st.slider("Max SQL length (tokens)", 10, 80, 45)
    st.divider()
    auto_exec = st.checkbox("Execute query automatically", value=True)
    nocase = st.checkbox("Case-insensitive text matching", value=True,
                         help="Declares text columns COLLATE NOCASE so case differences in text filters do not change results.")
    constrain = st.checkbox("Restrict columns to the schema", value=True,
                            help="Forces generated column identifiers to be real columns.")
    clean_units = st.checkbox("Strip units from numeric-looking text columns", value=True,
                              help="Turns values such as '26.6 kmpl' into numeric values for aggregation and ordering.")

ckpt = resolve_ckpt(ckpt_raw)
model = None
if ckpt:
    try:
        model, src_vocab, tgt_vocab, cfg = get_model(ckpt, device)
        with st.sidebar:
            st.success("Model loaded")
            st.caption(f"{cfg['emb_dim']}d emb · {cfg['hid_dim']}d hidden · copy {'on' if cfg['use_copy'] else 'off'} · trained on {cfg['dataset']}")
    except Exception as e:
        st.sidebar.error(f"Could not load checkpoint: {e}")
else:
    with st.sidebar:
        st.warning("No checkpoint found at that path.")
        st.caption("Use an absolute checkpoint path or launch Streamlit from the project directory.")

# -----------------------------------------------------------------------------
# 1. Schema
# -----------------------------------------------------------------------------
st.subheader("1 · Your table")
mode = st.radio("Schema source", ["Upload a CSV", "Enter columns manually"], horizontal=True,
                label_visibility="collapsed")

df, columns, table_name = None, [], MODEL_TABLE

if mode == "Upload a CSV":
    up = st.file_uploader("Upload a .csv file", type=["csv"])
    if up is not None:
        try:
            df = normalise_columns(pd.read_csv(up))
            converted = []
            if clean_units:
                df, converted = strip_units(df)
            columns = list(df.columns)
            table_name = re.sub(r"\W+", "_", os.path.splitext(up.name)[0]).lower()
            st.success(f"Read **{len(df):,} rows × {len(columns)} columns** from `{up.name}`")
            if converted:
                st.info("Converted numeric-looking columns: " + ", ".join(f"`{c}`" for c in converted))
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(df.head(8), width="stretch")
            with c2:
                st.write("**Detected schema**")
                st.dataframe(pd.DataFrame({"column": columns, "type": [str(t) for t in df.dtypes]}),
                             width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
else:
    table_name = st.text_input("Table name", value="employees")
    raw = st.text_input("Columns (comma-separated)", value="employee_id, name, department, salary, hire_date")
    columns = [re.sub(r"\s+", " ", c.replace("_", " ")).strip().lower() for c in raw.split(",") if c.strip()]
    if columns:
        st.caption(f"{len(columns)} columns: " + ", ".join(f"`{c}`" for c in columns))

# -----------------------------------------------------------------------------
# 2. Question + optional ground truth
# -----------------------------------------------------------------------------
st.subheader("2 · Your question")
st.caption("Example: How many cars have fuel type Petrol and transmission Manual?")
question = st.text_input("Question", placeholder="Enter a natural-language question…", label_visibility="collapsed")
expected_sql = st.text_area(
    "Expected SQL (Ground Truth — for assessment)",
    placeholder="Paste the expected SQL here to calculate Exact-Match, Component-wise, Execution Accuracy and BLEU-4. Leave blank if you only want generation.",
    height=90,
)

go = st.button("Generate SQL", type="primary", disabled=not (model and columns and question.strip()))

if not columns:
    st.info("Upload a CSV or enter columns to continue.")

# -----------------------------------------------------------------------------
# 3. Generate + assess live
# -----------------------------------------------------------------------------
if go:
    with st.spinner("Generating…"):
        result = predict_sql_with_values(
            model, question, MODEL_TABLE, columns, src_vocab, tgt_vocab,
            beam=beam, max_len=max_len, device=device, df_or_rows=df,
            constrain_columns=constrain,
        )

    raw_sql = result["raw_sql"]
    sql = result["sql"]

    st.session_state["raw_sql"] = raw_sql
    st.session_state["sql"] = sql
    st.session_state["editor"] = sql
    st.session_state["src_tokens"] = result["src_tokens"]
    st.session_state["out_tokens"] = result["out_tokens"]
    st.session_state["attn"] = result["attn"]
    st.session_state["current_question"] = question
    st.session_state["current_assessment"] = assess_live_query(
        question, sql, expected_sql, df, nocase
    )

    history = st.session_state.setdefault("assessment_history", [])
    history.append(st.session_state["current_assessment"].copy())

# -----------------------------------------------------------------------------
# 3. Generated SQL
# -----------------------------------------------------------------------------
if "sql" in st.session_state:
    st.subheader("3 · Generated SQL")
    st.code(st.session_state["sql"], language="sql")
    st.caption("Raw model output")
    st.code(st.session_state.get("raw_sql", ""), language="sql")

    current = st.session_state.get("current_assessment")
    if current and current.get("Expected SQL"):
        st.subheader("Live Assessment")
        values = [current.get("Exact-Match Accuracy"), current.get("Component-wise Accuracy"),
                  current.get("Execution Accuracy"), current.get("BLEU-4")]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Exact Match", "N/A" if values[0] is None else ("100%" if values[0] else "0%"))
        c2.metric("Component-wise", "N/A" if values[1] is None else f"{values[1]:.1%}")
        c3.metric("Execution", "N/A" if values[2] is None else ("100%" if values[2] else "0%"))
        c4.metric("BLEU-4", "N/A" if values[3] is None else f"{values[3]:.3f}")
        st.caption("Assessment is calculated against the Expected SQL entered above. Execution accuracy compares the generated and expected result sets on the uploaded CSV.")
    elif current:
        st.info("No Expected SQL was supplied, so assessment metrics are not calculated for this query.")

    with st.expander("How the model read your input"):
        st.write("**Encoder input** (question + serialised schema)")
        st.code(" ".join(st.session_state["src_tokens"]), language="text")
        attn = st.session_state.get("attn")
        if attn:
            st.write("**Attention** — which input token each output token attended to.")
            import numpy as np
            src_toks = st.session_state["src_tokens"]
            out_toks = st.session_state["out_tokens"][: len(attn)]
            m = pd.DataFrame(np.array(attn),
                             index=[f"{i}. {t}" for i, t in enumerate(out_toks)],
                             columns=[f"{i}. {t}" for i, t in enumerate(src_toks)])
            st.dataframe(m.style.background_gradient(axis=1).format("{:.2f}"), width="stretch")

    # ----------------------------------------------------------------------
    # 4. Execute
    # ----------------------------------------------------------------------
    st.subheader("4 · Edit and run")
    if "editor" not in st.session_state:
        st.session_state["editor"] = st.session_state["sql"]

    edited = st.text_area(
        "SQL — edit freely, then run",
        key="editor", height=110,
        help="The query below is what actually runs. Assessment metrics remain based on the generated SQL, not later edits.")

    def _reset_editor():
        st.session_state["editor"] = st.session_state["sql"]

    c1, c2, c3 = st.columns([1, 1, 3])
    run_clicked = c1.button("▶ Run query", type="primary")
    c2.button("↺ Reset to generated", on_click=_reset_editor)

    dirty = norm_ws(edited) != norm_ws(st.session_state["sql"])
    c3.caption("✏️ edited — differs from generated query" if dirty else "unmodified model output")

    problems = validate_sql(edited, columns)
    for p in problems:
        st.warning(p)

    if df is None:
        st.info("Upload a CSV to execute the query against real data.")
    elif run_clicked or (auto_exec and not dirty):
        result_df, err = run_sql(df, edited, MODEL_TABLE, nocase)
        if err:
            st.error(f"SQLite error: {err}")
            st.caption("Edit the query above and press Run query.")
        else:
            st.success(f"{len(result_df):,} row(s) returned" + (" · from your edited query" if dirty else ""))
            st.dataframe(result_df, width="stretch")
            st.download_button("Download result as CSV",
                               result_df.to_csv(index=False).encode(),
                               file_name="result.csv", mime="text/csv")
            # Record the actual executed SQL/result rows against the latest assessment entry.
            history = st.session_state.setdefault("assessment_history", [])
            if history:
                history[-1]["Executed SQL"] = edited
                history[-1]["Executed Rows"] = int(len(result_df))
        
# -----------------------------------------------------------------------------
# 5. Assessment history + downloads
# -----------------------------------------------------------------------------
history = st.session_state.get("assessment_history", [])
if history:
    st.subheader("5 · Assessment History")
    hist_df = pd.DataFrame(history)

    assessed = hist_df[hist_df["Expected SQL"].astype(str).str.strip() != ""]
    if not assessed.empty:
        c1, c2, c3, c4 = st.columns(4)
        exact_avg = pd.to_numeric(assessed["Exact-Match Accuracy"], errors="coerce").mean()
        comp_avg = pd.to_numeric(assessed["Component-wise Accuracy"], errors="coerce").mean()
        exec_avg = pd.to_numeric(assessed["Execution Accuracy"], errors="coerce").mean()
        bleu_avg = pd.to_numeric(assessed["BLEU-4"], errors="coerce").mean()
        c1.metric("Exact Match", "N/A" if pd.isna(exact_avg) else f"{exact_avg:.1%}")
        c2.metric("Component-wise", "N/A" if pd.isna(comp_avg) else f"{comp_avg:.1%}")
        c3.metric("Execution", "N/A" if pd.isna(exec_avg) else f"{exec_avg:.1%}")
        c4.metric("BLEU-4", "N/A" if pd.isna(bleu_avg) else f"{bleu_avg:.3f}")

    st.dataframe(hist_df, width="stretch", hide_index=True)

    b1, b2, b3 = st.columns([1, 1, 2])
    b1.download_button(
        "Download history CSV",
        history_csv_bytes(history),
        file_name="text2sql_assessment_history.csv",
        mime="text/csv",
    )
    xlsx = history_xlsx_bytes(history)
    if xlsx is not None:
        b2.download_button(
            "Download history XLSX",
            xlsx,
            file_name="text2sql_assessment_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        b2.warning("XLSX export requires openpyxl in the Streamlit environment.")

    if b3.button("Clear assessment history"):
        st.session_state["assessment_history"] = []
        st.rerun()

st.divider()
st.caption("Encoder–Decoder (BiLSTM + Bahdanau attention + pointer-generator copy), trained on WikiSQL. Generated SQL is not guaranteed correct — always review it before trusting the result.")
