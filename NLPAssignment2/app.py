"""
STEP 7 - Streamlit web application.

    pip install streamlit pandas torch numpy
    streamlit run app.py

Flow
    1. Load a trained checkpoint (sidebar).
    2. Provide a schema, either by uploading a .csv (headers are read
       automatically and become the column list) or by typing one in.
    3. Ask a question in plain English.
    4. The model generates SQL, which is shown and can be edited.
    5. If a CSV was uploaded, the query is executed against it with
       SQLite and the result table is displayed.

Everything model-side is imported from text2sql.py - the app adds no
modelling logic of its own, only the interface and the execution layer.
"""

from __future__ import annotations

import difflib
import glob
import io
import os
import re
import sqlite3
import tempfile

import pandas as pd
import streamlit as st
import torch

from text2sql import (Vocab, load_checkpoint, quote_table_refs,
                      predict_sql_with_values)

st.set_page_config(page_title="Text-to-SQL", page_icon="🗄️", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# The WikiSQL-trained model always emits this table name (see the placeholder
# discussion in the adapter); we register the uploaded CSV under it.
MODEL_TABLE = "table"


def resolve_ckpt(path: str) -> str | None:
    """
    Find a checkpoint whether the path is absolute, relative to the current
    working directory, or relative to app.py.

    `streamlit run` resolves relative paths against the directory you LAUNCHED
    from, not the directory app.py lives in. That is why a checkpoint that
    plainly exists shows up as "not found" - the app was started from somewhere
    else. Checking APP_DIR too removes the trap.
    """
    if not path:
        return None
    for cand in (path, os.path.join(os.getcwd(), path),
                 os.path.join(APP_DIR, path)):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


@st.cache_data(show_spinner=False)
def discover_ckpts() -> list[str]:
    """Every *.pt under runs/ in either the launch dir or the app dir."""
    found = []
    for root in {os.getcwd(), APP_DIR}:
        for pat in ("runs/*/*.pt", "*.pt", "runs/*.pt"):
            found += glob.glob(os.path.join(root, pat))
    return sorted({os.path.abspath(f) for f in found})


# ==========================================================================
# Model loading
# ==========================================================================
@st.cache_resource(show_spinner="Loading model…")
def get_model(ckpt_path: str, device: str):
    return load_checkpoint(ckpt_path, device)


# Model inference is centralized in text2sql.py.


# Execution layer
# ==========================================================================
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
    """Create the in-memory SQLite table with optional case-insensitive TEXT."""
    cols = []
    for name, dtype in zip(df.columns, df.dtypes):
        t = _sqlite_type(dtype)
        collate = " COLLATE NOCASE" if (nocase and t == "TEXT") else ""
        cols.append(f'"{name}" {t}{collate}')
    con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    con.execute(f'CREATE TABLE "{table_name}" ({", ".join(cols)})')


def run_sql(df: pd.DataFrame, sql: str, table_name: str = MODEL_TABLE,
            nocase: bool = True):
    """Execute generated/edited SQL against the uploaded DataFrame."""
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


def validate_sql(sql: str, columns) -> list:
    """
    Cheap static checks so obvious mistakes surface before hitting SQLite.
    Advisory only - never blocks execution, because a wrong warning is worse
    than no warning.
    """
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
    elif re.search(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER)\b", s, re.I):
        msgs.append("Statement modifies data - it runs against the in-memory "
                    "copy only, never your CSV file.")
    known = {str(c).lower() for c in (columns or [])}
    if known:
        used = {m.lower() for m in re.findall(r'"([^"]+)"', s)}
        unknown = sorted(u for u in used if u not in known and u != MODEL_TABLE)
        if unknown:
            msgs.append("Not a column in this table: "
                        + ", ".join('"%s"' % u for u in unknown))
    return msgs


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make CSV headers look like the schema the model was trained on.

    WikiSQL headers are natural language with SPACES ("years in toronto",
    "school/club team"). CSV headers are usually snake_case
    ("Kilometers_Driven"). Fed in raw, `kilometers_driven` is a single unseen
    token, so the model cannot link it to the words "kilometers" and "driven"
    in the question - it emits a fragment and invents a WHERE clause.
    Underscores therefore become spaces, matching training.
    """
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).replace("_", " ")).strip().lower()
                  for c in df.columns]
    return df


_NUM_UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[a-zA-Z/%]+\s*$")


def strip_units(df: pd.DataFrame, min_ratio: float = 0.8) -> tuple:
    """
    Convert text columns like '26.6 kmpl', '998 CC', '58.16 bhp' into numbers.

    Real CSVs store measurements as strings with the unit attached. SQLite
    coerces a leading number for AVG/SUM, so those happen to work - but
    ORDER BY, MIN/MAX and comparisons fall back to LEXICOGRAPHIC ordering and
    return confidently WRONG answers with no error at all. Measured on
    used_cars_data.csv:

        MAX("engine")                 raw '999 CC'   vs   converted 5998
        ORDER BY "mileage" DESC       raw '9.9 kmpl' vs   converted 33.54
        COUNT(*) WHERE "mileage" > 25 raw 568        vs   converted 494

    A silent wrong number is worse than a crash, so columns are converted when
    at least `min_ratio` of their non-null values match number-plus-unit.
    """
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
                errors="coerce")
            converted.append(col)
    return df, converted


# UI
# ==========================================================================
st.title("🗄️ Text-to-SQL")
st.caption("Pipeline: decoder-level SQL constraints v5")
st.caption("Ask a question in plain English. The model reads your table's schema and writes the SQL.")
st.caption("Pipeline version: mandatory semantic inference v4")

with st.sidebar:
    st.header("Model")
    found = discover_ckpts()
    if found:
        labels = [os.path.relpath(f, os.getcwd()) if f.startswith(os.getcwd())
                  else f for f in found]
        pick = st.selectbox("Detected checkpoints", labels + ["Other path…"])
        ckpt_raw = (st.text_input("Checkpoint path",
                                  value="runs/seq2seq_copy/best.pt")
                    if pick == "Other path…" else found[labels.index(pick)])
    else:
        ckpt_raw = st.text_input("Checkpoint path",
                                 value="runs/seq2seq_copy/best.pt")
    device = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available()
                          else ["cpu"])
    st.divider()
    st.header("Decoding")
    strategy = st.radio("Strategy", ["Greedy", "Beam search"], horizontal=True)
    beam = st.slider("Beam size", 2, 10, 5) if strategy == "Beam search" else 1
    max_len = st.slider("Max SQL length (tokens)", 10, 80, 45)
    st.divider()
    auto_exec = st.checkbox("Execute query automatically", value=True)
    nocase = st.checkbox("Case-insensitive text matching", value=True,
                         help="Declares text columns COLLATE NOCASE so "
                              "WHERE \"city\" = 'mumbai' matches 'Mumbai'. "
                              "The model lowercases everything, so without "
                              "this most WHERE clauses return zero rows.")
    constrain = st.checkbox("Restrict columns to the schema", value=True,
                            help="Forces every generated column name to be a "
                                 "real column. Without it the model emits "
                                 "fragments like \"type\" instead of "
                                 "\"fuel type\".")
    clean_units = st.checkbox("Strip units from numeric-looking text columns",
                              value=True,
                              help="Turns '26.6 kmpl' into 26.6 so AVG/MIN/MAX "
                                   "work on those columns.")

ckpt = resolve_ckpt(ckpt_raw)
model = None
if ckpt:
    try:
        model, src_vocab, tgt_vocab, cfg = get_model(ckpt, device)
        with st.sidebar:
            st.success("Model loaded")
            st.caption(f"{cfg['emb_dim']}d emb · {cfg['hid_dim']}d hidden · "
                       f"copy {'on' if cfg['use_copy'] else 'off'} · "
                       f"trained on {cfg['dataset']}")
    except Exception as e:
        st.sidebar.error(f"Could not load checkpoint: {e}")
else:
    with st.sidebar:
        st.warning("No checkpoint found at that path.")
        st.caption("Relative paths are resolved from the directory you ran "
                   "`streamlit run` in — not from app.py.")
        st.code(f"looked for : {ckpt_raw}\n"
                f"working dir: {os.getcwd()}\n"
                f"app dir    : {APP_DIR}", language="text")
        st.caption("Fix: paste the **absolute** path above, or `cd` into the "
                   "project folder before launching. Train one with "
                   "`python text2sql.py train --wikisql WikiSQL/data`.")

# --------------------------------------------------------------------------
# 1. Schema
# --------------------------------------------------------------------------
st.subheader("1 · Your table")
mode = st.radio("Schema source", ["Upload a CSV", "Enter columns manually"],
                horizontal=True, label_visibility="collapsed")

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
            st.success(f"Read **{len(df):,} rows × {len(columns)} columns** "
                       f"from `{up.name}`")
            if converted:
                st.info("Converted to numeric by stripping units: "
                        + ", ".join(f"`{c}`" for c in converted)
                        + " — without this, ORDER BY and MIN/MAX on these "
                          "columns sort as text and give wrong answers.")
            c1, c2 = st.columns([2, 1])
            with c1:
                st.dataframe(df.head(8), width='stretch')
            with c2:
                st.write("**Detected schema**")
                st.dataframe(pd.DataFrame({
                    "column": columns,
                    "type": [str(t) for t in df.dtypes]}),
                    width='stretch', hide_index=True)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
else:
    table_name = st.text_input("Table name", value="employees")
    raw = st.text_input("Columns (comma-separated)",
                        value="employee_id, name, department, salary, hire_date")
    columns = [re.sub(r"\s+", " ", c.replace("_", " ")).strip().lower()
               for c in raw.split(",") if c.strip()]
    if columns:
        st.caption(f"{len(columns)} columns: " + ", ".join(f"`{c}`" for c in columns))

# --------------------------------------------------------------------------
# 2. Question
# --------------------------------------------------------------------------
st.subheader("2 · Your question")
examples = [
    "What is the average salary in the sales department?",
    "How many employees are there in each department?",
    "Show the name of the employee with the highest salary",
    "Count the records where the status is shipped",
]
ex = st.selectbox("Example questions", ["—"] + examples)
question = st.text_input(
    "Question",
    value="" if ex == "—" else ex,
    placeholder="e.g. what is the average salary in the sales department?",
    label_visibility="collapsed")

go = st.button("Generate SQL", type="primary", disabled=not (model and columns
                                                             and question.strip()))

if not columns:
    st.info("Upload a CSV or enter columns to continue.")

# --------------------------------------------------------------------------
# 3. Generate
# --------------------------------------------------------------------------
if go:
    with st.spinner("Generating…"):
        result = predict_sql_with_values(
            model, question, MODEL_TABLE, columns, src_vocab, tgt_vocab,
            beam=beam, max_len=max_len, device=device, df_or_rows=df,
            constrain_columns=constrain)

    raw_sql = result["raw_sql"]
    sql = result["sql"]
    src_tokens = result["src_tokens"]
    out_tokens = result["out_tokens"]
    attn = result["attn"]

    st.session_state["raw_sql"] = raw_sql
    st.session_state["post"] = ["decoder-level question/SQL constraints enforced"]
    st.session_state["sql"] = sql
    st.session_state["editor"] = sql
    st.session_state["src_tokens"] = src_tokens
    st.session_state["out_tokens"] = out_tokens
    st.session_state["attn"] = attn

if "sql" in st.session_state:
    st.subheader("3 · Generated SQL")
    st.code(st.session_state["sql"], language="sql")
    for note in st.session_state.get("post", []):
        st.caption("🔧 post-processing: " + note)

    with st.expander("How the model read your input"):
        st.write("**Encoder input** (question + serialised schema)")
        st.code(" ".join(st.session_state["src_tokens"]), language="text")
        attn = st.session_state.get("attn")
        if attn:
            st.write("**Attention** — which input token each output token "
                     "attended to. Column names lighting up next to question "
                     "words is schema linking working.")
            import numpy as np
            src_toks = st.session_state["src_tokens"]
            out_toks = st.session_state["out_tokens"][: len(attn)]
            # Labels MUST be unique: the input repeats `<col>` once per column
            # and the output can repeat a token, and pandas' Styler refuses to
            # apply a gradient to a frame with duplicate index or columns.
            m = pd.DataFrame(
                np.array(attn),
                index=[f"{i}. {t}" for i, t in enumerate(out_toks)],
                columns=[f"{i}. {t}" for i, t in enumerate(src_toks)])
            st.dataframe(m.style.background_gradient(axis=1, cmap="Blues")
                         .format("{:.2f}"), width='stretch')
        else:
            st.caption("Attention weights are only collected for greedy decoding.")

    # ----------------------------------------------------------------------
    # 4. Execute
    # ----------------------------------------------------------------------
    st.subheader("4 · Edit and run")

    # The editor MUST have a key. Without one, Streamlit re-seeds the widget
    # from `value` on every rerun, so clicking Execute silently discarded the
    # user's edits and re-ran the model's original SQL.
    if "editor" not in st.session_state:
        st.session_state["editor"] = st.session_state["sql"]

    edited = st.text_area(
        "SQL — edit freely, then run",
        key="editor", height=110,
        help="Fix a wrong column, add a GROUP BY, change the WHERE value — "
             "anything. The query below is what actually runs.")

    def _reset_editor():
        # MUST be an on_click callback: Streamlit refuses to mutate a widget's
        # session_state key after the widget has been instantiated this run.
        # Callbacks execute before the script reruns, so this is legal.
        st.session_state["editor"] = st.session_state["sql"]

    c1, c2, c3 = st.columns([1, 1, 3])
    run_clicked = c1.button("▶ Run query", type="primary")
    c2.button("↺ Reset to generated", on_click=_reset_editor)

    dirty = norm_ws(edited) != norm_ws(st.session_state["sql"])
    if dirty:
        c3.caption("✏️ edited — differs from the generated query")
    else:
        c3.caption("unmodified model output")

    # cheap static checks before touching the database
    problems = validate_sql(edited, columns)
    for p in problems:
        st.warning(p)

    if df is None:
        st.info("Upload a CSV to execute the query against real data.")
    else:
        if run_clicked or (auto_exec and not dirty):
            result, err = run_sql(df, edited, MODEL_TABLE, nocase)
            if err:
                st.error(f"SQLite error: {err}")
                st.caption("Edit the query above and press Run query.")
            else:
                st.success(f"{len(result):,} row(s) returned"
                           + (" · from your edited query" if dirty else ""))
                st.dataframe(result, width='stretch')
                st.download_button("Download result as CSV",
                                   result.to_csv(index=False).encode(),
                                   file_name="result.csv", mime="text/csv")
                hist = st.session_state.setdefault("history", [])
                if not hist or hist[-1][0] != edited:
                    hist.append((edited, len(result)))
        elif dirty:
            st.info("Query edited — press **Run query** to execute it.")

    if st.session_state.get("history"):
        with st.expander(f"Query history ({len(st.session_state['history'])})"):
            for i, (q, n) in enumerate(reversed(st.session_state["history"][-10:]), 1):
                st.code(q, language="sql")
                st.caption(f"{n:,} row(s)")

st.divider()
st.caption("Encoder–Decoder (BiLSTM + Bahdanau attention + pointer-generator "
           "copy), trained on WikiSQL. Generated SQL is not guaranteed correct — "
           "always read it before trusting the result.")
