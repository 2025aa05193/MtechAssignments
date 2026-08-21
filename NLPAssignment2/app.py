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

import glob
import io
import os
import re
import sqlite3
import tempfile

import pandas as pd
import streamlit as st
import torch

from text2sql import (SOS_ID, UNK_ID, EOS_ID, Vocab, beam_decode,
                      build_extended, build_source_sequence, detokenize_sql,
                      greedy_decode, load_checkpoint, outputids_to_tokens)

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


@torch.no_grad()
def generate(model, src_vocab, tgt_vocab, question, table, columns,
             beam, max_len, device):
    """Returns (sql_string, source_tokens, attention_matrix)."""
    src_tokens = build_source_sequence(question, table, list(columns))
    src_ids, src_ext, _, _, oovs = build_extended(src_tokens, [], src_vocab, tgt_vocab)

    if beam > 1:
        ids = beam_decode(model, src_ids, src_ext, len(oovs), beam_size=beam,
                          max_len=max_len, device=device)
        attn = None
    else:
        ids, attn = _greedy_with_attention(model, src_ids, src_ext, len(oovs),
                                           max_len, device)
    tokens = outputids_to_tokens(ids, tgt_vocab, oovs)
    return detokenize_sql(tokens), src_tokens, tokens, attn


@torch.no_grad()
def _greedy_with_attention(model, src_ids, src_ext_ids, n_oov, max_len, device):
    """Greedy decoding that also collects the attention weights, so the app can
    show WHICH input tokens the model looked at - the schema-linking evidence."""
    V = model.tgt_vocab_size
    src = torch.tensor([src_ids], device=device)
    src_ext = torch.tensor([src_ext_ids], device=device)
    enc_out, state, mask = model.encode(src, torch.tensor([len(src_ids)]))
    ctx = torch.zeros(1, model.enc_dim, device=device)
    y = torch.tensor([SOS_ID], device=device)

    out, attns = [], []
    for _ in range(max_len):
        p, state, ctx, a, _ = model.decoder.step(
            y, state, ctx, enc_out, mask, src_ext, n_oov)
        nxt = int(p.argmax(-1))
        if nxt == EOS_ID:
            break
        out.append(nxt)
        attns.append(a.squeeze(0).cpu().numpy())
        y = torch.tensor([nxt if nxt < V else UNK_ID], device=device)
    return out, attns


# ==========================================================================
# Execution layer
# ==========================================================================
def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match the preprocessing: column names are lowercased and whitespace
    collapsed, so the SQL the model generates lines up with the real table."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]
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


def quote_table_refs(sql: str, table_name: str = MODEL_TABLE) -> str:
    """
    `table` is a RESERVED WORD in SQL, so the bare `FROM table` that the
    WikiSQL-trained model emits is a syntax error in SQLite. Quoting the
    identifier makes it legal. Without this, every generated query fails to
    execute no matter how good the model is.
    """
    return re.sub(rf'(?<![\w."])(FROM|JOIN|INTO|UPDATE)\s+{re.escape(table_name)}'
                  r'(?![\w."])',
                  lambda m: f'{m.group(1)} "{table_name}"', sql, flags=re.I)


def run_sql(df: pd.DataFrame, sql: str, table_name: str = MODEL_TABLE):
    """Execute the query against the dataframe using an in-memory SQLite db."""
    con = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, con, index=False, if_exists="replace")
        # also register a couple of common aliases so a query that says
        # FROM data / FROM df still runs
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


# ==========================================================================
# UI
# ==========================================================================
st.title("🗄️ Text-to-SQL")
st.caption("Ask a question in plain English. The model reads your table's schema "
           "and writes the SQL.")

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
    columns = [c.strip().lower() for c in raw.split(",") if c.strip()]
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
        sql, src_tokens, out_tokens, attn = generate(
            model, src_vocab, tgt_vocab, question, MODEL_TABLE, columns,
            beam, max_len, device)
    st.session_state["sql"] = sql
    st.session_state["src_tokens"] = src_tokens
    st.session_state["out_tokens"] = out_tokens
    st.session_state["attn"] = attn

if "sql" in st.session_state:
    st.subheader("3 · Generated SQL")
    st.code(st.session_state["sql"], language="sql")

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
    st.subheader("4 · Run it")
    editable = st.text_area("Edit before running if needed",
                            value=st.session_state["sql"], height=90)

    if df is None:
        st.info("Upload a CSV to execute the query against real data.")
    else:
        run = auto_exec or st.button("Execute")
        if run:
            result, err = run_sql(df, editable, MODEL_TABLE)
            if err:
                st.error(f"SQLite error: {err}")
                st.caption("The model may have referenced a column that does "
                           "not exist, or produced invalid syntax. Edit the "
                           "query above and re-run.")
            else:
                st.success(f"{len(result):,} row(s) returned")
                st.dataframe(result, width='stretch')
                st.download_button("Download result as CSV",
                                   result.to_csv(index=False).encode(),
                                   file_name="result.csv", mime="text/csv")

st.divider()
st.caption("Encoder–Decoder (BiLSTM + Bahdanau attention + pointer-generator "
           "copy), trained on WikiSQL. Generated SQL is not guaranteed correct — "
           "always read it before trusting the result.")
