# Text-to-SQL Application

A natural-language interface for querying CSV-based tabular data. Users upload a CSV file, ask a question in plain English, and the system generates SQL that can be reviewed, edited, and executed against the uploaded data.

## Overview

The system uses a neural Text-to-SQL model based on:

- BiLSTM encoder
- Bahdanau attention
- Pointer-generator copy mechanism
- WikiSQL training data
- Schema serialization for column-aware SQL generation

The Streamlit application provides the user interface and executes generated SQL against an in-memory SQLite database.

## Application Flow

```text
CSV Upload
   │
   ├── Column/schema extraction
   ├── Data type detection
   └── Value availability for grounding
          │
          ▼
Natural-language question
          │
          ▼
Question + serialized schema
          │
          ▼
BiLSTM + Bahdanau Attention + Copy Decoder
          │
          ▼
Generated SQL
          │
          ▼
Inference-time semantic/value handling
          │
          ▼
SQL validation
          │
          ▼
In-memory SQLite execution
          │
          ▼
Query result
```

## Supported Question Types

The application is designed primarily for single-table analytical questions, including:

- Column selection
- Filtering with `WHERE`
- Numeric comparisons (`>`, `<`, `>=`, `<=`)
- Counting and `COUNT(*)`
- Aggregations such as `AVG`, `SUM`, `MIN`, and `MAX`
- Multiple filter conditions
- Value-based questions using values present in the uploaded data
- Queries involving ranking or ordering where supported by the trained/query-generation pipeline

The underlying WikiSQL model is focused on relatively simple single-table SQL. More complex constructs such as arbitrary joins, nested queries, and advanced SQL are outside the primary scope.

## Input

The system accepts:

### 1. CSV file

The CSV provides:

- Table data
- Column names
- Column data types
- Values that can be used for value grounding

Column names are normalized for compatibility with the model's training representation.

### 2. Natural-language question

Example:

```text
How many cars have fuel type Petrol?
```

The model receives the question together with a serialized schema similar to:

```text
<sos>
how many cars have fuel type petrol
<sep>
<tab> table
<col> s.no.
<col> name
<col> year
<col> selling price
<col> km driven
<col> fuel
<col> transmission
...
<eos>
```

## Output

The primary output is an executable SQL query, for example:

```sql
SELECT COUNT(*)
FROM "table"
WHERE "fuel type" = 'Petrol';
```

The generated SQL can be reviewed and edited before execution.

When a CSV is uploaded, the query is executed against an in-memory SQLite database and the result is displayed in the Streamlit interface.

## Model Architecture

### Encoder

A bidirectional LSTM encodes the question and serialized schema. Including the schema in the encoder input helps the model associate natural-language terms with available columns.

### Attention

Bahdanau attention allows the decoder to focus on relevant question and schema tokens while generating SQL.

### Pointer-Generator

The copy mechanism allows the decoder to reproduce tokens that are not part of the fixed target vocabulary, which is important for:

- Unseen column names
- Database values
- Schema-specific tokens

## Training Dataset

The primary training dataset is **WikiSQL**.

The current training configuration used for the reported model includes:

```text
Train examples:       56,355
Validation examples:   8,421
Test examples:        15,878
Embedding size:         256
Hidden size:            512
Encoder layers:           1
Dropout:                0.3
Batch size:              64
Learning rate:        0.001
Copy mechanism:          Yes
Maximum source length:   60 tokens
Maximum target length:   45 tokens
```

The project also contains a synthetic corpus generator with multiple schemas and SQL templates for experimentation and ablation.

## Model Evaluation

The project supports:

- Token-level validation accuracy
- Logical-form / SQL exact accuracy
- Execution accuracy

Execution accuracy is particularly useful because two syntactically different SQL queries can sometimes return the same result.

For the reported training run, the best validation loss was:

```text
0.2096
```

with validation token accuracy reaching approximately:

```text
94.31%
```

The application should be evaluated separately on raw model output and the complete inference pipeline so that post-processing improvements are not mistaken for improvements in the neural model itself.

## Inference-Time Handling

The application includes inference-side mechanisms intended to improve robustness without changing the trained checkpoint.

These include:

- Schema-aware column selection
- Value grounding against values available in the uploaded table
- Handling of `COUNT(*)` semantics for "how many" questions
- Suppression of clearly unsupported or hallucinated filter conditions
- SQL structural repair
- Schema-constrained column generation

These mechanisms operate during inference and therefore do not require retraining the existing checkpoint.

## Invalid and Unanswerable Questions

The system performs basic SQL validation before execution.

Examples of detected problems include:

- Empty SQL
- Unbalanced parentheses
- Unclosed quotes
- Unknown column references
- Non-`SELECT` statements

Questions that require information not available in the uploaded schema should not result in fabricated columns or data. The generated SQL should be reviewed when the question is ambiguous or outside the model's supported query patterns.

## Project Structure

```text
.
├── app.py
├── text2sql.py
├── runs/
│   └── seq2seq_copy/
│       └── best.pt
└── README.md
```

`text2sql.py` contains the data processing, model, training, decoding, evaluation, and SQL-related logic.

`app.py` contains the Streamlit interface and CSV/SQLite execution layer.

## Installation

Install the required Python packages:

```bash
pip install streamlit pandas torch numpy matplotlib
```

## Running the Application

Start Streamlit from the project directory:

```bash
streamlit run app.py
```

Then:

1. Load the trained checkpoint.
2. Upload a CSV file.
3. Review the detected schema.
4. Enter a natural-language question.
5. Generate SQL.
6. Review or edit the SQL.
7. Execute it against the uploaded data.

## Training the Model

Prepare WikiSQL:

```bash
git clone --depth 1 https://github.com/salesforce/WikiSQL.git
cd WikiSQL
tar xjf data.tar.bz2
cd ..
```

Prepare the dataset:

```bash
python text2sql.py prepare --wikisql WikiSQL/data
```

Train:

```bash
python text2sql.py train --wikisql WikiSQL/data --epochs 15
```

The trained checkpoint is written under:

```text
runs/seq2seq_copy/best.pt
```

## Decoding and Evaluation

Decode the WikiSQL test split:

```bash
python text2sql.py decode \
  --ckpt runs/seq2seq_copy/best.pt \
  --wikisql WikiSQL/data \
  --split test \
  --beam 5
```

Interactive mode:

```bash
python text2sql.py decode \
  --ckpt runs/seq2seq_copy/best.pt \
  --interactive
```

## Current Scope and Limitations

The system is intended for single-table CSV analysis and should not be treated as a general-purpose enterprise Text-to-SQL engine.

In particular:

- The neural model is trained primarily on WikiSQL-style SQL.
- Complex joins and nested SQL are outside the main training distribution.
- Grouping and other advanced SQL patterns should only be considered reliable when they are covered by the active training/inference pipeline.
- Generated SQL is not guaranteed to be correct and should be reviewed before being trusted for important decisions.

## Example Questions

```text
What is the average selling price of petrol cars?

How many cars have fuel type Diesel?

What is the maximum price?

What is the minimum mileage?

How many cars have a mileage greater than 20?

Show the cars with the highest selling price.

What is the average price for cars in Mumbai?
```

## Technology Stack

- Python
- PyTorch
- Streamlit
- Pandas
- SQLite
- NumPy
- Matplotlib
- WikiSQL

## License

This project is an academic Text-to-SQL application. Review the licenses of external datasets and dependencies before redistribution.
