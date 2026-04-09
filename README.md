# Wald Decision Agent

Wald Decision Agent is a grounded document intelligence system for enterprise files. It ingests business documents, preserves structured tables, answers questions with source-backed evidence, performs deterministic calculations for numeric queries, and generates plots when the data supports them.

## Overview

The system is designed for mixed-document analysis across:

- `PDF`
- `DOCX`
- `XLSX` / `XLS`
- `CSV` / `TSV`
- `TXT` / `MD`
- visual attachments such as charts and scanned pages

It combines:

- document retrieval for narrative questions
- SQLite-backed structured reasoning for tables
- deterministic numeric computation for calculations
- planner-based routing to choose the right path per query
- Groq-based answer formatting by default
- ChromaDB for persistent, high-performance vector storage
- Gemini for vision extraction and embeddings

## Key Features

- Chat-style web interface for uploading document folders and asking questions
- Chat-scoped storage so each chat keeps its own files, SQLite database, plots, and reports
- Add, replace, or delete uploaded documents without deleting the chat
- Spreadsheet-aware ingestion that preserves table structure
- Grounded answers with references to the uploaded source files
- Safe handling of unsupported or missing metrics by abstaining instead of guessing
- Plot generation for supported trend and comparison questions
- Professional HNSW indexing for sub-second retrieval on large file sets

## Requirements

- Python `3.9+`
- `pip`
- API keys:
  - `GROQ_API_KEY` (Primary for answer formatting)
  - `GEMINI_API_KEY` (For vision extraction and embeddings)

## Setup

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Example `.env`:

```env
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```


## Run The Web App

### macOS / Linux
```bash
PYTHONPATH=src python -m wald_decision_agent.main serve --host 0.0.0.0 --port 8000
```

### Windows
```bash
set PYTHONPATH=src
python -m wald_decision_agent.main serve --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## How to Test

To see the system in action:

1. **Upload Data**: You can use the sample business documents provided in the `data/raw/` folder. Simply upload this folder during the "Upload Folder" step in the UI.
2. **Ask Questions**: Try asking questions from the `data/sample_questions/` folder to see how the agent handles structured, narrative, and visual data.
3. **Review Reports**: After an answer is generated, check the grounded evidence, source links, and any generated plots.

## Run From The CLI

```bash
PYTHONPATH=src python -m wald_decision_agent.main ask --docs data/raw --question "What is our current revenue trend?" --plot
```

## Repository Layout

```text
config/settings.yaml
data/raw/               # Sample document folder for testing
data/sample_questions/  # Sample questions to try
src/wald_decision_agent/
  core/
  ingestion/
  retrieval/
  reasoning/
  rendering/
  web/
```

## Notes

- Generated plots and reports are created per chat during execution.
- The system is designed to favor structured data (CSV/Excel) for numeric queries while using narrative text for explanations.
