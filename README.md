# Wald Agent Reference

Wald Agent Reference is a grounded document intelligence system for enterprise files. It ingests business documents, preserves structured tables, answers questions with source-backed evidence, performs deterministic calculations for numeric queries, and generates plots when the data supports them.

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
- optional Gemini and Supermemory integrations

## Key Features

- Chat-style web interface for uploading document folders and asking questions
- Chat-scoped storage so each chat keeps its own files, SQLite database, plots, and reports
- Add, replace, or delete uploaded documents without deleting the chat
- Spreadsheet-aware ingestion that preserves table structure
- Grounded answers with references to the uploaded source files
- Safe handling of unsupported or missing metrics by abstaining instead of guessing
- Plot generation for supported trend and comparison questions

## Requirements

- Python `3.11+` recommended
- `pip`
- optional API keys:
  - `GEMINI_API_KEY`
  - `SUPERMEMORY_API_KEY`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Example `.env`:

```env
GEMINI_API_KEY=
SUPERMEMORY_API_KEY=
```

The project runs locally without keys. Add `GEMINI_API_KEY` for Gemini formatting and vision-based extraction. Add `SUPERMEMORY_API_KEY` if you want to enable Supermemory-backed retrieval.

## Run The Web App

```bash
PYTHONPATH=src python -m wald_agent_reference.main serve --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Typical flow:

1. Create a chat
2. Upload a folder of documents
3. Ask questions in the chat
4. Review the grounded answer, evidence links, and generated plots

## Run From The CLI

```bash
PYTHONPATH=src python -m wald_agent_reference.main ask --docs data/raw --question "What is our current revenue trend?" --plot
```

## Repository Layout

```text
config/settings.yaml
data/raw/
notebooks/demo.ipynb
src/wald_agent_reference/
  core/
  ingestion/
  retrieval/
  reasoning/
  memory/
  rendering/
  web/
```

## Notes

- `data/raw/` contains sample files for local runs
- generated plots and reports are created per chat during execution
- the notebook is optional and can be used as an alternate demo surface
