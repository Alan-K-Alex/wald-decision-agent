# Wald Agent Reference

Wald Agent Reference is a grounded document intelligence agent for enterprise documents. It ingests `PDF`, `DOCX`, `XLSX/XLS`, `CSV/TSV`, `TXT`, `MD`, and visual attachments, preserves structured tables, performs deterministic calculations, and generates concise answers with source references.

## What the system does

- Ingests narrative and structured business documents from a local folder
- Preserves spreadsheet and document tables as structured data
- Stores extracted structured data in SQLite for reliable querying
- Uses Supermemory as an optional retrieval layer for narrative and visual questions
- Uses a planner to choose the right route for each query:
  - retrieval
  - calculator
  - SQL sub-agent
  - visual reasoning
- Produces concise answers grounded only in the provided data
- Includes source references so outputs can be cross-checked
- Generates plots for trend/comparison questions when appropriate

## Key implementation choices

- `DOCX`, spreadsheet, and `PDF` tables are handled with separate extraction paths
- Numeric answers are computed with deterministic Python logic instead of raw LLM arithmetic
- Cross-table questions are answered through SQL over SQLite
- Long documents are chunked for retrieval, while canonical extracted content is also preserved
- Visual artifacts such as charts can be parsed and used in answers
- SQLite remains the source of truth for structured tables and SQL-based reasoning
- Supermemory is used as an optional retrieval layer, with local hybrid retrieval as fallback

## Requirements

- Python `3.11+` recommended
- `pip`
- Optional API keys:
  - `GEMINI_API_KEY` for Gemini formatting / embeddings / visual extraction
  - `SUPERMEMORY_API_KEY` for Supermemory sync
- `tesseract` is optional if OCR is needed for image-heavy attachments

## Environment setup

Create a local environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
cp .env.example .env
```

The example file is:

```env
GEMINI_API_KEY=
SUPERMEMORY_API_KEY=
```

The project runs without API keys using local fallback paths. Add `GEMINI_API_KEY` for live model formatting and `SUPERMEMORY_API_KEY` for managed retrieval.

## How to run

Run a sample question:

```bash
PYTHONPATH=src python -m wald_agent_reference.main ask --docs data/raw --question "What is our current revenue trend?" --plot
```

More examples:

```bash
PYTHONPATH=src python -m wald_agent_reference.main ask --docs data/raw --question "Which departments are underperforming?"
PYTHONPATH=src python -m wald_agent_reference.main ask --docs data/raw --question "Which region missed revenue plan by the largest amount?" --plot
PYTHONPATH=src python -m wald_agent_reference.main ask --docs data/raw --question "What does the quarterly revenue chart show?"
```

Run the validation set:

```bash
PYTHONPATH=src python -m wald_agent_reference.main evaluate --docs data/raw --validation data/sample_questions/validation.json
```

## Project structure

```text
config/settings.yaml              Runtime configuration
data/raw/                        Sample input documents
data/sample_questions/           Validation set
notebooks/demo.ipynb             Demo notebook
src/wald_agent_reference/          Core implementation
src/wald_agent_reference/core/            Settings, models, agent orchestration, logging
src/wald_agent_reference/ingestion/       PDF/DOCX/XLSX/TXT/visual extraction
src/wald_agent_reference/retrieval/       Hybrid lexical + vector retrieval
src/wald_agent_reference/reasoning/       Planner, calculator, SQL agent, answer composer
src/wald_agent_reference/memory/          SQLite store and optional Supermemory sync
src/wald_agent_reference/rendering/       Plot generation
tests/                           Automated tests
```

## Optional validation

```bash
.venv/bin/python -m pytest
```
