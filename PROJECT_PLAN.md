# Wald Agent Reference Project Plan

## 1. Goal

Build a working Python-based `Wald Agent Reference` system that can:

- ingest a folder of company documents,
- answer leadership questions in natural language,
- ground every answer in retrieved evidence from the documents,
- compute numeric answers and intermediate calculations reliably,
- generate visualizations when the question benefits from charts or trend analysis,
- produce concise report-style outputs suitable for leadership review.

The assignment only explicitly requires the simpler insight agent, so the main submission should optimize for a clean, reliable `Task 1` implementation. The design should still leave a clear upgrade path toward a more autonomous decision agent.

## 2. Recommended Submission Strategy

The strongest submission is:

- a working GitHub repo,
- a small notebook for demo and exploration,
- a README with setup and test instructions,
- sample documents and sample queries,
- an evaluation script with a small validation set,
- optional plots for revenue/risk/theme summaries.

This is better than a notebook-only submission because it signals engineering maturity, reproducibility, and production thinking.

## 3. Scope and Assumptions

### In scope

- document ingestion from local files,
- parsing and chunking documents,
- embedding and indexing chunks,
- retrieval-augmented generation,
- grounded question answering,
- deterministic numeric calculation and validation,
- structured report-style natural language output,
- simple evaluation on a validation set,
- tool-driven charts derived from extracted numeric data.

### Out of scope

- full autonomous agent loops,
- external web browsing,
- multi-user deployment,
- enterprise auth,
- large-scale production infra.

### Explicit assumptions to state in the submission

- The solution focuses on `Task 1` from the assignment.
- Documents are provided locally in a folder.
- Input documents may include narrative files and structured spreadsheet files.
- The evaluator will provide their own API key through environment variables.
- Model names remain configurable in a config file or `.env`.
- Answers should be natural language, but backed by source citations/snippets.
- The evaluation set can be manually curated from sample prompts plus a few additional paraphrased leadership questions.

### Expected file types

The company documents can realistically arrive in several formats:

- `.pdf` for annual reports and quarterly reports
- `.docx` for strategy notes, operational updates, board memos, and internal writeups
- `.xlsx` / `.xls` for KPI trackers, departmental scorecards, budgets, forecasts, and raw operational metrics
- `.csv` / `.tsv` for exports from BI tools or finance systems
- `.txt` / `.md` for notes and internal documentation

The plan should treat `DOCX` and spreadsheet formats as required support, not optional support.

## 4. Solution Design

### Core architecture

1. `Ingestion layer`
   - Load documents from a local folder.
   - Support these formats as first-class inputs:
     - `.pdf`
     - `.docx`
     - `.xlsx` / `.xls`
     - `.csv` / `.tsv`
     - `.txt` / `.md`
   - Use format-specific parsers rather than forcing every file into a plain-text path.

2. `Preprocessing layer`
   - Clean text.
   - Split narrative documents into semantic chunks with overlap.
   - Preserve spreadsheet structure as structured tables instead of flattening them into unstructured text.
   - Preserve metadata:
     - file name
     - sheet name for spreadsheet sources
     - section heading
     - page number if available
     - document type
     - reporting period if detected
     - table name or inferred table region when available
     - row and column references for extracted spreadsheet cells

### Spreadsheet and table handling

Spreadsheet documents need a separate ingestion path because flattening them into text loses the structure needed for accurate querying and calculations.

Required implementation details:

- Parse each workbook sheet independently.
- Detect used ranges, header rows, and table-like regions.
- Preserve:
  - workbook name
  - sheet name
  - column headers
  - row labels
  - cell coordinates when useful
  - basic number/date formatting
- Store each detected table in a structured form such as a `pandas` dataframe plus metadata.
- Create a text representation for retrieval, but keep the canonical structured table for computation.

Recommended retrieval representation for spreadsheets:

- a compact textual summary for embedding search
- linked structured payload containing the exact dataframe/table
- metadata such as:
  - `source_file`
  - `sheet_name`
  - `table_id`
  - `header_map`
  - `time_period_columns`

This allows the agent to retrieve a relevant spreadsheet section semantically, then answer using the original structured data rather than hallucinated values.

3. `Indexing layer`
   - Create embeddings for chunks.
   - Store them in a vector index.
   - Good practical choice:
     - `FAISS` for local simplicity
     - fallback: `Chroma`

4. `Retrieval layer`
   - Retrieve top-k relevant chunks for a user query.
   - Use metadata-aware ranking when possible.
   - Add reranking if time permits.

5. `Answer generation layer`
   - Use an LLM to synthesize a concise answer.
   - Force grounding through prompt instructions:
     - answer only from retrieved context
     - cite sources
     - state uncertainty if evidence is insufficient

6. `Numeric reasoning and tool layer`
   - Do not rely on the LLM for arithmetic when the question involves numbers, trends, percentages, comparisons, or aggregations.
   - Route these queries through deterministic tools built in Python:
     - table extraction / metric extraction from retrieved context
     - spreadsheet table loader that resolves the exact workbook, sheet, and table region
     - calculation engine using `pandas` and standard Python math
     - validation checks for intermediate and final values
   - Return both:
     - the computed result
     - the calculation trace used to derive it
   - Example operations:
     - quarter-over-quarter growth
     - department ranking by performance
     - percentage change
     - risk frequency counts
     - trend summaries over time
     - aggregations from workbook tables
     - comparisons across sheets or reporting periods

### Spreadsheet query strategy

For spreadsheet-origin questions, the agent should follow a stricter path:

1. identify the relevant workbook and sheet from retrieval
2. resolve the matching table or cell range
3. normalize the values into a dataframe
4. run deterministic calculations
5. return the answer with:
   - source workbook
   - sheet name
   - columns used
   - calculation trace

This is the right way to answer prompts like:

- "Which departments are underperforming based on the scorecard?"
- "What is the quarter-over-quarter growth in revenue?"
- "Which business unit had the lowest margin in the latest sheet?"

7. `Visualization tool layer`
   - Add a custom chart-generation tool the agent can call when a query asks for trends, comparisons, distributions, or summaries that are easier to understand visually.
   - Input:
     - structured table or metric series extracted from the retrieved evidence
   - Output:
     - saved chart image
     - short textual interpretation
   - Recommended chart types:
     - line chart for revenue or KPI trends
     - bar chart for department comparisons
     - stacked bar for risk category comparisons
     - pie chart only when category counts are small and clear

8. `Reporting layer`
   - Return a structured natural-language output with sections such as:
     - Executive Summary
     - Key Findings
     - Calculations Performed
     - Evidence
     - Visual Insights
     - Risks / Caveats
     - Source References

9. `Evaluation layer`
   - Run a small benchmark over validation questions.
   - Score answer quality and grounding.
   - Separately score numeric correctness and visualization usefulness.

### Why this design scores well

- It is directly aligned with the assignment.
- It demonstrates modern GenAI engineering patterns.
- It is realistic to implement in limited time.
- It shows clear thinking about reliability, traceability, and future extensibility.

## 5. Recommended Tech Stack

### Language

- Python 3.11+

### Libraries

- `langchain` or a thin custom pipeline
- `faiss-cpu` or `chromadb`
- `pydantic` for typed configs and response schemas
- `python-dotenv` for environment loading
- `pypdf` for PDFs
- `python-docx` for DOCX parsing
- `openpyxl` for `.xlsx` workbook parsing and table preservation
- `xlrd` only if legacy `.xls` support is needed
- `pandas` for tabular outputs
- `numpy` for stable numerical operations
- `matplotlib` or `plotly` for charts
- `seaborn` optionally for cleaner plots
- `pytest` for tests

### Model choices

Keep these configurable. A strong default plan is:

- LLM: `gpt-4.1-mini` or `gpt-4o-mini`
- Embeddings: `text-embedding-3-large` or `text-embedding-3-small`

If you want to look more optimization-aware, position it like this:

- `gpt-4.1-mini` for cost-efficient synthesis,
- `text-embedding-3-large` for stronger retrieval quality,
- all model names exposed via config.

## 6. Output Format

Every answer should be natural language and follow a consistent template.

Example:

```text
Question
What is our current revenue trend?

Executive Summary
Revenue appears to be growing quarter-over-quarter, with the strongest gains coming from the enterprise segment.

Key Findings
1. Q2 revenue exceeded Q1 by X%.
2. Growth was concentrated in A and B units.
3. Margin pressure remains visible in C.

Calculations Performed
- Q2 vs Q1 growth = ((Q2 - Q1) / Q1) * 100
- Inputs were extracted from Q1_Business_Update.pdf (p.3) and Q2_Business_Update.pdf (p.4)

Visual Insights
- A line chart of quarterly revenue shows a consistent upward trend, with the sharpest increase between Q1 and Q2.

Evidence
- Annual_Report_2024.pdf, page 12: ...
- Q2_Business_Update.pdf, page 4: ...

Risks / Caveats
- Some departments reported incomplete monthly data.
- No direct regional split was available in the retrieved documents.

Source References
- Annual_Report_2024.pdf (p.12)
- Q2_Business_Update.pdf (p.4)
```

This makes the output feel leadership-ready rather than chatbot-like.

## 7. Evaluation Plan

The assignment explicitly said no fixed metric is required, so use a small but defensible evaluation framework.

### Validation set

Create `10-20` questions:

- 3 from the assignment examples,
- 5-10 paraphrased leadership questions,
- a few negative or ambiguous questions to test uncertainty handling.

Example categories:

- revenue trend
- underperforming departments
- major risks
- strategy priorities
- operational blockers
- missing-information questions

### Metrics

Use lightweight practical metrics:

- `Retrieval relevance`: did top-k contain the supporting evidence?
- `Answer grounding`: does the answer stay faithful to retrieved context?
- `Answer completeness`: does it answer the full question?
- `Numeric correctness`: are all reported values and intermediate computations correct?
- `Calculation trace quality`: are formulas and source inputs clearly shown?
- `Spreadsheet structure preservation`: were the right sheet, headers, and table boundaries used?
- `Citation quality`: are references clear and useful?
- `Visualization usefulness`: when a chart is generated, does it accurately represent the evidence and help interpretation?
- `Abstention quality`: does the system say "insufficient evidence" when needed?

### Evaluation method

- Manual scoring on a 1-5 rubric for each metric.
- Optional LLM-as-judge for a secondary automated pass.
- Include one summary table in the README or notebook.
- Add at least `3-5` calculation-heavy questions specifically to test arithmetic reliability.
- Add at least `3-5` spreadsheet-origin questions to test workbook retrieval and table-preserving computations.

## 8. Graphs and Visualizations

Graphs should be treated as an agent capability, not just a nice-to-have, especially for trend and comparison questions.

Recommended plots:

- revenue trend over time,
- department performance comparison,
- count of risks by category,
- document coverage by reporting period.

### When the agent should call the visualization tool

- The user explicitly asks for a graph, chart, trend, or comparison.
- The question asks about movement over time.
- The answer depends on comparing multiple entities or periods.
- A visual summary would materially improve readability for leadership.

### Visualization implementation detail

- First retrieve evidence.
- Then extract structured numeric fields into a dataframe.
- Validate the extracted values before plotting.
- If the source is a spreadsheet, preserve the original sheet semantics in labels and legends.
- Generate the chart and save it under `outputs/plots/`.
- Return both the chart path and a short interpretation in the final answer.

Important: charts should support the insight agent, not replace it. The narrative answer still needs grounding and citations.

## 9. Repo Structure

Use a clean structure like this:

```text
wald-agent-reference/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── sample_questions/
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── spreadsheet_parser.py
│   ├── embed_index.py
│   ├── retrieve.py
│   ├── calculator.py
│   ├── answer.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── tools.py
│   └── main.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_spreadsheet_parser.py
│   ├── test_calculator.py
│   ├── test_visualize.py
│   └── test_answer_format.py
└── outputs/
    ├── reports/
    └── plots/
```

## 10. Implementation Phases

### Phase 1: Baseline pipeline

- Set up repo and config.
- Add document loaders.
- Add dedicated spreadsheet parsing with header and table preservation.
- Build chunking and embedding flow.
- Create vector index.
- Implement query -> retrieve -> answer pipeline.
- Add structured numeric extraction for key financial and operational metrics.

### Phase 2: Tool-augmented grounded report generation

- Design a strong answer prompt.
- Enforce citation output.
- Add a deterministic calculator tool for numeric questions.
- Add logic to detect when tool use is required.
- Add structured response sections.
- Add fallback behavior for insufficient evidence.

### Phase 3: Evaluation and quality

- Create validation set.
- Add evaluation script.
- Test retrieval and output formatting.
- Test calculation-heavy prompts and intermediate arithmetic.
- Test spreadsheet queries against expected workbook/sheet/table selections.
- Test graph generation on trend/comparison prompts.
- Tune chunk size, overlap, and top-k.

### Phase 4: Polish

- Add notebook demo.
- Add plots for selected sample questions.
- Improve README.
- Add example inputs and outputs.

## 11. Practical Timeline

If you want this to feel realistic and interview-ready, use a 3-day plan:

### Day 1

- finalize assumptions,
- scaffold repo,
- implement ingestion and indexing,
- run first retrieval tests.

### Day 2

- implement answer generation,
- add structured reporting,
- create validation questions,
- test end-to-end on sample docs.

### Day 3

- add evaluation table,
- add plots,
- write README,
- clean notebook,
- prepare final repo submission.

## 12. README Requirements

The README should include:

- problem statement summary,
- approach overview,
- architecture diagram,
- explanation of the calculator and visualization tools,
- setup instructions,
- how to provide API key,
- how to ingest docs,
- how to run a sample query,
- how to run evaluation,
- assumptions and limitations,
- sample outputs.

## 13. What Will Make the Submission Stand Out

These are the highest-value differentiators:

- grounded answers with citations,
- deterministic calculations instead of trusting raw LLM arithmetic,
- visible calculation traces for numeric questions,
- proper workbook ingestion with preserved table structure,
- accurate answers to spreadsheet-based questions using structured data rather than flattened text,
- on-demand charts for trend and comparison questions,
- clear handling of uncertainty,
- clean modular Python repo,
- a small but credible evaluation framework,
- leadership-style report formatting,
- explicit assumptions and tradeoffs,
- optional visuals that support conclusions.

## 14. What to Avoid

- overbuilding an autonomous agent that the assignment did not require,
- letting the LLM perform business-critical arithmetic without verification,
- flattening Excel workbooks into plain text and losing headers, sheet context, or table boundaries,
- generating charts directly from unvalidated extracted numbers,
- returning uncited answers,
- using a notebook with no reusable code structure,
- skipping evaluation entirely,
- vague README instructions,
- making unsupported claims from the documents.

## 15. Recommended Final Positioning in the Submission

Frame the solution like this:

> This submission implements Wald Agent Reference, a grounded enterprise document intelligence agent for question answering over internal reports, tables, and visual artifacts. The system ingests internal reports, retrieves relevant evidence, and generates concise leadership-ready answers with citations. The design prioritizes factuality, modularity, and extensibility, with a clear path toward a future autonomous decision-support agent.

## 16. Suggested Next Step

The best next move is to build the repo around a strong `Task 1` RAG pipeline and treat autonomy as a clearly described future extension, not the centerpiece of the current submission.
