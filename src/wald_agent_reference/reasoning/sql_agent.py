from __future__ import annotations

from dataclasses import dataclass

from ..core.models import CalculationResult
from ..memory.structured_store import CatalogEntry, StructuredMemoryStore
from ..utils import tokenize


@dataclass
class SQLQueryAgent:
    store: StructuredMemoryStore

    def answer(self, question: str, catalog: list[CatalogEntry]) -> CalculationResult | None:
        lowered = question.lower()
        if any(term in lowered for term in ["plan", "target", "variance", "missed", "beat"]):
            result = self._variance_query(question, catalog)
            if result:
                return result
        if any(term in lowered for term in ["risk", "owner", "open high"]):
            result = self._risk_query(question, catalog)
            if result:
                return result
        return None

    def _variance_query(self, question: str, catalog: list[CatalogEntry]) -> CalculationResult | None:
        focus_keywords = self._focus_keywords(question)
        actual = self._find_table(catalog, ["actual", *focus_keywords])
        target = self._find_table(catalog, ["target", "plan", *focus_keywords])
        if not actual or not target:
            return None

        metric_keywords = self._metric_keywords(question)
        # Map natural-language intent to the best available join key and numeric columns.
        actual_dim, actual_metric = self._pick_dimension_and_metric(actual, metric_keywords + ["actual"])
        target_dim, target_metric = self._pick_dimension_and_metric(target, metric_keywords + ["target", "plan"])
        if not actual_dim or not actual_metric or not target_dim or not target_metric:
            return None

        actual_row_col = actual.columns.get("_source_row")
        target_row_col = target.columns.get("_source_row")
        direction = "ASC" if any(term in question.lower() for term in ["beat plan", "above plan", "highest positive"]) else "ASC"
        sql = f"""
            SELECT
                a.{actual_dim} AS entity,
                CAST(a.{actual_metric} AS REAL) AS actual_value,
                CAST(t.{target_metric} AS REAL) AS target_value,
                CAST(a.{actual_metric} AS REAL) - CAST(t.{target_metric} AS REAL) AS variance_value
                {f", a.{actual_row_col} AS actual_source_row" if actual_row_col else ""}
                {f", t.{target_row_col} AS target_source_row" if target_row_col else ""}
            FROM {actual.sqlite_table} a
            JOIN {target.sqlite_table} t
                ON LOWER(TRIM(a.{actual_dim})) = LOWER(TRIM(t.{target_dim}))
            ORDER BY variance_value {direction}
            LIMIT 3
        """
        columns, rows = self.store.execute(sql)
        if not rows:
            return None

        best = rows[0]
        answer = (
            f"{best[0]} has the largest negative variance at {best[3]:,.2f} "
            f"(actual {best[1]:,.2f} vs target {best[2]:,.2f})."
        )
        actual_row_index = 4 if actual_row_col else None
        target_row_index = 5 if target_row_col else (4 if target_row_col else None)
        findings = [
            (
                f"{row[0]}: actual {row[1]:,.2f}, target {row[2]:,.2f}, variance {row[3]:,.2f}"
                + (f" (source rows actual={int(row[actual_row_index])}, target={int(row[target_row_index])})" if actual_row_col and target_row_col else "")
            )
            for row in rows
        ]
        trace = [
            f"SQL query executed across {actual.source_file} and {target.source_file}.",
            "Join key matched on normalized entity name.",
            "Variance formula: actual_value - target_value.",
            f"Actual source: sheet={actual.metadata.get('sheet_name')}, dimension={self._original_column(actual, actual_dim)}, metric={self._original_column(actual, actual_metric)}, range={actual.metadata.get('source_range', 'n/a')}.",
            f"Target source: sheet={target.metadata.get('sheet_name')}, dimension={self._original_column(target, target_dim)}, metric={self._original_column(target, target_metric)}, range={target.metadata.get('source_range', 'n/a')}.",
            sql.strip(),
        ]
        evidence = [
            f"{actual.source_file} [{actual.metadata.get('sheet_name') or actual.logical_name} | row {int(best[actual_row_index]) if actual_row_col else '?'} | columns {self._original_column(actual, actual_dim)}, {self._original_column(actual, actual_metric)}]",
            f"{target.source_file} [{target.metadata.get('sheet_name') or target.logical_name} | row {int(best[target_row_index]) if target_row_col else '?'} | columns {self._original_column(target, target_dim)}, {self._original_column(target, target_metric)}]",
        ]
        chart_data = {
            "type": "bar",
            "labels": [row[0] for row in rows],
            "values": [float(row[3]) for row in rows],
            "title": "Variance by entity",
        }
        return CalculationResult(
            answer=answer,
            findings=findings,
            trace=trace,
            evidence_refs=evidence,
            chart_data=chart_data,
            numeric_value=float(best[3]),
        )

    def _risk_query(self, question: str, catalog: list[CatalogEntry]) -> CalculationResult | None:
        risk_table = self._find_table(catalog, ["risk"])
        if not risk_table:
            return None
        severity_col = self._find_column(risk_table, ["severity"])
        status_col = self._find_column(risk_table, ["status"])
        owner_col = self._find_column(risk_table, ["owner"])
        if not severity_col or not status_col or not owner_col:
            return None
        sql = f"""
            SELECT {owner_col} AS owner, COUNT(*) AS risk_count
            FROM {risk_table.sqlite_table}
            WHERE LOWER(TRIM({severity_col})) = 'high'
              AND LOWER(TRIM({status_col})) = 'open'
            GROUP BY {owner_col}
            ORDER BY risk_count DESC, owner ASC
        """
        _, rows = self.store.execute(sql)
        if not rows:
            return None
        answer = f"{rows[0][0]} owns the highest number of open high-severity risks with {rows[0][1]} items."
        findings = [f"{owner} has {count} open high-severity risks." for owner, count in rows]
        trace = [
            f"SQL query executed against {risk_table.source_file}.",
            "Filtered on severity='High' and status='Open'.",
            sql.strip(),
        ]
        return CalculationResult(
            answer=answer,
            findings=findings,
            trace=trace,
            evidence_refs=[f"{risk_table.source_file} [{risk_table.metadata.get('sheet_name') or risk_table.logical_name}]"],
            chart_data={"type": "bar", "labels": [row[0] for row in rows], "values": [float(row[1]) for row in rows], "title": "Open high risks by owner"},
            numeric_value=float(rows[0][1]),
        )

    def _find_table(self, catalog: list[CatalogEntry], keywords: list[str]) -> CatalogEntry | None:
        best: CatalogEntry | None = None
        best_score = -1
        for entry in catalog:
            # Score tables with lightweight lexical matching so the SQL agent stays deterministic.
            haystack = " ".join([entry.table_id, entry.logical_name, entry.source_file, *entry.columns.keys()]).lower()
            score = sum(1 for keyword in keywords if keyword in haystack)
            if score > best_score:
                best = entry
                best_score = score
        return best if best_score > 0 else None

    def _pick_dimension_and_metric(self, entry: CatalogEntry, metric_keywords: list[str]) -> tuple[str | None, str | None]:
        dimension = self._find_column(entry, ["department", "region", "business", "owner", "team"])
        metric = self._find_column(entry, metric_keywords, prefer_numeric=True)
        return dimension, metric

    def _find_column(self, entry: CatalogEntry, keywords: list[str], prefer_numeric: bool = False) -> str | None:
        best_col: str | None = None
        best_score = -1
        for original, sqlite_name in entry.columns.items():
            if original == "_source_row":
                continue
            tokens = tokenize(original)
            score = sum(1 for keyword in keywords if keyword in original.lower() or keyword in tokens)
            if prefer_numeric and any(word in original.lower() for word in ["score", "revenue", "margin", "cost", "target", "plan", "actual"]):
                score += 1
            if score > best_score:
                best_col = sqlite_name
                best_score = score
        return best_col if best_score > 0 else None

    def _focus_keywords(self, question: str) -> list[str]:
        lowered = question.lower()
        focus: list[str] = []
        for candidate in ["region", "department", "business", "owner", "team"]:
            if candidate in lowered:
                focus.append(candidate)
        focus.extend(self._metric_keywords(question))
        return focus

    def _metric_keywords(self, question: str) -> list[str]:
        lowered = question.lower()
        metrics = [candidate for candidate in ["revenue", "score", "margin", "cost"] if candidate in lowered]
        return metrics or ["score", "revenue", "margin", "cost"]

    @staticmethod
    def _original_column(entry: CatalogEntry, sqlite_name: str) -> str:
        for original, candidate in entry.columns.items():
            if candidate == sqlite_name:
                return original
        return sqlite_name
