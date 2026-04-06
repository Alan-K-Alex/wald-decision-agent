from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from ..core.models import CalculationResult, StructuredTable
from ..utils import compact_whitespace, tokenize


@dataclass
class CalculationEngine:
    def calculate(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        if not tables:
            return None

        lowered = question.lower()
        if any(term in lowered for term in ["trend", "growth", "quarter-over-quarter", "qoq"]):
            result = self._calculate_trend(question, tables)
            if result:
                return result

        if any(term in lowered for term in ["underperform", "lowest", "bottom", "highest", "top"]):
            result = self._calculate_ranking(question, tables)
            if result:
                return result

        if any(term in lowered for term in ["count", "how many", "total", "sum", "average"]):
            result = self._calculate_aggregate(question, tables)
            if result:
                return result

        return self._calculate_ranking(question, tables) or self._calculate_trend(question, tables)

    def _calculate_trend(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        target_metric = self._detect_target_metric(question)
        for table in tables:
            extracted = self._extract_series(table.dataframe, target_metric)
            if not extracted:
                continue
            series, source_row = extracted
            labels = list(series.keys())
            values = list(series.values())
            if len(values) < 2:
                continue
            delta = values[-1] - values[-2]
            growth = (delta / values[-2] * 100.0) if values[-2] not in {0, 0.0} else math.inf
            direction = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "flat"
            answer = f"{target_metric.title()} shows an overall {direction} trend, moving from {values[0]:,.2f} to {values[-1]:,.2f}."
            findings = [
                f"The latest period ({labels[-1]}) is {values[-1]:,.2f}.",
                f"The previous period ({labels[-2]}) is {values[-2]:,.2f}.",
                f"Period-over-period change is {delta:,.2f} ({growth:,.2f}%).",
            ]
            trace = [
                f"Source workbook/table: {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}",
                f"Source row: {source_row}" if source_row is not None else "Source row: not available",
                f"Source range: {table.metadata.get('source_range', 'not available')}",
                f"Series used: {', '.join(f'{label}={value:,.2f}' for label, value in series.items())}",
                f"Growth formula: (({values[-1]:,.2f} - {values[-2]:,.2f}) / {values[-2]:,.2f}) * 100 = {growth:,.2f}%",
            ]
            return CalculationResult(
                answer=answer,
                findings=findings,
                trace=trace,
                evidence_refs=[
                    f"{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | row {source_row if source_row is not None else '?'} | range {table.metadata.get('source_range', 'n/a')}]"
                ],
                chart_data={"type": "line", "labels": labels, "values": values, "title": f"{target_metric.title()} trend"},
                numeric_value=growth,
            )
        return None

    def _calculate_ranking(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        lowered = question.lower()
        worst = any(term in lowered for term in ["underperform", "lowest", "bottom"])
        target_metric = self._detect_target_metric(question)
        for table in tables:
            ranking = self._extract_ranking(table.dataframe, target_metric)
            if ranking is None:
                continue
            dimension_col, metric_col, ranked = ranking
            if ranked.empty:
                continue
            ranked = ranked.sort_values(metric_col, ascending=worst).reset_index(drop=True)
            selected = ranked.head(3)
            best_row = selected.iloc[0]
            comparator = "lowest" if worst else "highest"
            answer = (
                f"{best_row[dimension_col]} has the {comparator} {metric_col} value at {best_row[metric_col]:,.2f}."
            )
            findings = [
                f"Ranking uses `{metric_col}` from sheet `{table.metadata.get('sheet_name', 'Sheet1')}`.",
            ] + [
                f"{row[dimension_col]} = {row[metric_col]:,.2f} (source row {int(row['_source_row']) if '_source_row' in row and pd.notna(row['_source_row']) else '?'})"
                for _, row in selected.iterrows()
            ]
            trace = [
                f"Dimension column: {dimension_col}",
                f"Metric column: {metric_col}",
                f"Source workbook/table: {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}",
                f"Source range: {table.metadata.get('source_range', 'not available')}",
            ]
            return CalculationResult(
                answer=answer,
                findings=findings,
                trace=trace,
                evidence_refs=[
                    f"{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | row {int(best_row['_source_row']) if '_source_row' in best_row and pd.notna(best_row['_source_row']) else '?'} | metric {metric_col}]"
                ],
                chart_data={
                    "type": "bar",
                    "labels": selected[dimension_col].astype(str).tolist(),
                    "values": [float(value) for value in selected[metric_col].tolist()],
                    "title": f"{metric_col} by {dimension_col}",
                },
                numeric_value=float(best_row[metric_col]),
            )
        return None

    def _calculate_aggregate(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        target_metric = self._detect_target_metric(question)
        lowered = question.lower()
        for table in tables:
            frame = table.dataframe.copy()
            numeric_columns = self._numeric_columns(frame)
            if not numeric_columns:
                continue
            metric_col = self._best_numeric_column(frame, target_metric)
            if not metric_col:
                continue
            series = pd.to_numeric(frame[metric_col], errors="coerce").dropna()
            if series.empty:
                continue
            if "average" in lowered or "mean" in lowered:
                value = float(series.mean())
                op_label = "Average"
            elif "count" in lowered or "how many" in lowered:
                value = float(series.count())
                op_label = "Count"
            else:
                value = float(series.sum())
                op_label = "Total"
            answer = f"{op_label} {metric_col} is {value:,.2f}."
            return CalculationResult(
                answer=answer,
                findings=[f"{op_label} computed from column `{metric_col}`."],
                trace=[
                    f"Operation: {op_label.lower()}",
                    f"Column used: {metric_col}",
                    f"Source workbook/table: {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}",
                    f"Source range: {table.metadata.get('source_range', 'not available')}",
                ],
                evidence_refs=[f"{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | column {metric_col}]"],
                numeric_value=value,
            )
        return None

    def _detect_target_metric(self, question: str) -> str:
        lowered = question.lower()
        if "underperform" in lowered:
            return "performance"
        candidates = ["revenue", "margin", "profit", "score", "performance", "risk", "cost"]
        for candidate in candidates:
            if candidate in lowered:
                return candidate
        tokens = tokenize(question)
        return tokens[-1] if tokens else "metric"

    def _extract_series(self, frame: pd.DataFrame, target_metric: str) -> tuple[dict[str, float], int | None] | None:
        frame = frame.copy()
        metric_columns = [column for column in frame.columns if self._looks_temporal(column)]
        if not metric_columns:
            return self._extract_series_from_long_format(frame, target_metric)

        id_columns = [column for column in frame.columns if column not in metric_columns and not str(column).startswith("_")]
        if not id_columns:
            row = frame.iloc[0]
            series = {}
            for column in metric_columns:
                value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    series[str(column)] = float(value)
            return (series, int(row["_source_row"]) if "_source_row" in row and pd.notna(row["_source_row"]) else None) if series else None

        for _, row in frame.iterrows():
            row_text = " ".join(compact_whitespace(str(row[column])).lower() for column in id_columns)
            if target_metric in row_text:
                series = {}
                for column in metric_columns:
                    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        series[str(column)] = float(value)
                if series:
                    return series, (int(row["_source_row"]) if "_source_row" in row and pd.notna(row["_source_row"]) else None)
        return None

    def _extract_series_from_long_format(self, frame: pd.DataFrame, target_metric: str) -> tuple[dict[str, float], int | None] | None:
        lowered_columns = {str(column).lower(): column for column in frame.columns}
        time_col = next((column for name, column in lowered_columns.items() if name in {"period", "quarter", "month", "year"}), None)
        metric_col = next((column for name, column in lowered_columns.items() if target_metric in name), None)
        if time_col and metric_col:
            series_frame = frame[[time_col, metric_col]].copy()
            series_frame[metric_col] = pd.to_numeric(series_frame[metric_col], errors="coerce")
            series_frame = series_frame.dropna(subset=[metric_col])
            if series_frame.empty:
                return None
            source_row = int(series_frame.iloc[0]["_source_row"]) if "_source_row" in series_frame.columns and pd.notna(series_frame.iloc[0]["_source_row"]) else None
            return {str(row[time_col]): float(row[metric_col]) for _, row in series_frame.iterrows()} or None, source_row
        return None

    def _extract_ranking(self, frame: pd.DataFrame, target_metric: str) -> tuple[str, str, pd.DataFrame] | None:
        prepared = frame.copy()
        metric_col = self._best_numeric_column(prepared, target_metric)
        if not metric_col:
            return None
        prepared[metric_col] = pd.to_numeric(prepared[metric_col], errors="coerce")
        prepared = prepared.dropna(subset=[metric_col])
        if prepared.empty:
            return None

        non_numeric = [column for column in prepared.columns if column != metric_col and not str(column).startswith("_")]
        preferred = ["department", "business_unit", "business unit", "team", "region", "function"]
        dimension_col = next(
            (column for column in non_numeric if str(column).strip().lower() in preferred),
            non_numeric[0] if non_numeric else None,
        )
        if dimension_col is None:
            return None
        prepared[dimension_col] = prepared[dimension_col].astype(str)
        prepared = prepared[prepared[dimension_col].str.strip() != ""]
        return dimension_col, metric_col, prepared[[dimension_col, metric_col]]

    def _best_numeric_column(self, frame: pd.DataFrame, target_metric: str) -> str | None:
        numeric_columns = self._numeric_columns(frame)
        if not numeric_columns:
            return None
        tokens = tokenize(target_metric)
        best = None
        best_score = -1
        for column in numeric_columns:
            name_tokens = tokenize(str(column))
            score = len(set(tokens) & set(name_tokens))
            if score > best_score:
                best = column
                best_score = score
        if best_score <= 0 and target_metric not in {"metric", "value"}:
            return None
        return best or numeric_columns[0]

    def _numeric_columns(self, frame: pd.DataFrame) -> list[str]:
        numeric_columns: list[str] = []
        for column in frame.columns:
            if str(column).startswith("_"):
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().sum() >= max(2, len(frame) // 2):
                numeric_columns.append(str(column))
        return numeric_columns

    @staticmethod
    def _looks_temporal(column: object) -> bool:
        text = str(column).lower()
        return bool(re.search(r"\b(q[1-4]|20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", text))
