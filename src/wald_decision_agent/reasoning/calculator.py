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
        if any(term in lowered for term in ["estimated", "estimate", "calculated", "calculate", "computed", "derive", "derived", "formula", "methodology"]):
            result = self._explain_metric_method(question, tables)
            if result:
                return result
        if self._has_temporal_constraint(lowered):
            result = self._calculate_temporal_lookup(question, tables)
            if result:
                return result
        if self._is_breakdown_query(lowered):
            result = self._calculate_breakdown(question, tables)
            if result:
                return result

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

        if any(term in lowered for term in ["risk", "risks", "challenge", "challenges", "item", "items", "priority", "priorities"]):
            result = self._calculate_list(question, tables)
            if result:
                return result

        return self._calculate_ranking(question, tables) or self._calculate_trend(question, tables)

    def _calculate_temporal_lookup(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        requested_period = self._detect_requested_period(question)
        if not requested_period:
            return None
        target_metric = self._detect_target_metric(question)
        for table in tables:
            extracted = self._extract_series(table.dataframe, target_metric)
            if not extracted:
                continue
            series, source_row = extracted
            matched_label = next((label for label in series if requested_period in str(label).lower()), None)
            if not matched_label:
                continue
            value = series[matched_label]
            answer = f"{target_metric.title()} in {matched_label} is {value:,.2f}."
            findings = [
                f"The grounded value for {matched_label} is {value:,.2f}.",
                f"Series available: {', '.join(f'{label}={metric_value:,.2f}' for label, metric_value in series.items())}",
            ]
            trace = [
                f"Temporal lookup resolved from {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}.",
                f"Requested period: {matched_label}",
                f"Source row: {source_row}" if source_row is not None else "Source row: not available",
                f"Source range: {table.metadata.get('source_range', 'not available')}",
            ]
            return CalculationResult(
                answer=answer,
                findings=findings,
                trace=trace,
                evidence_refs=[
                    f"[{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | row {source_row if source_row is not None else '?'} | period {matched_label}]]({table.source_path})"
                ],
                chart_data={
                    "type": "line",
                    "labels": list(series.keys()),
                    "values": list(series.values()),
                    "title": f"{target_metric.title()} trend",
                },
                numeric_value=float(value),
            )
        return None

    def _calculate_breakdown(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        metric_requests, unsupported_metrics = self._detect_metric_requests(question)
        if not metric_requests:
            return None

        best_candidate: tuple[int, StructuredTable, str, list[str], pd.DataFrame] | None = None
        for table in tables:
            frame = table.dataframe.copy()
            numeric_columns = self._numeric_columns(frame)
            if not numeric_columns:
                continue
            dimension_col = self._best_dimension_column(frame, metric_col="")
            if not dimension_col:
                continue
            metric_columns: list[str] = []
            match_score = 0
            for metric_request in metric_requests:
                metric_col = self._best_numeric_column(frame, metric_request)
                if metric_col and metric_col not in metric_columns:
                    metric_columns.append(metric_col)
                    match_score += self._metric_match_score(metric_request, metric_col)
            if match_score == 0:
                continue
            candidate = (match_score, table, dimension_col, metric_columns, frame)
            if best_candidate is None or candidate[0] > best_candidate[0]:
                best_candidate = candidate

        if best_candidate is None:
            return None

        _, table, dimension_col, metric_columns, frame = best_candidate
        working = frame[[dimension_col, *metric_columns, *([col for col in frame.columns if str(col) == "_source_row"])]].copy()
        for metric_col in metric_columns:
            working[metric_col] = pd.to_numeric(working[metric_col], errors="coerce")
        working = working.dropna(subset=metric_columns, how="all")
        if working.empty:
            return None

        requested_labels = ", ".join(metric_columns)
        summary_parts = []
        findings = []
        for _, row in working.iterrows():
            entity = str(row[dimension_col]).strip()
            if not entity:
                continue
            metric_parts = []
            for metric_col in metric_columns:
                if pd.notna(row[metric_col]):
                    metric_parts.append(f"{metric_col} = {float(row[metric_col]):,.2f}")
            if not metric_parts:
                continue
            summary_parts.append(f"{entity} {', '.join(metric_parts)}")
            findings.append(f"{entity}: {', '.join(metric_parts)}")

        if not findings:
            return None

        answer = f"{requested_labels} by {dimension_col}: " + "; ".join(summary_parts[:4]) + "."
        if unsupported_metrics:
            answer += f" I do not see a grounded metric named {', '.join(f'`{metric}`' for metric in unsupported_metrics)} in the uploaded tables."
            findings.append(
                "Unsupported requested metric(s): "
                + ", ".join(f"`{metric}`" for metric in unsupported_metrics)
                + ". The answer only reports grounded columns present in the uploaded data."
            )

        trace = [
            f"Breakdown query resolved from {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}.",
            f"Dimension column: {dimension_col}",
            f"Metric columns: {', '.join(metric_columns)}",
            f"Source range: {table.metadata.get('source_range', 'not available')}",
        ]

        first_metric = metric_columns[0]
        chart_rows = working.dropna(subset=[first_metric])
        chart_data = None
        if not chart_rows.empty:
            chart_data = {
                "type": "bar",
                "labels": chart_rows[dimension_col].astype(str).tolist(),
                "values": [float(value) for value in chart_rows[first_metric].tolist()],
                "title": f"{first_metric} by {dimension_col}",
            }

        evidence_ref = (
            f"[{table.source_path.name} "
            f"[{table.metadata.get('sheet_name', 'Sheet1')} | columns {dimension_col}, {', '.join(metric_columns)}]]({table.source_path})"
        )
        return CalculationResult(
            answer=answer,
            findings=findings,
            trace=trace,
            evidence_refs=[evidence_ref],
            chart_data=chart_data,
        )

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
                    f"[{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | row {source_row if source_row is not None else '?'} | range {table.metadata.get('source_range', 'n/a')}]]({table.source_path})"
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
                    f"[{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | row {int(best_row['_source_row']) if '_source_row' in best_row and pd.notna(best_row['_source_row']) else '?'} | metric {metric_col}]]({table.source_path})"
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
                evidence_refs=[f"[{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | column {metric_col}]]({table.source_path})"],
                numeric_value=value,
            )
        return None

    def _calculate_list(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        from ..core.logging import get_logger
        logger = get_logger("reasoning.calculator")
        
        target_metric = self._detect_target_metric(question)
        logger.info(f"Qualitative list lookup for metric: {target_metric}")
        if target_metric not in ["risk", "item", "priority", "challenge"]:
            return None

        # Look for the best table matching the target metric using a scoring system
        candidates: list[tuple[int, StructuredTable]] = []
        
        for table in tables:
            frame = table.dataframe.copy()
            cols = [str(c).lower() for c in frame.columns]
            score = 0
            
            # Bonus if filename matches target metric
            if target_metric in table.source_path.name.lower():
                score += 5
            
            # Bonus if columns match keywords
            if target_metric == "risk":
                if any(m in c for m in ["risk", "severity", "status"] for c in cols):
                    score += 2
            elif any(target_metric in c for c in cols):
                score += 2
                
            if score > 0:
                candidates.append((score, table))
                
        if not candidates:
            return None
            
        # Select best table by score
        best_table = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
        logger.info(f"Target metric '{target_metric}' selected best table: {best_table.source_path.name} (score {sorted(candidates, key=lambda x: x[0], reverse=True)[0][0]})")
            
        frame = best_table.dataframe.copy()
        cols = list(frame.columns)
        
        # Determine sorting (e.g. prioritize High severity)
        severity_col = next((c for c in cols if "severity" in str(c).lower()), None)
        if severity_col:
            sev_map = {"high": 3, "medium": 2, "low": 1}
            frame["_sev_score"] = frame[severity_col].astype(str).str.lower().map(sev_map).fillna(0)
            frame = frame.sort_values("_sev_score", ascending=False)
            
        # Determine primary column to list
        primary_col = next((c for c in cols if any(m in str(c).lower() for m in [target_metric, "category", "item", "description", "title", "strategy"])), cols[0])
        
        items = []
        findings = []
        for _, row in frame.iterrows():
            item_name = str(row[primary_col]).strip()
            if not item_name or item_name.lower() in ["none", "nan", "null"]:
                continue
            
            detail = ""
            if severity_col:
                detail = f" ({row[severity_col]} severity)"
            
            items.append(f"{item_name}{detail}")
            findings.append(f"{item_name}: " + ", ".join(f"{c}={row[c]}" for c in cols if c != primary_col and not str(c).startswith("_")))
            
        if not items:
            return None
            
        answer = f"The identified {target_metric}s include: " + ", ".join(items[:5]) + "."
        trace = [
            f"Qualitative list generated from {best_table.source_path.name} / {best_table.metadata.get('sheet_name', 'Sheet1')}.",
            f"Primary column: {primary_col}",
            f"Items found: {len(items)}",
        ]
        
        return CalculationResult(
            answer=answer,
            findings=findings[:5],
            trace=trace,
            evidence_refs=[f"[{best_table.source_path.name} [{best_table.metadata.get('sheet_name', 'Sheet1')}]]({best_table.source_path})"],
        )

    def _explain_metric_method(self, question: str, tables: list[StructuredTable]) -> CalculationResult | None:
        target_metric = self._detect_target_metric(question)
        supporting_refs: list[str] = []

        for table in tables:
            frame = table.dataframe.copy()
            metric_col = self._best_numeric_column(frame, target_metric)
            if not metric_col:
                continue

            dimension_col = self._best_dimension_column(frame, metric_col)
            preview = []
            if dimension_col:
                preview_frame = frame[[dimension_col, metric_col]].copy()
                preview_frame[metric_col] = pd.to_numeric(preview_frame[metric_col], errors="coerce")
                preview_frame = preview_frame.dropna(subset=[metric_col]).head(3)
                preview = [
                    f"{row[dimension_col]}={float(row[metric_col]):,.2f}"
                    for _, row in preview_frame.iterrows()
                ]

            for related_table in tables:
                related_columns = " ".join(str(column).lower() for column in related_table.dataframe.columns)
                if target_metric == "risk" and "severity" in related_columns:
                    supporting_refs.append(
                        f"[{related_table.source_path.name} [{related_table.metadata.get('sheet_name', 'Sheet1')} | qualitative risk fields]]({related_table.source_path})"
                    )

            evidence_refs = [
                f"[{table.source_path.name} [{table.metadata.get('sheet_name', 'Sheet1')} | column {metric_col}]]({table.source_path})"
            ]
            for ref in supporting_refs:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)

            preview_text = ", ".join(preview) if preview else f"reported in column `{metric_col}`"
            answer = (
                f"The uploaded files report {metric_col} values in {table.source_path.name} "
                f"({table.metadata.get('sheet_name', 'Sheet1')}), but they do not document the formula or methodology used to estimate them. "
                f"I can confirm the recorded values, not how the score was derived."
            )
            findings = [
                f"{metric_col} values are present in `{table.source_path.name}` / `{table.metadata.get('sheet_name', 'Sheet1')}`.",
                f"Recorded examples: {preview_text}.",
            ]
            if supporting_refs:
                findings.append("Related qualitative risk fields are available, but they do not expose a numeric scoring formula.")
            trace = [
                f"Methodology check searched numeric metric column `{metric_col}`.",
                f"Source workbook/table: {table.source_path.name} / {table.metadata.get('sheet_name', 'Sheet1')}",
                f"Source range: {table.metadata.get('source_range', 'not available')}",
            ]
            return CalculationResult(
                answer=answer,
                findings=findings,
                trace=trace,
                evidence_refs=evidence_refs,
            )
        return None

    def _detect_target_metric(self, question: str) -> str:
        lowered = question.lower()
        if "underperform" in lowered:
            return "performance"
        candidates = ["revenue", "margin", "profit", "score", "performance", "risk", "risks", "item", "priority", "challenge", "cost"]
        for candidate in candidates:
            if candidate in lowered:
                # Normalize plural
                if candidate == "risks": return "risk"
                return candidate
        tokens = tokenize(question)
        return tokens[-1] if tokens else "metric"

    def _detect_metric_requests(self, question: str) -> tuple[list[str], list[str]]:
        lowered = question.lower()
        requests: list[str] = []
        unsupported: list[str] = []

        phrase_candidates = [
            "actual revenue",
            "actual margin",
            "actual cost",
            "revenue target",
            "margin target",
            "cost target",
            "performance score",
            "risk score",
            "revenue",
            "margin",
            "cost",
            "score",
        ]
        for phrase in phrase_candidates:
            if phrase in lowered and phrase not in requests:
                requests.append(phrase)

        if any(term in lowered for term in ["gain", "gained", "gains"]):
            unsupported.append("gained")

        return requests, unsupported

    def _is_breakdown_query(self, lowered_question: str) -> bool:
        return any(
            phrase in lowered_question
            for phrase in [
                "for each",
                "each region",
                "each regions",
                "by region",
                "across regions",
                "for every region",
                "per region",
            ]
        )

    def _has_temporal_constraint(self, lowered_question: str) -> bool:
        return any(term in lowered_question for term in ["q1", "q2", "q3", "q4", "quarter", "quarterly", "fy"])

    def _detect_requested_period(self, question: str) -> str | None:
        lowered = question.lower()
        for period in ["q1", "q2", "q3", "q4"]:
            if period in lowered:
                return period
        return None

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
        cols_to_keep = [dimension_col, metric_col]
        if "_source_row" in prepared.columns:
            cols_to_keep.append("_source_row")
        return dimension_col, metric_col, prepared[cols_to_keep]

    def _best_numeric_column(self, frame: pd.DataFrame, target_metric: str) -> str | None:
        numeric_columns = self._numeric_columns(frame)
        if not numeric_columns:
            return None
        best = None
        best_score = -1
        for column in numeric_columns:
            score = self._metric_match_score(target_metric, str(column))
            if score > best_score:
                best = column
                best_score = score
        if best_score <= 0 and target_metric not in {"metric", "value"}:
            return None
        return best or numeric_columns[0]

    def _metric_match_score(self, target_metric: str, column_name: str) -> int:
        tokens = tokenize(target_metric)
        name_tokens = tokenize(str(column_name))
        qualifier_tokens = {"actual", "target", "plan"}
        core_tokens = [token for token in tokens if token not in qualifier_tokens]
        overlap = set(tokens) & set(name_tokens)
        core_overlap = set(core_tokens) & set(name_tokens)
        score = len(overlap)
        if core_tokens and not core_overlap:
            return -1
        if "actual" in tokens and "actual" in name_tokens:
            score += 2
        if "target" in tokens and "target" in name_tokens:
            score += 2
        if "plan" in tokens and "plan" in name_tokens:
            score += 2
        return score

    def _best_dimension_column(self, frame: pd.DataFrame, metric_col: str) -> str | None:
        non_numeric = [column for column in frame.columns if str(column) != metric_col and not str(column).startswith("_")]
        preferred = ["department", "business_unit", "business unit", "team", "region", "function"]
        return next(
            (column for column in non_numeric if str(column).strip().lower() in preferred),
            non_numeric[0] if non_numeric else None,
        )

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
