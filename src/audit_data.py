"""Chunked annual-file quality audit. Run with ``python -m src.audit_data``."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import CHUNK_SIZE, PRICE_COLUMNS, PROCESSED_DIR, REQUIRED_COLUMNS, ensure_output_dirs
from .data_io import annual_files, canonicalize_frame, clean_text, normalize_column_name, source_manifest


def _blank_count(series: pd.Series) -> int:
    return int((series.isna() | series.astype("string").str.strip().eq("")).sum())


def audit_file(path: Path, chunksize: int = CHUNK_SIZE) -> dict[str, object]:
    """Scan one input once and return aggregate quality measures."""
    raw_header = pd.read_csv(path, nrows=0)
    normalized_columns = [normalize_column_name(value) for value in raw_header.columns]
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(normalized_columns))
    result: dict[str, object] = {"file": path.name, "filename_year": int(path.stem) if path.stem.isdigit() else None, "raw_columns": " | ".join(map(str, raw_header.columns)), "normalized_columns": " | ".join(normalized_columns), "missing_required_columns": " | ".join(missing_columns), "file_bytes": path.stat().st_size}
    counts: Counter[str] = Counter()
    commodities: set[str] = set(); states: set[str] = set(); markets: set[str] = set(); months: set[str] = set()
    min_date = max_date = None; within_chunk_duplicates = likely_duplicates = 0
    for raw in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        counts["row_count"] += len(raw)
        normalized = raw.copy(); normalized.columns = normalized_columns
        for column in REQUIRED_COLUMNS:
            if column in normalized:
                counts[f"missing_{column}"] += _blank_count(normalized[column])
        if "arrival_date" not in normalized:
            continue
        frame = canonicalize_frame(normalized); dates = frame["arrival_date"]
        counts["invalid_date_count"] += int(dates.isna().sum())
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min() if min_date is None else min(min_date, valid_dates.min())
            max_date = valid_dates.max() if max_date is None else max(max_date, valid_dates.max())
            filename_year = result["filename_year"]
            if filename_year is not None: counts["date_outside_filename_year_count"] += int(valid_dates.dt.year.ne(filename_year).sum())
            months.update(valid_dates.dt.to_period("M").astype(str))
        for column in PRICE_COLUMNS:
            if column in normalized:
                raw_price, numeric = normalized[column], frame[column]
                missing_or_blank = raw_price.isna() | raw_price.astype("string").str.strip().eq("")
                counts[f"non_numeric_{column}_count"] += int((numeric.isna() & ~missing_or_blank).sum())
                counts[f"zero_{column}_count"] += int(numeric.eq(0).sum()); counts[f"negative_{column}_count"] += int(numeric.lt(0).sum())
        range_valid = frame["min_price"].notna() & frame["max_price"].notna()
        counts["min_greater_than_max_count"] += int((range_valid & frame["min_price"].gt(frame["max_price"])).sum())
        bounded = frame[list(PRICE_COLUMNS)].notna().all(axis=1) & frame["min_price"].le(frame["max_price"])
        counts["modal_outside_min_max_count"] += int((bounded & ((frame["modal_price"] < frame["min_price"]) | (frame["modal_price"] > frame["max_price"]))).sum())
        key_columns = [column for column in ("state", "district", "market", "commodity", "variety", "grade", "arrival_date", "min_price", "max_price", "modal_price") if column in frame]
        within_chunk_duplicates += int(frame.duplicated().sum()); likely_duplicates += int(frame.duplicated(subset=key_columns).sum())
        for column, target in (("commodity", commodities), ("state", states), ("market", markets)):
            target.update(clean_text(frame[column]).dropna().unique().tolist())
    result.update(counts); total = int(counts["row_count"])
    present_month_numbers = {pd.Period(value).month for value in months}
    result.update({"date_start": min_date.date().isoformat() if min_date is not None else None, "date_end": max_date.date().isoformat() if max_date is not None else None, "months_present": len(months), "missing_calendar_months": " | ".join(str(month) for month in range(1, 13) if month not in present_month_numbers), "distinct_commodities": len(commodities), "distinct_states": len(states), "distinct_markets": len(markets), "within_chunk_exact_duplicate_count": within_chunk_duplicates, "within_chunk_likely_duplicate_count": likely_duplicates, "estimated_csv_memory_bytes": int(path.stat().st_size * 1.5)})
    for column in REQUIRED_COLUMNS: result[f"missing_{column}_rate"] = counts[f"missing_{column}"] / total if total else None
    return result


def boundary_overlap_report(first: Path, second: Path) -> dict[str, object]:
    """Compare complete row keys across adjacent actual-date boundaries, never date alone."""
    key = ["state", "district", "market", "commodity", "variety", "grade", "arrival_date", "min_price", "max_price", "modal_price"]
    later = pd.concat([canonicalize_frame(chunk)[key] for chunk in pd.read_csv(second, usecols=lambda column: normalize_column_name(column) in key, chunksize=CHUNK_SIZE, low_memory=False)], ignore_index=True)
    later = later.loc[later["arrival_date"].dt.year.lt(int(second.stem))].dropna(subset=key)
    if later.empty: return {"earlier_file": first.name, "later_file": second.name, "later_rows_outside_filename_year": 0, "overlap_dates": "", "exact_key_matches": 0}
    later_keys = set(map(tuple, later.assign(arrival_date=later["arrival_date"].astype(str)).to_numpy())); dates = set(later["arrival_date"].dt.normalize()); matches = 0
    for raw in pd.read_csv(first, usecols=lambda column: normalize_column_name(column) in key, chunksize=CHUNK_SIZE, low_memory=False):
        frame = canonicalize_frame(raw)[key]; frame = frame.loc[frame["arrival_date"].dt.normalize().isin(dates)].dropna(subset=key)
        matches += sum(tuple(row) in later_keys for row in frame.assign(arrival_date=frame["arrival_date"].astype(str)).to_numpy())
    return {"earlier_file": first.name, "later_file": second.name, "later_rows_outside_filename_year": len(later), "overlap_dates": " | ".join(sorted(date.date().isoformat() for date in dates)), "exact_key_matches": matches}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--files", nargs="*", help="Optional annual filenames, e.g. 2025.csv 2026.csv."); parser.add_argument("--overlap-only", action="store_true", help="Only regenerate the exact 2025/2026 boundary comparison."); args = parser.parse_args()
    if args.overlap_only:
        ensure_output_dirs(); all_files = {int(path.stem): path for path in annual_files()}
        if not {2025, 2026}.issubset(all_files): raise FileNotFoundError("Both 2025.csv and 2026.csv are required for boundary comparison.")
        overlap = boundary_overlap_report(all_files[2025], all_files[2026]); pd.DataFrame([overlap]).to_csv(PROCESSED_DIR / "data_boundary_overlap.csv", index=False); print(json.dumps(overlap, indent=2)); return
    ensure_output_dirs(); files = annual_files(); files = [path for path in files if not args.files or path.name in set(args.files)]
    if not files: raise FileNotFoundError("No numeric annual CSV files found in data/raw.")
    summaries = []
    for path in files: print(f"Auditing {path.name} …", flush=True); summaries.append(audit_file(path))
    report = pd.DataFrame(summaries).sort_values("filename_year"); report.to_csv(PROCESSED_DIR / "data_audit_by_file.csv", index=False); source_manifest(files).to_csv(PROCESSED_DIR / "source_manifest.csv", index=False)
    all_files = {int(path.stem): path for path in annual_files()}; overlaps = []
    if 2025 in all_files and 2026 in all_files and (not args.files or {"2025.csv", "2026.csv"}.issubset(args.files)):
        print("Checking 2025/2026 boundary overlap …", flush=True); overlaps.append(boundary_overlap_report(all_files[2025], all_files[2026]))
    pd.DataFrame(overlaps).to_csv(PROCESSED_DIR / "data_boundary_overlap.csv", index=False)
    payload = {"files_audited": len(report), "total_rows": int(report["row_count"].sum()), "date_start": report["date_start"].min(), "date_end": report["date_end"].max(), "outputs": ["data_audit_by_file.csv", "data_boundary_overlap.csv", "source_manifest.csv"]}; (PROCESSED_DIR / "data_audit_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8"); print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
