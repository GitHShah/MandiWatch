# Methodology

The audit scans CSVs in chunks and records schema, date coverage, missingness, parse failures, price signs, range flags, coverage, and duplicate indicators. Exact/likely duplicate counts in the annual audit are explicitly **within chunk**; the 2025/2026 report separately compares full normalized row keys on actual boundary dates.

The multi-commodity cache is versioned from source file metadata and schema version, then partitioned by commodity and parsed arrival year (not filename year). It holds derived valid records only; raw CSVs are untouched. PyArrow applies selected commodity, date, and dimension predicates before values are materialized. Dashboard monthly medians are calculated from all selected cached observations, not medians of chunks. This makes cache building disk-intensive but prevents ordinary UI navigation from rescanning the raw CSV corpus.

Cache construction is restartable: each part is written as a hidden temporary file, read back with a bounded Arrow thread pool, and renamed only after validation. `build_progress.json` records source-file/chunk/part provenance and is updated atomically as work completes. Ctrl+C leaves valid completed parts and an interrupted journal; rerunning resumes them. Only a directory containing a validated `cache_manifest.json` is exposed through `commodity_cache_latest.json`. If a Windows sync provider denies a directory rename, the completion manifest is copied last so a partial fallback directory remains undiscoverable.

An exact normalized row identity consists of state, district, market, commodity, variety, grade, parsed arrival date, and the three reported price fields. When an identical identity appears in selected cache partitions, the earliest source filename is retained and the later row is excluded. Source filenames remain in cache records, the dashboard declares any removals, and the boundary audit reports the affected 2025/2026 keys. This is intentionally narrower than fuzzy matching: similar commodity names, varieties, grades, or prices are not merged.

Commodity trend charts report monthly median and mean modal prices with observations and market counts. Rows with an invalid date or non-positive/non-numeric modal price are excluded only from derived analyses. Commodities are never alias-merged; cross-commodity price levels may not be comparable because units, grades, and varieties are unverified.

The seasonality test uses each complete year as a paired block. Values are divided by their own annual median before a Friedman test; Kendall's W is reported. Pairwise Wilcoxon tests use Holm correction. These procedures do not make raw rows independent and results remain sensitive to market composition.

Anomaly scores use prior observations only: a rolling median and MAD of `t-1` backward. A zero MAD produces no score instead of a division-by-zero artefact.

Forecast validation reserves the final 12 existing monthly points in chronological order and compares naïve and seasonal-naïve baselines. A seasonal forecast is refused if calendar months are missing, rather than using positional values as a substitute. No random split or future values are used. Forward baseline tables are kept distinct from observed validation values; these simple baselines do not supply prediction intervals.
