# Data dictionary

The raw files are immutable. Derived frames normalize only column formatting (case, surrounding whitespace, and punctuation) into the canonical names below; values are trimmed but not alias-merged.

| Canonical field | Meaning inferred from header | Handling |
|---|---|---|
| `state`, `district`, `market` | Geographic/market labels | Trim blank values; no alias mapping. |
| `commodity`, `variety`, `grade` | Product labels | Trim blank values; no inferred equivalence. |
| `arrival_date` | Reported arrival date | Parse with errors coerced to missing in derived data. |
| `min_price`, `max_price`, `modal_price` | Reported price fields | Numeric parse in derived data; values are audited, not silently corrected. |
| `commodity_code` | Source commodity identifier | Preserved as text where present. |

The source price unit is **not verified** by local metadata and is labelled “source unit to be confirmed”.
