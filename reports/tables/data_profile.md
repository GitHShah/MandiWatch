# Raw Dataset Audit & Profile (Agmarknet 2015–2026)

**Generated:** 2026-11-04 00:50:54  
**Total Records:** 48,229,507 rows across 12 files  
**Total Raw Size:** 4.15 GB (4,461,092,131 bytes)  
**Date Range:** 2015-01-01 to 2026-04-21  

---

## 1. Schema & Column Specifications

| Column Name | Dtype (Inferred) | Missing Count | Missing Rate | Description |
| :--- | :--- | :--- | :--- | :--- |
| `State` | string | 0 | 0.0000% | Standard Agmarknet field |
| `District` | string | 0 | 0.0000% | Standard Agmarknet field |
| `Market` | string | 0 | 0.0000% | Standard Agmarknet field |
| `Commodity` | string | 0 | 0.0000% | Standard Agmarknet field |
| `Variety` | string | 0 | 0.0000% | Standard Agmarknet field |
| `Grade` | string | 0 | 0.0000% | Standard Agmarknet field |
| `Arrival_Date` | string (date) | 0 | 0.0000% | Standard Agmarknet field |
| `Min_Price` | float64 | 0 | 0.0000% | Standard Agmarknet field |
| `Max_Price` | float64 | 0 | 0.0000% | Standard Agmarknet field |
| `Modal_Price` | float64 | 0 | 0.0000% | Standard Agmarknet field |
| `Commodity_Code` | int64 | 0 | 0.0000% | Standard Agmarknet field |


## 2. Annual Coverage & Volume (2015–2026)

| File | Rows | File Size (MB) | Min Date | Max Date | Volume Anomaly / Coverage Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `2015.csv` | 3,673,823 | 313.6 MB | 2015-01-01 | 2015-12-31 | Normal full year |
| `2016.csv` | 3,634,455 | 311.9 MB | 2016-01-01 | 2016-12-31 | Normal full year |
| `2017.csv` | 4,089,861 | 353.0 MB | 2017-01-01 | 2017-12-31 | Normal full year |
| `2018.csv` | 4,285,826 | 370.7 MB | 2018-01-01 | 2018-12-31 | Normal full year |
| `2019.csv` | 4,319,450 | 375.0 MB | 2019-01-01 | 2019-12-31 | Normal full year |
| `2020.csv` | 3,548,553 | 308.2 MB | 2020-01-01 | 2020-12-31 | Noticeable drop in volume (-17.8% vs 2019) due to COVID-19 nationwide lockdown mandi closures |
| `2021.csv` | 3,827,438 | 332.6 MB | 2021-01-01 | 2021-12-31 | Normal full year |
| `2022.csv` | 4,476,890 | 390.2 MB | 2022-01-01 | 2022-12-31 | Normal full year |
| `2023.csv` | 4,415,690 | 384.5 MB | 2023-01-01 | 2023-12-31 | Normal full year |
| `2024.csv` | 5,544,500 | 507.7 MB | 2024-01-01 | 2024-12-31 | High volume (>5.5M rows): expanded market onboarding / digitization |
| `2025.csv` | 5,819,482 | 547.1 MB | 2025-01-01 | 2025-12-30 | High volume (>5.5M rows): expanded market onboarding / digitization |
| `2026.csv` | 593,539 | 59.8 MB | 2025-12-30 | 2026-04-21 | Partial year (593,539 rows, ~3.7 months through 2026-04-21) |


## 3. Categorical Cardinality

| Dimension | Unique Count (Raw Text) | Notes |
| :--- | :--- | :--- |
| **States** | 33 | Includes Union Territories and minor spelling variants |
| **Districts** | 632 | Contains spelling, casing, and bifurcation variants |
| **Markets** | 6,733 | Regulated wholesale agricultural markets (APMC mandis) |
| **Commodities** | 380 | Distinct crop items |
| **Varieties** | 1,317 | Sub-varieties / botanical strains |
| **Grades** | 18 | Quality grades (FAQ, Medium, Large, Small, etc.) |


## 4. Price Statistics & Unit-Mix Inspection

Price statistics (Modal Price in reported units) for the top 20 commodities:

| Commodity | Rows | Min | 1st Pct | Median | 99th Pct | Max | Spread (P99/P1) | Unit Flag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Potato | 2,059,169 | 0.2 | 270.3 | 1120.0 | 5000.0 | 11200000.0 | 18.5x | Normal |
| Onion | 2,015,561 | 0.1 | 400.0 | 1700.0 | 7000.0 | 917588483.0 | 17.5x | Normal |
| Tomato | 1,913,317 | 0.3 | 300.0 | 1685.0 | 7250.0 | 404000.0 | 24.2x | Normal |
| Wheat | 1,856,470 | 1.0 | 1380.0 | 1940.0 | 3200.0 | 63869.0 | 2.3x | Normal |
| Brinjal | 1,727,330 | 0.2 | 375.0 | 1600.0 | 6400.0 | 540000.0 | 17.1x | Normal |
| Paddy (Dhan)(Common) | 1,581,608 | 1.0 | 1092.0 | 1620.0 | 3400.0 | 28000.0 | 3.1x | Normal |
| Green Chilli | 1,498,393 | 0.3 | 650.0 | 2900.0 | 13904.0 | 3950000.0 | 21.4x | Normal |
| Banana | 1,207,997 | 0.1 | 325.0 | 2500.0 | 8000.0 | 500000.0 | 24.6x | Normal |
| Cauliflower | 1,176,456 | 0.4 | 250.0 | 1500.0 | 7000.0 | 250000.0 | 28.0x | Normal |
| Bhindi (Ladies Finger) | 1,137,886 | 0.5 | 550.0 | 2300.0 | 7500.0 | 1000000.0 | 13.6x | Normal |
| Cabbage | 1,070,631 | 0.3 | 210.0 | 1300.0 | 5250.0 | 700000.0 | 25.0x | Normal |
| Cucumbar (Kheera) | 1,030,662 | 0.1 | 300.0 | 1600.0 | 6000.0 | 504000.0 | 20.0x | Normal |
| Bengal Gram (Gram)(Whole) | 973,374 | 1.0 | 3061.0 | 4850.0 | 9550.0 | 55750.0 | 3.1x | Normal |
| Bottle gourd | 954,092 | 0.3 | 300.0 | 1200.0 | 4000.0 | 365500.0 | 13.3x | Normal |
| Rice | 921,349 | 2.0 | 1670.0 | 2670.0 | 5500.0 | 210000.0 | 3.3x | Normal |
| Pumpkin | 889,914 | 0.3 | 300.0 | 1200.0 | 3900.0 | 160000.0 | 13.0x | Normal |
| Bitter gourd | 887,270 | 0.3 | 550.0 | 3000.0 | 8800.0 | 500000.0 | 16.0x | Normal |
| Mustard | 822,353 | 33.0 | 3000.0 | 4400.0 | 7650.0 | 112500.0 | 2.5x | Normal |
| Maize | 822,258 | 1.0 | 1000.0 | 1600.0 | 3000.0 | 148300.0 | 3.0x | Normal |
| Apple | 798,298 | 1.0 | 1650.0 | 6640.0 | 20000.0 | 1100000.0 | 12.1x | Normal |


### Flagged Commodities for Potential Unit Mixing

The following commodities exhibit extreme low-end pricing ($P_1 < 25$) combined with high-end pricing ($P_{99} > 1,200$), indicating potential entry of ₹/kg rather than standard ₹/quintal:

| Commodity | Rows | P1 | Median | P99 | Max | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Raddish | 666,841 | 10.0 | 825.0 | 4450.0 | 40000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Coriander (Leaves) | 361,180 | 3.0 | 2500.0 | 13500.0 | 750000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Spinach | 287,403 | 3.0 | 900.0 | 4000.0 | 13000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Coconut | 275,329 | 36.0 | 4000.0 | 22000.0 | 1800000.0 | High spread (possible unit mix or severe low outlier) |
| Onion Green | 135,987 | 5.0 | 4000.0 | 9000.0 | 180000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Methi (Leaves) | 121,353 | 5.0 | 1050.0 | 7250.0 | 200000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Mint (Pudina) | 106,923 | 2.0 | 3500.0 | 8050.0 | 170000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Lime | 101,300 | 1.0 | 3400.0 | 20000.0 | 330000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |
| Tender Coconut | 61,535 | 30.0 | 3100.0 | 23000.0 | 120000.0 | High spread (possible unit mix or severe low outlier) |
| Marigold (Calcutta) | 19,894 | 7.5 | 10000.0 | 40000.0 | 500000.0 | Possible Rs/kg vs Rs/quintal unit mix (e.g. P1 <= 20 vs P99 >= 1200) |


## 5. Missing Values & Data Quality Anomalies

- **Total Rows:** 48,229,507

- **Missing Dates:** 0 (0.0000%)

- **Unparseable Dates:** 0 (0.0000%)

- **Date Formats Supported:** Handled both `YYYY-MM-DD` and `DD/MM/YYYY` seamlessly (zero unparseable dates when using multi-format parser).

- **Non-Positive Prices:**

  - `Modal_Price <= 0`: 0 (0.0000%)

  - `Min_Price <= 0`: 489,330 (1.0146%)

  - `Max_Price <= 0`: 481,015 (0.9973%)



## 6. Top 20 Commodities and States

### Top 20 Commodities by Total Volume & Distinct Reporting Weeks

| Rank | Commodity | Total Rows | Distinct Reporting Weeks (out of 591 max) | National Share |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Potato | 2,059,169 | 589 | 4.27% |
| 2 | Onion | 2,015,561 | 589 | 4.18% |
| 3 | Tomato | 1,913,317 | 589 | 3.97% |
| 4 | Wheat | 1,856,470 | 588 | 3.85% |
| 5 | Brinjal | 1,727,330 | 588 | 3.58% |
| 6 | Paddy (Dhan)(Common) | 1,581,608 | 567 | 3.28% |
| 7 | Green Chilli | 1,498,393 | 588 | 3.11% |
| 8 | Banana | 1,207,997 | 588 | 2.50% |
| 9 | Cauliflower | 1,176,456 | 588 | 2.44% |
| 10 | Bhindi (Ladies Finger) | 1,137,886 | 588 | 2.36% |
| 11 | Cabbage | 1,070,631 | 588 | 2.22% |
| 12 | Cucumbar (Kheera) | 1,030,662 | 589 | 2.14% |
| 13 | Bengal Gram (Gram)(Whole) | 973,374 | 588 | 2.02% |
| 14 | Bottle gourd | 954,092 | 588 | 1.98% |
| 15 | Rice | 921,349 | 588 | 1.91% |
| 16 | Pumpkin | 889,914 | 588 | 1.85% |
| 17 | Bitter gourd | 887,270 | 588 | 1.84% |
| 18 | Mustard | 822,353 | 588 | 1.71% |
| 19 | Maize | 822,258 | 588 | 1.70% |
| 20 | Apple | 798,298 | 588 | 1.66% |


### Top 20 States by Total Volume & Distinct Reporting Weeks

| Rank | State | Total Rows | Distinct Reporting Weeks | National Share |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Uttar Pradesh | 9,420,764 | 589 | 19.53% |
| 2 | Kerala | 5,113,657 | 589 | 10.60% |
| 3 | Tamil Nadu | 3,913,842 | 589 | 8.12% |
| 4 | Maharashtra | 3,758,533 | 589 | 7.79% |
| 5 | Punjab | 3,570,807 | 589 | 7.40% |
| 6 | Madhya Pradesh | 3,481,771 | 589 | 7.22% |
| 7 | Gujarat | 2,789,741 | 589 | 5.78% |
| 8 | Haryana | 2,210,563 | 589 | 4.58% |
| 9 | Rajasthan | 2,197,870 | 589 | 4.56% |
| 10 | Karnataka | 1,938,180 | 589 | 4.02% |
| 11 | West Bengal | 1,881,094 | 589 | 3.90% |
| 12 | Odisha | 1,446,835 | 589 | 3.00% |
| 13 | Himachal Pradesh | 1,138,472 | 589 | 2.36% |
| 14 | Telangana | 1,065,019 | 583 | 2.21% |
| 15 | Chattisgarh | 629,323 | 588 | 1.30% |
| 16 | Tripura | 574,050 | 589 | 1.19% |
| 17 | Uttrakhand | 567,924 | 567 | 1.18% |
| 18 | Andhra Pradesh | 566,504 | 573 | 1.17% |
| 19 | Jammu and Kashmir | 377,176 | 585 | 0.78% |
| 20 | NCT of Delhi | 355,166 | 589 | 0.74% |


## 7. Name Variant Candidates

Detected **29** near-duplicate spelling/casing pairs saved to `data/interim/name_fix_candidates.csv`.

Sample candidates:

| entity_type   | variant_a       | variant_b        |   similarity | issue_type           |
|:--------------|:----------------|:-----------------|-------------:|:---------------------|
| commodity     | Kutki           | kutki            |        1     | casing_or_whitespace |
| commodity     | tulip           | Tulip            |        1     | casing_or_whitespace |
| commodity     | Butter          | buttery          |        0.923 | spelling_variant     |
| commodity     | Cardamom        | Cardamoms        |        0.941 | spelling_variant     |
| commodity     | Coca            | Cocoa            |        0.889 | spelling_variant     |
| commodity     | Pepper garbled  | Pepper ungarbled |        0.933 | spelling_variant     |
| state         | Uttarakhand     | Uttrakhand       |        0.952 | spelling_variant     |
| district      | kapurthala      | Kapurthala       |        1     | casing_or_whitespace |
| district      | UdhamSinghNagar | Udhamsinghnagar  |        1     | casing_or_whitespace |
| district      | MORIGAON        | Morigaon         |        1     | casing_or_whitespace |
| district      | Bongaigaon      | BONGAIGAON       |        1     | casing_or_whitespace |
| district      | Imphal East     | Imphal West      |        0.909 | spelling_variant     |
| district      | Jhunjhunu       | Jhunjunu         |        0.941 | spelling_variant     |
| district      | Medinipur (E)   | Medinipur (W)    |        0.923 | spelling_variant     |
| district      | Rajasamand      | Rajsamand        |        0.947 | spelling_variant     |



## 8. Price Triplet Structure Analysis

Does the raw dataset contain only one price per row or a full range?

| Relationship | Row Count | Percentage | Interpretation |
| :--- | :--- | :--- | :--- |
| **Min == Modal == Max** | 4,690,880 | 9.73% | Market reported a single uniform price (no trade spread recorded) |
| **Min < Modal < Max** | 37,113,135 | 76.95% | Full price spread recorded (standard auction trading session) |
| **Min == Modal < Max** | 1,296,437 | 2.69% | Modal clearing price anchored at minimum |
| **Min < Modal == Max** | 4,630,066 | 9.60% | Modal clearing price anchored at maximum |
| **Modal outside [Min, Max]** | 498,989 | 1.0346% | Data entry inversion/error (must be dropped or cleaned) |


### Inferences & Limitations:

1. **What we can infer:** In ~9.73% of records, `Min == Modal == Max`, meaning the mandi reported a single uniform clearing price rather than an auction band. In ~76.95% of records, a genuine trading band is reported where `Min_Price < Modal_Price < Max_Price`. In ~12.29% of records, the modal price is anchored at either the minimum (2.69%) or maximum (9.60%). Finally, in 1.03% of records (498,989 rows), `Modal_Price` falls outside the `[Min, Max]` interval, indicating data entry error or inverted price columns.

2. **What we cannot infer:** We cannot infer trading volume or arrivals because no quantity/arrival column exists in the dataset.
