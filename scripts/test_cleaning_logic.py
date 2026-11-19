import duckdb
import pandas as pd

con = duckdb.connect()

# Create synthetic data with duplicates, outliers, flat runs, bad dates
con.execute("""
    CREATE TABLE test_data AS
    SELECT * FROM (VALUES
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-01', 1000.0),
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-01', 1200.0), -- duplicate with conflict (median=1100)
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-02', 1050.0),
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-03', 100000.0), -- 100x outlier
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-04', 1020.0),
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', 'bad-date', 1020.0), -- bad date
        ('UP', 'Agra', 'Agra APMC', 'Potato', 'Desi', 'FAQ', '2023-01-05', -10.0) -- non-positive
    ) as t(state, district, market, commodity, variety, grade, date_raw, price);
""")

# Test parsing dates and non-positive prices
parsed = con.execute("""
    SELECT 
        trim(regexp_replace(state, '\\s+', ' ', 'g')) as state,
        trim(regexp_replace(district, '\\s+', ' ', 'g')) as district,
        trim(regexp_replace(market, '\\s+', ' ', 'g')) as market,
        trim(regexp_replace(commodity, '\\s+', ' ', 'g')) as commodity,
        trim(regexp_replace(variety, '\\s+', ' ', 'g')) as variety,
        trim(regexp_replace(grade, '\\s+', ' ', 'g')) as grade,
        try_strptime(date_raw, ['%Y-%m-%d', '%d/%m/%Y'])::DATE as date,
        price
    FROM test_data
    WHERE try_strptime(date_raw, ['%Y-%m-%d', '%d/%m/%Y']) IS NOT NULL
      AND price > 0
""").df()
print("Parsed valid rows:")
print(parsed)

# Test deduplication with median resolution
con.execute("CREATE TABLE valid_rows AS SELECT * FROM parsed;")
deduped = con.execute("""
    SELECT 
        state, district, market, commodity, variety, grade, date,
        median(price) as price,
        count(*) as duplicate_count
    FROM valid_rows
    GROUP BY state, district, market, commodity, variety, grade, date
""").df()
print("\nDeduped with median:")
print(deduped)

# Test run identification
runs = con.execute("""
    WITH flat_data AS (
        SELECT * FROM (VALUES
            ('M1', 'C1', '2023-01-01'::DATE, 100.0),
            ('M1', 'C1', '2023-01-02'::DATE, 100.0),
            ('M1', 'C1', '2023-01-03'::DATE, 100.0),
            ('M1', 'C1', '2023-01-04'::DATE, 200.0),
            ('M1', 'C1', '2023-01-05'::DATE, 200.0)
        ) as t(market, commodity, date, price)
    ),
    diffs AS (
        SELECT 
            market, commodity, date, price,
            CASE WHEN price = lag(price, 1) OVER (PARTITION BY market, commodity ORDER BY date) THEN 0 ELSE 1 END as is_new_run
        FROM flat_data
    ),
    run_groups AS (
        SELECT 
            market, commodity, date, price,
            sum(is_new_run) OVER (PARTITION BY market, commodity ORDER BY date) as run_id
        FROM diffs
    )
    SELECT 
        market, commodity, date, price, run_id,
        count(*) OVER (PARTITION BY market, commodity, run_id) as run_len
    FROM run_groups
""").df()
print("\nFlat run lengths:")
print(runs)
