import duckdb

con = duckdb.connect()
df = con.execute("""
    WITH test_series AS (
        SELECT * FROM (VALUES
            ('M1', 'C1', '2023-01-01'::DATE, 100.0),
            ('M1', 'C1', '2023-01-02'::DATE, 102.0),
            ('M1', 'C1', '2023-01-03'::DATE, 98.0),
            ('M1', 'C1', '2023-01-04'::DATE, 105.0),
            ('M1', 'C1', '2023-01-05'::DATE, 1000.0), -- 10x outlier
            ('M1', 'C1', '2023-01-06'::DATE, 99.0),
            ('M1', 'C1', '2023-01-07'::DATE, 101.0),
            ('M1', 'C1', '2023-01-08'::DATE, 103.0)
        ) as t(market, commodity, date, price)
    )
    SELECT 
        market, commodity, date, price,
        MEDIAN(price) OVER (PARTITION BY market, commodity) as series_median,
        MEDIAN(price) OVER (PARTITION BY market, commodity ORDER BY date ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) as rolling_median
    FROM test_series;
""").df()
print(df)
