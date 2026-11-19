import duckdb
from pathlib import Path

con = duckdb.connect()
print("DuckDB version:", duckdb.__version__)

# Test reading one row from 2015.csv
df = con.execute("SELECT * FROM 'data/raw/2015.csv' LIMIT 5").df()
print("Columns:", list(df.columns))
print("Sample rows:")
print(df)
