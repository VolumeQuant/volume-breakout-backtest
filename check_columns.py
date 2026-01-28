import pandas as pd

# Read the flow data
df = pd.read_csv('data/investor_flow_data.csv', encoding='utf-8-sig', nrows=1)

print("Column names:")
for i, col in enumerate(df.columns):
    print(f"{i}: {repr(col)}")

print(f"\nTotal: {len(df.columns)} columns")
