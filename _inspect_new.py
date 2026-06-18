import pandas as pd
import sys

df = pd.read_csv('data/College_Students_Academic_Performance_Dataset.csv')
print('Shape:', df.shape)
print('\nColumns:')
for i, c in enumerate(df.columns):
    dtype = str(df[c].dtype)
    nuniq = df[c].nunique()
    if df[c].dtype == 'object':
        vals = df[c].value_counts().head(5).to_dict()
        print(f'  [{i+1:2d}] {c:35s} type={dtype:10s} unique={nuniq:3d}  top_values: {vals}')
    else:
        print(f'  [{i+1:2d}] {c:35s} type={dtype:10s} unique={nuniq:3d}  min={df[c].min():.2f} mean={df[c].mean():.2f} max={df[c].max():.2f}')

print('\nFirst 3 rows:')
print(df.head(3).to_string())
print('\nMissing values per column:')
print(df.isnull().sum())
