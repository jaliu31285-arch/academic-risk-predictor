import pandas as pd

df = pd.read_csv('data/College_Students_Academic_Performance_Dataset.csv')
print('Shape: rows=', len(df), ' cols=', len(df.columns))
print()

for i, c in enumerate(df.columns):
    dtype = str(df[c].dtype)
    nuniq = df[c].nunique()
    nmiss = df[c].isnull().sum()
    if pd.api.types.is_numeric_dtype(df[c]):
        print('  [{:2d}] {:35s} type={:10s} unique={:4d} missing={:3d} min={:.3f} mean={:.3f} max={:.3f}'.format(
            i+1, c, dtype, nuniq, nmiss, df[c].min(), df[c].mean(), df[c].max()))
    else:
        top = df[c].value_counts().head(6).to_dict()
        print('  [{:2d}] {:35s} type={:10s} unique={:4d} missing={:3d} values={}'.format(
            i+1, c, dtype, nuniq, nmiss, top))

print()
print('First 2 rows (transposed):')
print(df.head(2).T.to_string())

print()
print('Target-like columns:')
for c in df.columns:
    low = c.lower()
    if any(k in low for k in ['risk', 'score', 'class', 'label', 'grade', 'gpa', 'performance']):
        print('  {}: unique={} dtype={}'.format(c, df[c].nunique(), df[c].dtype))
        if df[c].dtype == 'object':
            print('    ', df[c].value_counts().to_dict())
        else:
            print('    min={:.2f} max={:.2f} mean={:.2f}'.format(df[c].min(), df[c].max(), df[c].mean()))
