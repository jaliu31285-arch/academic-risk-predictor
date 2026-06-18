import pandas as pd
df = pd.read_csv('data/college_students_academic_performance_dataset.csv')
exclude = {'Student_ID', 'Academic_Risk_Level', 'Intervention_Recommendation',
           'Performance_Score', 'Academic_Risk_Class', 'Final_Academic_Score'}
feats = [c for c in df.columns if c not in exclude]
print('Total columns:', len(df.columns))
print('All columns:', list(df.columns))
print()
print('Feature count:', len(feats))
for i, f in enumerate(feats):
    col = df[f]
    print(f"  [{i+1:2d}] {f:35s} min={col.min():.3f} mean={col.mean():.3f} max={col.max():.3f}")
