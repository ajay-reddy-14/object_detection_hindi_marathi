import pandas as pd

file_path = 'benchmark_results.csv'
df = pd.read_csv(file_path)

cols_to_convert = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']

for col in cols_to_convert:
    if col in df.columns:
        df[col] = (df[col] * 100).round(2)

df.to_csv(file_path, index=False)
print("Converted to percentage successfully.")
