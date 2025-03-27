import pandas as pd
import os

# 1. Set the file path to your downloaded UCI file
file_path = 'data/processed.cleveland.data'

# 2. Column names from the UCI documentation
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# 3. Load the .data file correctly (space-separated, some missing)
df = pd.read_csv(
    file_path,
    names=columns,
    na_values='?',
    sep=',',              # <-- YES, it’s actually comma-separated
    engine='python'
)

# 4. Drop any rows where target is missing or invalid
df['target'] = pd.to_numeric(df['target'], errors='coerce')
df = df[df['target'].isin([0, 1, 2, 3, 4])]

# 5. Drop other missing rows
df = df.dropna()

# 6. Convert selected columns to category
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    df[col] = df[col].astype('category')

# 7. 
print(df["exang"].head(5))
print(df["oldpeak"].head(10))

# 7. Save to a cleaned CSV
output_path = 'data/heart.csv'
df.to_csv(output_path, index=False)

# 8. Print final confirmation
print("✅ Final cleaned shape:", df.shape)
print("✅ Target values:", df['target'].unique())
print("📁 File saved to:", os.path.abspath(output_path))
