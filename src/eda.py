import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load cleaned data
file_path = 'data/heart.csv'

df = pd.read_csv(file_path)

print("✅ Loaded data:", df.shape)

os.makedirs('eda visualizations', exist_ok=True)

# 2. View target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.xlabel("Heart Disease Severity (0 = no disease)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('eda visualizations/target_distribution.png')
plt.close()

# 3. Plot numerical feature histograms
num_cols = df.select_dtypes(include='number').columns.tolist()
df[num_cols].hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout()
plt.savefig('eda visualizations/histograms.png')
plt.close()

# 4. Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('eda visualizations/correlation_heatmap.png')
plt.close()

# 5. Categorical Feature: Sex vs. Target
plt.figure(figsize=(6, 4))
sns.countplot(x='sex', hue='target', data=df)
plt.title("Heart Disease by Sex (0 = Female, 1 = Male)")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Target", loc='upper right')
plt.tight_layout()
plt.savefig('eda visualizations/sex_vs_target.png')
plt.close()

# 6. Numeric Feature: Age distribution by Target
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='age', data=df)
plt.title("Age Distribution by Heart Disease Severity")
plt.xlabel("Heart Disease Severity (0 = no disease)")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig('eda visualizations/age_vs_target_boxplot.png')
plt.close()
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='age', data=df)

# ✏️ Add title and labels
plt.title("Age Distribution by Heart Disease Severity")
plt.xlabel("Heart Disease Severity (0 = no disease)")
plt.ylabel("Age")

# ✏️ Add mean age labels above each box
mean_ages = df.groupby('target')['age'].mean()
for target_level, mean_age in mean_ages.items():
    plt.text(
        x=target_level,
        y=mean_age + 1.5,  # place label slightly above the mean
        s=f"Mean: {mean_age:.1f}",
        ha='center',
        fontsize=9,
        color='darkblue'
    )
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.figure(figsize=(10, 6))

#7 scatterplot with oldpeak and trestbps vs. target
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 1. Select features
x_col = 'trestbps'
y_col = 'oldpeak'

# 2. Set up figure
plt.figure(figsize=(10, 6))

# 3. Color scatterplot by heart disease severity
scatter = sns.scatterplot(
    x=x_col,
    y=y_col,
    hue='target',
    palette='coolwarm',
    data=df,
    s=80,
    edgecolor='white'
)

# 4. Optional: Add cluster centers
X = df[[x_col, y_col]].dropna()
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)
centers = kmeans.cluster_centers_

plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, marker='X', label='Cluster Center')

# 5. Title and labels
plt.title('Oldpeak vs Resting BP by Heart Disease Severity')
plt.xlabel('Resting Blood Pressure (trestbps)')
plt.ylabel('ST Depression (oldpeak)')
plt.legend(title='Heart Disease Severity', loc='upper right')
plt.tight_layout()

# 6. Save or show
plt.savefig('eda visualizations/oldpeak_vs_trestbps.png')
plt.show()

#8 strip plot ca and chol vs. target

df['ca'] = df['ca'].astype(str)
plt.figure(figsize=(10, 6))
sns.stripplot(x='ca', y='chol', hue='target', data=df, dodge=True, jitter=True, palette='coolwarm')

plt.title("Serum Cholesterol by Number of Vessels (ca) and Heart Disease Severity")
plt.xlabel("Number of Major Vessels (ca)")
plt.ylabel("Cholesterol (mg/dL)")
plt.legend(title="Heart Disease Severity", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("eda visualizations/ca_vs_chol_by_target.png")
plt.show()

print("✅ All charts saved to /eda visualizations")

# logistic_model 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

# 1. Load cleaned data
file_path = 'data/heart.csv'
df = pd.read_csv(file_path)
print("\n✅ Loaded data:", df.shape)

# 2. Separate features and target
X = df.drop(columns='target')
y = df['target']

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Create and train the model using 'saga' solver
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# 8. Feature importance (approximate after scaling)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("\n✅ Feature Influence (higher = more impact):\n")
print(coefficients)

print("""
📏 Scaling Applied:
- StandardScaler was used to scale all features so the model treats them fairly.
- Solver 'saga' is now used — it works better for scaled and multiclass data.
✅ Model should now learn more effectively and converge without warnings.
""")