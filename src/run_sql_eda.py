import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

#make folder for sql visualizations
os.makedirs('queries/eda_visualizations', exist_ok=True)

# Load CSV into SQLite (if not already done)
df = pd.read_csv('data/heart.csv')
conn = sqlite3.connect('heart_disease.db')
df.to_sql('heart_data', conn, if_exists='replace', index=False)

# Read SQL file
with open('queries/heart_eda.sql', 'r') as f:
    sql_script = f.read()

# Split queries by semicolon
queries = [q.strip() for q in sql_script.split(';') if q.strip()]

# Run each query
for i, query in enumerate(queries, 1):
    print(f"\n--- Query {i} Result ---")
    result = pd.read_sql_query(query, conn)
    print(result)

# Create charts for selected queries
    if i == 3:  # Target class distribution
        result.plot(kind='bar', x='target', y='count', legend=False)
        plt.title('Heart Disease Severity Count')
        plt.xlabel('Severity (0 = No disease)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/target_distribution_sql.png')
        plt.close()

    elif i == 4:  # Gender distribution
        result.plot(kind='bar', x='gender', y='count', legend=False)
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/gender_distribution_sql.png')
        plt.close()

    elif i == 5:  # Avg cholesterol and BP by target
        result.plot(x='target', y=['avg_chol', 'avg_bp'], kind='bar')
        plt.title('Average Cholesterol and BP by Severity')
        plt.xlabel('Heart Disease Severity')
        plt.ylabel('Average Value')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/avg_chol_bp_by_target_sql.png')
        plt.close()

    elif i == 6:  # Heart disease count by sex and target
        pivot = result.pivot(index='sex', columns='target', values='count')
        pivot.plot(kind='bar', stacked=True)
        plt.title('Heart Disease Count by Sex')
        plt.xlabel('Sex (0 = Female, 1 = Male)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/disease_by_sex_sql.png')
        plt.close()

    elif i == 7:  # Avg heart rate by severity
        result.plot(x='target', y='avg_heart_rate', kind='bar', legend=False)
        plt.title('Average Heart Rate by Severity')
        plt.xlabel('Heart Disease Severity')
        plt.ylabel('Avg Heart Rate')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/avg_heart_rate_sql.png')
        plt.close()

    elif i == 8:  # Heart disease rate by cholesterol range
        result.plot(x='chol_range', y='disease_rate_percent', kind='bar', legend=False)
        plt.title('Heart Disease Rate by Cholesterol Range')
        plt.xlabel('Cholesterol Range')
        plt.ylabel('Disease Rate (%)')
        plt.tight_layout()
        plt.savefig('queries/eda_visualizations/cholesterol_rate_sql.png')
        plt.close()

print("\n✅ All SQL queries executed and charts saved!")
