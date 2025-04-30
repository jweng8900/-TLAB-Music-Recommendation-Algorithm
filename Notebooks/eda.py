# 01_eda.ipynb

# EDA Notebook for Music Recommendation System

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Data/train.csv")

# Preview the data
df.head()

# Drop 'Unnamed: 0' if present
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Basic info
df.info()

# Check missing values
missing = df.isnull().sum()
print("Missing values:\n", missing[missing > 0])

# Plot distribution of genres
plt.figure(figsize=(12,6))
df['genre'].value_counts().plot(kind='bar', title='Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot release date distribution
plt.figure(figsize=(12,6))
df['release_date'].astype(str).value_counts().sort_index().plot(kind='bar', title='Release Date Distribution')
plt.xlabel('Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation heatmap of numeric features
numeric_cols = df.select_dtypes(include='number').columns
plt.figure(figsize=(16,10))
sns.heatmap(df[numeric_cols].corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Relationship between lyrics length and sadness
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='len', y='sadness')
plt.title('Lyrics Length vs. Sadness')
plt.xlabel('Length of Lyrics')
plt.ylabel('Sadness Score')
plt.tight_layout()
plt.show()
