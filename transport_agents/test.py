# Test what's actually in your CSV
import pandas as pd
df = pd.read_csv("Airports1.csv")
print("Cities containing 'paris':", df[df['City'].str.contains('paris', case=False, na=False)]['City'].tolist())
print("Cities containing 'london':", df[df['City'].str.contains('london', case=False, na=False)]['City'].tolist())