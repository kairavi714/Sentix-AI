import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sys

print("1. Loading dataset...")
# Try reading it normally first
try:
    df = pd.read_csv('dataset.csv', encoding='latin-1')
    
    # If the dataset has exactly 6 unnamed columns, it is the classic Twitter dataset
    if len(df.columns) == 6 and 'text' not in df.columns:
        print("-> Detected raw Twitter dataset format.")
        df = pd.read_csv('dataset.csv', encoding='latin-1', header=None, usecols=[0, 5])
        df.columns = ['sentiment', 'text']
    else:
        # Otherwise, find the text and sentiment columns automatically
        text_col = next((c for c in df.columns if c.lower() in ['text', 'tweet', 'comment']), None)
        sentiment_col = next((c for c in df.columns if c.lower() in ['sentiment', 'label', 'target']), None)
        
        if text_col and sentiment_col:
            df = df[[sentiment_col, text_col]]
            df.columns = ['sentiment', 'text']
        else:
            print(f"Error: Could not find columns. Available columns: {df.columns.tolist()}")
            sys.exit()
            
except Exception as e:
    print(f"Failed to read CSV: {e}")
    sys.exit()

# Drop completely empty rows
df = df.dropna()

print(f"2. Raw labels found in your dataset: {df['sentiment'].unique().tolist()}")

# Convert whatever the labels are (numbers or words) into our standard UI labels
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()

universal_map = {
    '0': 'Negative', '0.0': 'Negative', 'negative': 'Negative', 'neg': 'Negative',
    '1': 'Neutral', '1.0': 'Neutral', 'neutral': 'Neutral', 'neu': 'Neutral',
    '2': 'Neutral', '2.0': 'Neutral', # Some datasets use 2 for neutral
    '4': 'Positive', '4.0': 'Positive', 'positive': 'Positive', 'pos': 'Positive',
    '3': 'Positive', '3.0': 'Positive'
}

df['sentiment'] = df['sentiment'].map(universal_map)
df = df.dropna(subset=['sentiment'])

print(f"3. Rows ready for training: {len(df)}")
if len(df) == 0:
    print("ERROR: All rows were dropped. The labels didn't match.")
    sys.exit()

# If the dataset is massive, sample 100k rows so you don't wait 10 minutes
if len(df) > 100000:
    print("-> Massive dataset detected. Sampling 100,000 rows for fast training...")
    df = df.sample(n=100000, random_state=42)

print("4. Converting words to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

print("5. Training the Machine Learning Model (Logistic Regression)...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("6. Saving the brain to disk...")
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(model, 'model.pkl')

print("✅ SUCCESS! The model.pkl and vectorizer.pkl files are ready.")