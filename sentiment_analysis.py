import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# data processing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', '', text)
    return text

df = pd.read_csv("data.csv")

# remove rows with empty text
df = df.dropna(subset=[df.columns[3]])

df['cleaned'] = df.iloc[:, 3].apply(clean_text)
y = df.iloc[:, 2]  # Sentiment in column 2 ('Positive' etc.)

# convert text to numbers (Vectorization) - using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned']).toarray()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model training - Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# model evaluation (good accuracy: 80-90%)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# test with custom input
def predict_sentiment(text):
    text = clean_text(text)
    vectorized = vectorizer.transform([text]).toarray()
    return model.predict(vectorized)[0]

print(predict_sentiment("This product is amazing"))

# save the model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))