import pandas as pd
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

# Separate labels
x = df['text']
y = df['label']

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_vectorized = vectorizer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train, y_train)

# Accuracy
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print("Accuracy:", score)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully")