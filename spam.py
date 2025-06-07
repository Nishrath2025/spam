from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
texts = [
    "Win a free iPhone now",
    "Important meeting tomorrow",
    "Congratulations, you've won a lottery",
    "Call your mom",
    "Free entry in 2 million contest",
    "Let's have lunch together",
    "Buy now and get 50% off",
    "Your appointment is confirmed"
]

labels = ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
