import os
import email
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define a function to preprocess email text.
def preprocess_text(text):
    # Parse the email content.
    msg = email.message_from_string(text)
    
    # Extract the subject and body of the email.
    subject = msg['Subject'] if msg['Subject'] else ''
    body = ''
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition:
                body += part.get_payload()
    else:
        body = msg.get_payload()

    # Combine the subject and body into one text.
    email_text = subject + ' ' + body
    
    # Convert to lowercase, remove punctuation, and tokenize.
    email_text = email_text.lower()
    email_text = " ".join(email_text.split())  # Remove extra whitespace
    email_text = " ".join(email_text.splitlines())  # Remove line breaks
    
    # You can further preprocess the text here (e.g., stemming, removing stopwords).
    
    return email_text

# Apply the preprocessing function to each email.
X = X.apply(preprocess_text)

# Step 2: Convert Text Data to TF-IDF Features

# Create a TF-IDF vectorizer to convert text data to numerical features.
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the number of features as needed.

# Load your dataset.
data = pd.read_csv('emails.csv')

# Exclude the 'Email No.' column, assuming it contains non-numeric values.
data = data.drop('Email No.', axis=1)

# Separate the features (word frequencies) and the target variable (Prediction).
X = data.drop('Prediction', axis=1)  # Features (word frequencies)
y = data['Prediction']  # Target variable (1 for spam, 0 for not spam)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a Classifier (e.g., Naive Bayes)
classifier = MultinomialNB()

# Train the Classifier
classifier.fit(X_train, y_train)

# Make Predictions
y_pred = classifier.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)
#1 for spam, 0 for not spam
