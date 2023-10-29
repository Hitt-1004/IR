import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestCentroid
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = newsgroups.data
y = newsgroups.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rocchio_classifier = NearestCentroid()
rocchio_classifier.fit(X_train_tfidf, y_train)
rocchio_predictions = rocchio_classifier.predict(X_test_tfidf)

rocchio_accuracy = accuracy_score(y_test, rocchio_predictions)
print("Rocchio Classifier Accuracy:", rocchio_accuracy)
print(classification_report(y_test, rocchio_predictions))

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)
nb_predictions = nb_classifier.predict(X_test_tfidf)

nb_accuracy = accuracy_score(y_test, nb_predictions)
print("Naive Bayes Classifier Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_predictions))

knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
knn_classifier.fit(X_train_tfidf, y_train)
knn_predictions = knn_classifier.predict(X_test_tfidf)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print("k-Nearest Neighbor Classifier Accuracy:", knn_accuracy)
print(classification_report(y_test, knn_predictions))
