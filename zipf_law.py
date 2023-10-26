import nltk
import matplotlib.pyplot as plt
import math
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from wordcloud import WordCloud
import numpy as np
from sklearn.linear_model import LinearRegression

nltk.download('stopwords')
nltk.download('punkt')

with open('gutenberg.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
words = word_tokenize(text)

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]

# Stemming or Lemmatization (choose one)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
# lemmatizer = nltk.WordNetLemmatizer()
# lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

# Basic word statistics
freq_dist = FreqDist(stemmed_words)
top_words = freq_dist.most_common(50)

# Plot a histogram of the 50 most frequent words
words, frequencies = zip(*top_words)
plt.figure(figsize=(12, 6))
plt.bar(words, frequencies)
plt.xticks(rotation=90)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 50 Most Frequent Words')
plt.show()

# Create a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dist)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud')
plt.show()

# Validate Zipf's Law
rank = range(1, len(freq_dist) + 1)
frequencies = [freq for word, freq in top_words]
# frequencies = [freq for word, freq in freq_dist.items()]

top_rank = rank[:50]
top_frequencies = frequencies[:50]

# Fit a linear regression model
x = np.log(top_rank)
y = np.log(top_frequencies)
X = np.array(x).reshape(-1, 1)
Y = np.array(y)
model = LinearRegression()
model.fit(X, Y)

# Get the Zipf's law parameters
alpha = -model.coef_[0]
C = np.exp(model.intercept_)

# Print Zipf's law parameters
print(f'Zipf\'s Law Exponent (alpha): {alpha:.4f}')
print(f'Corpus Constant (C): {C:.4f}')

# Plot Zipf's Law
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'bo', label='Data')
plt.plot(x, model.predict(X), 'r-', label=f'Fit (alpha={alpha:.4f}, C={C:.4f})')
plt.xlabel('log(rank)')
plt.ylabel('log(frequency)')
plt.title('Zipf\'s Law Validation')
plt.legend()
plt.show()
