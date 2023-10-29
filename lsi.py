# Import necessary libraries
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups

# Step 1: Data Collection
newsgroups_data = fetch_20newsgroups(subset='all')
documents = newsgroups_data.data

# Step 2: Text Preprocessing
# Tokenization, lowercase conversion, stop word removal, and stemming
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = text.split()
    tokens = [token.lower() for token in tokens]
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Step 3: Create Term-Document Matrix
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Step 4: SVD Decomposition
num_topics = 100  # Number of topics to reduce the matrix to
svd = TruncatedSVD(n_components=num_topics)
lsa_matrix = svd.fit_transform(tfidf_matrix)

# Step 5: Topic Exploration
# Singular vectors and their corresponding terms
singular_vectors = svd.components_

# Print the top terms for each topic
for topic_idx, topic in enumerate(singular_vectors):
    top_terms_idx = topic.argsort()[::-1][:10]  # Get the indices of top terms
    top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_terms)}")

# Additional analysis or visualization can be performed based on the topics extracted.


# Step 6: Load Query from a Text Document
with open("query.txt", "r") as query_file:
    query_text = query_file.read()

# Preprocess the query in the same way as the dataset
preprocessed_query = preprocess_text(query_text)

# Step 7: Project the Query into LSI Space
query_vector = vectorizer.transform([preprocessed_query])  # Transform the query into TF-IDF space
query_lsi = svd.transform(query_vector)  # Project the query into the LSI space

# Step 8: Compute Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity between the query and LSI-transformed documents
similarities = cosine_similarity(query_lsi, lsa_matrix)

# Find the most relevant documents
top_n = 5  # You can change this to the number of top relevant documents you want
top_document_indices = similarities.argsort()[0][::-1][:top_n]

# Print the most relevant documents
print("Top Relevant Documents:")
for i, doc_idx in enumerate(top_document_indices):
    print(f"{i + 1}. Document {doc_idx + 1}: {documents[doc_idx]}")


from sklearn.datasets import fetch_20newsgroups

# Load the 20 Newsgroups dataset with predefined categories
newsgroups_data = fetch_20newsgroups(subset='all')

# Ground truth labels are stored in newsgroups_data.target
ground_truth_labels = newsgroups_data.target

# If you want to see the unique categories (topics) in the dataset, you can use:
unique_categories = list(newsgroups_data.target_names)
print("Unique Categories (Topics):", unique_categories)

from sklearn.cluster import KMeans

num_clusters = 20  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(lsa_matrix)

from sklearn.metrics import accuracy_score, confusion_matrix

# Compute the confusion matrix between ground truth labels and cluster labels
confusion = confusion_matrix(ground_truth_labels, cluster_labels)
print(confusion)

# Calculate purity
purity = np.sum(np.max(confusion, axis=0)) / np.sum(confusion)
print(f"purity: {purity}")

from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(ground_truth_labels, cluster_labels)
print(f"NMI score: {nmi}")


from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(lsa_matrix, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")
