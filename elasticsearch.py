import gensim
from elasticsearch import Elasticsearch
from gensim.models import FastText
from sklearn.datasets import fetch_20newsgroups

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Preprocess the documents
preprocessed_docs = []
for doc in newsgroups.data:
    # Tokenize the document
    tokens = gensim.utils.simple_preprocess(doc.lower())
    # Remove stop words and stem the tokens
    stemmed_tokens = [gensim.parsing.porter.PorterStemmer().stem(token) for token in tokens if token not in gensim.parsing.preprocessing.STOPWORDS]
    # Join the stemmed tokens back into a string
    preprocessed_doc = ' '.join(stemmed_tokens)
    preprocessed_docs.append(preprocessed_doc)

# Train the FastText model
model = FastText(preprocessed_docs, vector_size=300, window=5, min_count=5, workers=4)

# Initialize Elasticsearch client with URL
es = Elasticsearch(['http://localhost:9200'])

# Delete the index if it already exists
index_name = 'my_index'
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create index with appropriate mappings
index_mappings = {
    'mappings': {
        'properties': {
            'text': {
                'type': 'text'
            },
            'vector': {
                'type': 'dense_vector',
                'dims': 300
            }
        }
    }
}
es.indices.create(index=index_name, body=index_mappings)


# from gensim.models import Word2Vec
import numpy as np

# Load the pre-trained Word2Vec model
# word2vec_model = Word2Vec.load("word2vec_model.bin")
word2vec_model = model

# Example user query
user_query = "computer"

# Tokenize the query
query_tokens = user_query.lower().split()

# Initialize an empty vector for the query
query_vector = np.zeros(word2vec_model.vector_size)

# Iterate through query tokens and create a vector representation
for token in query_tokens:
    if token in word2vec_model.wv:
        query_vector += word2vec_model.wv[token]

# Normalize the query vector
query_vector /= len(query_tokens)

# Now, 'query_vector' contains the vector representation of the user's query
print(query_vector)


from elasticsearch import Elasticsearch

# Initialize Elasticsearch client with URL
es = Elasticsearch(['http://localhost:9200'])

# Index name where documents are stored (change to your index name)
index_name = 'my_index'

# Create a cosine similarity query
cosine_similarity_query = {
    'query': {
        'script_score': {
            'query': {
                'match_all': {}  # You can use a more specific query here if needed
            },
            'script': {
                'source': 'cosineSimilarity(params.query_vector, "vector") + 1.0',
                'params': {
                    'query_vector': query_vector.tolist()
                }
            }
        }
    },
    'size': 10  # Adjust the number of results you want
}

# Search for similar documents
response = es.search(index=index_name, body=cosine_similarity_query)

# Print the top 10 most similar documents
for hit in response['hits']['hits']:
    print(f"Score: {hit['_score']}, Text: {hit['_source']['text']}")


from elasticsearch import Elasticsearch

# Initialize Elasticsearch client with URL
es = Elasticsearch(['http://localhost:9200'])

# Index name where documents are stored (change to your index name)
index_name = 'my_index'

# Example user query
user_query = "computer"

# Create a BM25 query
bm25_query = {
    'query': {
        'match': {
            'text': user_query
        }
    },
    'size': 10  # Adjust the number of results you want
}

# Search for documents using BM25
bm25_response = es.search(index=index_name, body=bm25_query)
print(f"response: {bm25_response}")

# Print the top 10 documents retrieved using BM25
print(f"Results from BM25 query:")
for hit in bm25_response['hits']['hits']:
    print(f"Score: {hit['_score']}, Text: {hit['_source']['text']}")


from elasticsearch import Elasticsearch
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict

# Initialize Elasticsearch client with URL
es = Elasticsearch(['http://localhost:9200'])

# Define your index name and queries
index_name = 'my_index'
queries = ["relief", "subject", "world"] 

# Function to retrieve documents for a query
def retrieve_documents(query, top_k=10):
    search_body = {
        'query': {
            'match': {
                'text': query
            }
        },
        'size': top_k
    }
    response = es.search(index=index_name, body=search_body)
    return [hit['_source']['text'] for hit in response['hits']['hits']]

# Function to evaluate a query
def evaluate_query(query, relevant_documents, top_k=10):
    retrieved_documents = retrieve_documents(query, top_k)
    precision = len(set(retrieved_documents) & set(relevant_documents)) / top_k
    recall = len(set(retrieved_documents) & set(relevant_documents)) / len(relevant_documents)
    return precision, recall

# Prepare relevance judgments (manually or from a dataset)
relevance_judgments = {
    "query 1": ["smashman leland stanford edu adam samuel nash subject organ dsg stanford univers ca usa line light letter lisa thought start new iivx hear machin predat main line mo obsolet tech rumor sold iivx owner panti bunch joke tire repetit natur type dialog plu flame stress relief", "relevant_doc_2"],
    "query 2": ["jean andrew ja andrew cmu edu subject want larg dog cage organ graduat school industri administr carnegi mellon pittsburgh pa line nntp post host po andrew cmu edu need larg dog cage kind us housebreak dog adopt month old non housetrain huskei pittsburgh area number", "paul hsh com paul havemann subject clinton immun program distribut usa organ hsh associ line articl fjsl ncratl atlantaga ncr com mwilson ncratl atlantaga ncr com mark wilson write new night clinton bash republican stonewal call stimulu packag small item packag go pai free immun poor kid clinton claim republican hold health poor kid hostag blatantli polit gain asid merit lack thereof free immun program program supposedli creat job job hell job touchi feeli program new vapid administr fact major claim univers immun children immun absolut valid state program program result averag success rate better nation averag gummint hasn figur wai parent bring kid case shameless demagogueri new democrat agent chang clinton hot immun program democrat introduc stand isn possibl clinton blatant polit read pork manipul tell republican pass muti billion dollar packag peopl tell oppos immun poor kid clinton issu gain_ tell thought highli clinton stunt like lower opinion thought clinton campaign theme go new kind politician kind manuev lbj proud mon know word meet new boss old boss choru won fool paul havemann internet paul hsh com opinion caffein brain milligram cynic observ recommend minimum daili requir mg read"],
    "query 3": ["relevant_doc_1", "relevant_doc_4"],
    # Add more queries and relevant documents as needed
}

# Initialize dictionaries to store evaluation results
precision_results = defaultdict(list)
recall_results = defaultdict(list)

# Evaluate each query
for query in queries:
    if query in relevance_judgments:
        relevant_documents = relevance_judgments[query]
        precision, recall = evaluate_query(query, relevant_documents)
        precision_results[query].append(precision)
        recall_results[query].append(recall)

# Calculate and print average precision and recall for each query
for query in queries:
    if query in precision_results:
        avg_precision = sum(precision_results[query]) / len(precision_results[query])
        avg_recall = sum(recall_results[query]) / len(recall_results[query])
        print(f"Query: {query}")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print("\n")

# Calculate and print overall average precision and recall
total_precision = sum([sum(precision_results[query]) for query in queries])
total_recall = sum([sum(recall_results[query]) for query in queries])
print("Overall Average Precision:", total_precision / len(queries))
print("Overall Average Recall:", total_recall / len(queries))


#heat map
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# get random 20 documents
documents = np.random.choice(newsgroups.data, size=30, replace=False)

# Preprocess the documents (TF-IDF vectorization)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Calculate document similarity (cosine similarity)
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=range(1, 31), yticklabels=range(1, 31))
plt.title("Document Similarity Heatmap (First 30 Documents)")
plt.show()
