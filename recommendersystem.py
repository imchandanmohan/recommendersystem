import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import faiss
import numpy as np

# Load the FAISS index
index = faiss.read_index('airbnb_recommender.index')
print(f"Loaded index has {index.ntotal} vectors with {index.d} dimensions")

# Load the top 50 words and their frequencies from the CSV file
top_50_df = pd.read_csv('top_50_words_with_frequencies.csv')

# Load the new dataset
df = pd.read_csv('new_dataset.csv')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert integer columns to appropriate types (if needed)
int_columns = ['host_id', 'price', 'minimum_nights', 'number_of_reviews', 
               'calculated_host_listings_count', 'availability_365']
for col in int_columns:
    df[col] = df[col].astype(int)

# Extract the words in the same order as they were saved
top_50_words = top_50_df['word'].tolist()

# Create a DataFrame with the top 50 features
vectorizer = CountVectorizer(vocabulary=top_50_words, stop_words='english')

# Transform the 'name' column into a Bag of Words matrix (keeps the order)
new_data_bow = vectorizer.transform(df['name'])

# Convert the BoW matrix to a DataFrame
new_data_bow_df = pd.DataFrame(new_data_bow.toarray(), columns=top_50_words)

# Combine the additional features (e.g., price, room_type) with the BoW features
new_data_processed = pd.concat([df.reset_index(drop=True), new_data_bow_df], axis=1)

# Drop the 'name' column and unwanted features
new_data_processed.drop(columns=['name'], inplace=True)
processed_df = new_data_processed.drop(columns=['host_name', 'host_id', 'id', 'minimum_nights', 'last_review', 
                                                'reviews_per_month', 'calculated_host_listings_count', 
                                                'availability_365'])

# Prepare the data for FAISS (only numeric and one-hot encoded columns)
data = processed_df.select_dtypes(include=[np.number]).values

# Normalize the data for cosine similarity (inner product)
data_normalized = data / np.linalg.norm(data, axis=1, keepdims=True)

# Function to recommend similar listings
def recommend(new_data, k=5):
    """
    Given a new listing, recommend k similar listings.
    
    Args:
        new_data (numpy array): The feature vector for the new listing.
        k (int): Number of recommendations.
    
    Returns:
        indices (list): Indices of the top k similar listings.
        distances (list): Similarity scores of the top k listings.
    """
    # Normalize the new data
    new_data_normalized = new_data / np.linalg.norm(new_data, axis=1, keepdims=True)
    
    # Perform a search
    distances, indices = index.search(new_data_normalized, k)
    return indices, distances

# Example: Recommend for the first listing in the dataset
new_listing = data_normalized[0].reshape(1, -1)
recommended_indices, recommended_distances = recommend(new_listing)

# Output the recommendations
# print("Recommended Indices:", recommended_indices[0])
# print("Recommended Distances:", recommended_distances[0])

df = pd.read_csv('AB_NYC_2019.csv')
recommended_rows = df.iloc[recommended_indices[0]]

# Display the recommended rows
print(f"Recommended Rows:{recommended_rows}")



