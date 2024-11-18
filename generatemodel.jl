using CSV
using DataFrames
using LinearAlgebra

# Step 1: Load the dataset from the CSV file
data = CSV.File("data_normalized.csv") |> DataFrame

# Print the first few rows and the shape of the data to validate
println("Data preview:")
println(first(data, 5))  # Print first 5 rows
println("Shape of data: ", size(data))  # Print dimensions of the data

# Step 2: Ensure that the data contains only numeric values and no missing values
# Convert to a matrix of floats, ensuring no missing values
data_matrix = Matrix{Float64}(data)  # Convert DataFrame to matrix

# Print a preview of the matrix and its dimensions to check
println("Data Matrix preview (first 5 rows):")
println(data_matrix[1:5, :])  # Print the first 5 rows of the matrix
println("Shape of Data Matrix: ", size(data_matrix))  # Print dimensions of the matrix

# Step 3: Compute the cosine similarity between two vectors
function cosine_similarity(a, b)
    return dot(a, b) / (norm(a) * norm(b))
end

# Step 4: Select a feature (let's assume we are selecting the first feature)
selected_feature = 1

# Step 5: Calculate cosine similarity between the selected feature and all other features
similarities = []  # Initialize an empty list for storing similarities
for i in 1:size(data_matrix, 2)  # Loop through columns (features)
    if i != selected_feature
        similarity_value = cosine_similarity(data_matrix[:, selected_feature], data_matrix[:, i])
        push!(similarities, (i, similarity_value))
    end
end

# Ensure similarities are populated before proceeding
println("Number of similarities calculated: ", length(similarities))
if length(similarities) == 0
    println("No similarities found. Please check your data for issues.")
else
    # Step 6: Sort the features by similarity and recommend the top 5 most similar features
    sorted_similarities = sort(similarities, by = x -> -x[2])
    top_5_similar_features = sorted_similarities[1:5]
    
    println("Top 5 similar features to feature $selected_feature:")
    for (feature, similarity) in top_5_similar_features
        println("Feature $feature with similarity $similarity")
    end
    
    # Step 7: Load the AB_NYC_2019.csv file to fetch records based on the selected feature
    ab_nyc_data = CSV.File("AB_NYC_2019.csv") |> DataFrame
    
    # Step 8: Fetch and store rows corresponding to the most similar features
    top_5_rows = []
    for (feature, _) in top_5_similar_features
        # Ensure the feature index is within bounds
        if feature <= size(ab_nyc_data, 2)
            # Find rows based on this feature
            selected_feature_column = ab_nyc_data[:, feature]
            
            # Take the first 5 rows for simplicity
            for i in 1:min(5, size(ab_nyc_data, 1))
                push!(top_5_rows, ab_nyc_data[i, :])
            end
        else
            println("Feature index $feature is out of bounds in AB_NYC_2019.csv.")
        end
    end
    
    # Step 9: Print and save the recommendations
    println("\nTop 5 recommended rows based on similarity:")
    for (i, row) in enumerate(top_5_rows)
        println("Recommendation $i: ", row)
    end
    
    recommendations_df = DataFrame(top_5_rows)  # Convert list to DataFrame
    CSV.write("top_5_recommendations.csv", recommendations_df)  # Save to CSV
    println("\nRecommendations saved to 'top_5_recommendations.csv'.")
    
    # Step 10: Directly test a specific row
    test_row_id = 16  # Example row index for testing
    println("\nTesting with row ID: $test_row_id")
    println("Selected record: ", ab_nyc_data[test_row_id, :])  # Print the selected row
end
