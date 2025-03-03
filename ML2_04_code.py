"""
Week 4: Vector Representations & Similarity Measures
Section 1: Food Preference Vector Example Implementation

This module demonstrates simple vector representations of food preferences
and how to compute similarities between them.
"""

import numpy as np

def create_binary_vector(items, liked_items):
    """
    Create a binary vector where 1 represents liked items and 0 represents others.
    
    Example:
    items = ['pizza', 'sushi', 'tacos']
    liked_items = ['pizza', 'tacos']
    returns: [1, 0, 1]
    """
    return np.array([1 if item in liked_items else 0 for item in items])

def create_rating_vector(items, ratings):
    """
    Create a vector of ratings (0-5 scale) for items.
    Missing ratings default to 0.
    """
    return np.array([ratings.get(item, 0) for item in items])

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Returns value between -1 (opposite) and 1 (identical).
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def create_sparse_vector(all_ingredients, used_ingredients):
    """
    Create a sparse vector for ingredients at a very granular level.
    Demonstrates the sparsity problem with specific ingredients.
    """
    return np.array([1 if ing in used_ingredients else 0 for ing in all_ingredients])

def create_general_vector(items, preferences, category_mapping):
    """
    Create a vector using general categories instead of specific items.
    Shows how over-generalization can mask important differences.
    """
    general_prefs = []
    for item in preferences:
        if item in category_mapping:
            general_prefs.append(category_mapping[item])
    return np.array([1 if item in general_prefs else 0 for item in items])

def create_likert_vector(questions, responses):
    """
    Create a vector from Likert scale responses (1-5 scale).
    1: Strongly Disagree, 5: Strongly Agree
    """
    return np.array([responses.get(q, 0) for q in questions])

def simple_factor_analysis(data, n_factors=2):
    """
    Simplified demonstration of factor analysis concept.
    Uses PCA as a stand-in to show dimensionality reduction.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_factors)
    return pca.fit_transform(data)

def create_simple_embedding_model(n_items, embedding_dim=2):
    """
    Create a simple neural network that learns food item embeddings.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    model = Sequential([
        Dense(embedding_dim, activation='relu', input_shape=(n_items,)),
        Dense(n_items, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def part_one():
    # Define our food items
    cuisines = ['Italian', 'Indian', 'Japanese', 'Mexican', 'Thai']
    ingredients = ['tomato', 'curry', 'wasabi', 'chili', 'lemongrass']
    all_items = cuisines + ingredients

    # User preferences
    user1_likes = ['Italian', 'Indian', 'tomato', 'curry']
    user2_likes = ['Indian', 'Thai', 'curry', 'lemongrass']

    user1_ratings = {
        'Italian': 5.0, 'Indian': 4.0, 'Japanese': 2.0,
        'tomato': 5.0, 'curry': 4.0, 'wasabi': 1.0
    }
    
    user2_ratings = {
        'Indian': 5.0, 'Thai': 4.0, 'Japanese': 3.0,
        'curry': 5.0, 'lemongrass': 4.0, 'wasabi': 3.0
    }

    # Create vectors
    user1_binary = create_binary_vector(all_items, user1_likes)
    user2_binary = create_binary_vector(all_items, user2_likes)
    
    user1_ratings = create_rating_vector(all_items, user1_ratings)
    user2_ratings = create_rating_vector(all_items, user2_ratings)

    # Calculate similarities
    binary_sim = cosine_similarity(user1_binary, user2_binary)
    rating_sim = cosine_similarity(user1_ratings, user2_ratings)

    # Print results
    print("\nBinary vectors:")
    print("User 1:", user1_binary)
    print("User 2:", user2_binary)
    print("Binary similarity:", binary_sim)

    print("\nRating vectors:")
    print("User 1:", user1_ratings)
    print("User 2:", user2_ratings)
    print("Rating similarity:", rating_sim)

def part_two():
    """
    Demonstrates the challenges of sparse representations and the curse of dimensionality.
    """
    # Create a large ingredient space (simulating high dimensionality)
    specific_ingredients = [
        # Italian ingredients
        'basil_fresh', 'basil_dried', 'oregano_fresh', 'oregano_dried',
        'tomato_roma', 'tomato_cherry', 'tomato_san_marzano',
        'parmesan_aged', 'parmesan_fresh', 'mozzarella_buffalo',
        # Indian ingredients
        'curry_madras', 'curry_vindaloo', 'turmeric_ground', 'turmeric_fresh',
        'cardamom_green', 'cardamom_black', 'cumin_whole', 'cumin_ground',
        # Japanese ingredients
        'nori_gold', 'nori_silver', 'wasabi_fresh', 'wasabi_powder',
        'soy_light', 'soy_dark', 'mirin_pure', 'mirin_seasoned'
    ]

    # Two recipes that are conceptually similar (both pasta dishes)
    recipe1_ingredients = [
        'basil_fresh', 'tomato_roma', 'parmesan_aged'
    ]
    recipe2_ingredients = [
        'basil_dried', 'tomato_san_marzano', 'parmesan_fresh'
    ]

    # Create sparse vectors
    recipe1_vector = create_sparse_vector(specific_ingredients, recipe1_ingredients)
    recipe2_vector = create_sparse_vector(specific_ingredients, recipe2_ingredients)

    # Calculate similarity
    similarity = cosine_similarity(recipe1_vector, recipe2_vector)

    # Demonstrate the sparsity problem
    print("\nDemonstrating Sparse Representation Problems:")
    print(f"Total dimensions: {len(specific_ingredients)}")
    print(f"Recipe 1 non-zero elements: {np.count_nonzero(recipe1_vector)}")
    print(f"Recipe 2 non-zero elements: {np.count_nonzero(recipe2_vector)}")
    print(f"Sparsity: {1 - np.count_nonzero(recipe1_vector)/len(recipe1_vector):.2%}")
    print(f"Cosine Similarity: {similarity}")
    
    # Demonstrate missing relationships
    print("\nMissing Relationships Example:")
    print("Even though both recipes are pasta dishes with similar ingredients,")
    print("the similarity score is low because they use different specific variants")
    print("of the same ingredients (fresh vs dried basil, different tomatoes, etc.)")

def part_three():
    """
    Demonstrates the problems with over-generalization in food preferences.
    Shows how broad categories can create false equivalences.
    """
    # Define broad categories
    general_categories = ['protein', 'vegetable', 'grain', 'spice']
    
    # Category mapping for specific ingredients
    category_mapping = {
        'chicken': 'protein',
        'tofu': 'protein',
        'salmon': 'protein',
        'broccoli': 'vegetable',
        'carrot': 'vegetable',
        'rice': 'grain',
        'quinoa': 'grain',
        'pepper': 'spice',
        'cumin': 'spice'
    }

    # Two very different diets that look similar when over-generalized
    vegan_diet = ['tofu', 'broccoli', 'quinoa', 'cumin']
    meat_diet = ['chicken', 'carrot', 'rice', 'pepper']

    # Create generalized vectors
    vegan_vector = create_general_vector(general_categories, vegan_diet, category_mapping)
    meat_vector = create_general_vector(general_categories, meat_diet, category_mapping)

    # Calculate similarity with generalized categories
    general_similarity = cosine_similarity(vegan_vector, meat_vector)

    # Show how over-generalization masks differences
    print("\nDemonstrating Over-generalization Problems:")
    print("\nGeneral Categories:", general_categories)
    print("Vegan diet (specific):", vegan_diet)
    print("Meat diet (specific):", meat_diet)
    print("\nGeneralized vectors:")
    print("Vegan:", vegan_vector)
    print("Meat:", meat_vector)
    print(f"Similarity score: {general_similarity}")
    print("\nProblem: These diets appear similar because 'protein' doesn't")
    print("distinguish between animal and plant sources, losing crucial")
    print("information about dietary preferences and restrictions.")

def part_four():
    """
    Demonstrates traditional survey design approaches and
    statistical methods for handling preference data.
    """
    # Example survey questions about food preferences
    survey_questions = [
        'q1_like_spicy',
        'q2_prefer_healthy',
        'q3_enjoy_cooking',
        'q4_try_new_foods',
        'q5_eat_meat',
        'q6_like_sweets',
        'q7_prefer_organic',
        'q8_enjoy_ethnic'
    ]

    # Sample responses from different users (1-5 Likert scale)
    user_responses = [
        {  # Adventurous health-conscious user
            'q1_like_spicy': 4,
            'q2_prefer_healthy': 5,
            'q3_enjoy_cooking': 5,
            'q4_try_new_foods': 5,
            'q5_eat_meat': 2,
            'q6_like_sweets': 2,
            'q7_prefer_organic': 5,
            'q8_enjoy_ethnic': 5
        },
        {  # Traditional comfort food user
            'q1_like_spicy': 2,
            'q2_prefer_healthy': 3,
            'q3_enjoy_cooking': 4,
            'q4_try_new_foods': 2,
            'q5_eat_meat': 5,
            'q6_like_sweets': 4,
            'q7_prefer_organic': 2,
            'q8_enjoy_ethnic': 2
        },
        {  # Balanced diet user
            'q1_like_spicy': 3,
            'q2_prefer_healthy': 4,
            'q3_enjoy_cooking': 3,
            'q4_try_new_foods': 3,
            'q5_eat_meat': 3,
            'q6_like_sweets': 3,
            'q7_prefer_organic': 4,
            'q8_enjoy_ethnic': 4
        }
    ]

    # Create vectors from survey responses
    user_vectors = []
    for responses in user_responses:
        vector = create_likert_vector(survey_questions, responses)
        user_vectors.append(vector)
    
    # Convert to numpy array for analysis
    preference_matrix = np.array(user_vectors)
    
    # Demonstrate dimension reduction
    reduced_preferences = simple_factor_analysis(preference_matrix)

    # Print results
    print("\nDemonstrating Survey Design and Analysis:")
    print("\nOriginal survey responses (8 dimensions):")
    for i, vector in enumerate(user_vectors):
        print(f"User {i+1}:", vector)
    
    print("\nReduced representations (2 dimensions):")
    for i, reduced_vec in enumerate(reduced_preferences):
        print(f"User {i+1}:", reduced_vec)
        
    print("\nObservation: The reduced representations maintain relative")
    print("relationships between users while being more compact.")
    print("First dimension might represent 'adventurousness',")
    print("Second dimension might represent 'health-consciousness'.")

def part_five():
    """
    Demonstrates learning representations through a simple autoencoder.
    Shows how neural networks can automatically find useful features.
    """
    print("\nDemonstrating Learned Representations:")
    
    # Define food items and their properties
    food_items = [
        'pizza', 'sushi', 'salad', 'burger', 'curry', 
        'pasta', 'steak', 'tofu', 'soup', 'sandwich'
    ]
    
    # Create synthetic user preference data
    # 1 indicates like, 0 indicates dislike
    user_preferences = np.array([
        # Italian/Western preferences
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],  # User 1
        # Asian/Vegetarian preferences
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],  # User 2
        # Balanced preferences
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],  # User 3
        # Fast food preferences
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],  # User 4
    ])

    # Create and train a simple embedding model
    model = create_simple_embedding_model(len(food_items))
    model.fit(user_preferences, user_preferences, 
             epochs=100, verbose=0)
    
    # Get the learned embeddings
    embeddings = model.layers[0].get_weights()[0]
    
    # Demonstrate how similar items are closer in embedding space
    print("\nLearned 2D embeddings for food items:")
    for item, embedding in zip(food_items, embeddings):
        print(f"{item}: {embedding}")
    
    # Show some example similarities in the learned space
    print("\nSimilarities in learned space:")
    
    def embedding_similarity(item1, item2):
        idx1 = food_items.index(item1)
        idx2 = food_items.index(item2)
        return cosine_similarity(embeddings[idx1:idx1+1], embeddings[idx2:idx2+1])
    
    examples = [
        ('pizza', 'pasta'),    # Should be similar (Italian)
        ('sushi', 'curry'),    # Should be similar (Asian)
        ('pizza', 'sushi'),    # Should be less similar
        ('salad', 'tofu'),     # Should be similar (Healthy/Vegetarian)
    ]
    
    for item1, item2 in examples:
        sim = embedding_similarity(item1, item2)
        print(f"{item1} vs {item2}: {sim[0][0]:.3f}")
    
    print("\nObservation: The model has learned to embed foods with")
    print("similar properties closer together in the embedding space,")
    print("capturing relationships like cuisine type and dietary category")
    print("without explicitly being told about these properties.")

if __name__ == "__main__":
    part_one()
    part_two()
    part_three()
    part_four()
    part_five() 