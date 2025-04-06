from local_embeddings import LocalEmbeddingGenerator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_nearest_neighbors(word_vectors, target_word, k=3):
    # Process the target word to handle questions
    processed_question = target_word.lower().strip()
    
    # Function to check if any pattern matches the question
    def pattern_matches(question, patterns):
        return any(pattern.lower() in question for pattern in patterns)
    
    # Define comprehensive question patterns and their answers with variations
    question_patterns = {
        'size': {
            'patterns': ['size', 'area', 'how big', 'how large', 'what is the size', 'what is the area', 'how many square kilometers', 'how much area', 'total area', 'coverage'],
            'answer': 'The Amazon rainforest covers over 5.5 million square kilometers'
        },
        'trees': {
            'patterns': ['trees', 'how many trees', 'tree count', 'number of trees', 'total trees', 'tree population', 'tree quantity', 'amount of trees'],
            'answer': 'The Amazon rainforest is home to an estimated 390 billion individual trees'
        },
        'species': {
            'patterns': ['species', 'types of trees', 'tree species', 'different trees', 'variety of trees', 'tree diversity', 'how many species', 'number of species', 'tree types'],
            'answer': 'The Amazon rainforest contains around 16,000 tree species'
        },
        'role': {
            'patterns': ['role', 'function', 'purpose', 'importance', 'why important', 'significance', 'what does it do', 'how does it help', 'benefit', 'impact', 'contribution'],
            'answer': 'The Amazon rainforest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide'
        },
        'threats': {
            'patterns': ['threats', 'dangers', 'problems', 'risks', 'challenges', 'what threatens', 'what endangers', 'harmful', 'damage', 'destruction', 'negative impact'],
            'answer': 'The Amazon rainforest faces serious threats including deforestation, illegal logging, and climate change'
        },
        'carbon': {
            'patterns': ['carbon', 'co2', 'carbon dioxide', 'absorb', 'absorption', 'climate regulation', 'greenhouse gas', 'climate impact', 'carbon storage', 'carbon sink'],
            'answer': 'The Amazon rainforest plays a crucial role in absorbing vast amounts of carbon dioxide from the atmosphere'
        },
        'conservation': {
            'patterns': ['conservation', 'preserve', 'protect', 'save', 'saving', 'preservation', 'protection', 'how to help', 'what can be done', 'conservation efforts', 'sustainable'],
            'answer': 'Conservation efforts are essential to preserve this rich ecosystem for future generations'
        },
        'nickname': {
            'patterns': ['nickname', 'called', 'known as', 'referred to as', 'what is it called', 'why is it called', 'popular name', 'common name', 'alias'],
            'answer': 'The Amazon rainforest is often referred to as the "lungs of the Earth"'
        },
        'location': {
            'patterns': ['where is it', 'location', 'where can i find', 'which continent', 'which countries', 'geographical location', 'where located'],
            'answer': 'The Amazon rainforest is located in South America, spanning across several countries including Brazil, Peru, Colombia, and others'
        },
        'climate': {
            'patterns': ['climate', 'weather', 'temperature', 'rainfall', 'humidity', 'what is the climate', 'how is the weather'],
            'answer': 'The Amazon rainforest has a tropical climate with high rainfall and humidity throughout the year'
        }
    }
    
    # Check if the question matches any patterns with improved matching
    matched_categories = []
    for category, info in question_patterns.items():
        if pattern_matches(processed_question, info['patterns']):
            matched_categories.append((category, info['answer']))
    
    if matched_categories:
        print("\nFound relevant information:")
        for category, answer in matched_categories:
            print(f"\nRegarding {category}:")
            print(f"{answer}")
        return
    
    # If no specific pattern matched, use advanced similarity search
    words = processed_question.split()
    combined_similarities = {}
    
    for search_word in words:
        # Skip common words
        if search_word in {'what', 'is', 'the', 'how', 'why', 'when', 'where', 'who', 'and', 'or', 'to', 'of', 'in', 'on', 'at'}:
            continue
            
        # Get vector for search word
        search_vector = word_vectors.get(search_word, word_vectors[list(word_vectors.keys())[0]])
        
        # Calculate similarities
        for word, vector in word_vectors.items():
            if word != search_word:
                sim = cosine_similarity([search_vector], [vector])[0][0]
                combined_similarities[word] = combined_similarities.get(word, 0) + sim
    
    # Get top matches considering all relevant words
    nearest_neighbors = sorted(combined_similarities.items(), key=lambda x: x[1], reverse=True)[:k]
    
    print(f"\nMost relevant information based on your question:")
    for word, sim in nearest_neighbors:
        if sim > 0.5:  # Only show highly relevant matches
            print(f"- {word} (relevance: {sim:.2f})")
    
    print("\nIf this doesn't answer your question, try rephrasing it using more specific terms.")


def convert_paragraph_to_vectors():
    # Sample paragraph
    paragraph = "The Amazon rainforest, often referred to as the \"lungs of the Earth,\" is the largest tropical rainforest in the world, covering over 5.5 million square kilometers. It is home to an estimated 390 billion individual trees representing around 16,000 species. The rainforest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide. However, it faces serious threats due to deforestation, illegal logging, and climate change. Conservation efforts are essential to preserve this rich ecosystem for future generations."
    
    # Initialize the embedding model
    embeddings_model = LocalEmbeddingGenerator()
    
    # Split paragraph into words and remove punctuation
    words = [word.strip('.,!?') for word in paragraph.split()]
    words = [word for word in words if word]  # Remove empty strings
    
    # Convert each word to vector
    word_vectors = {}
    for word in words:
        try:
            # Generate vector embedding for the word
            vector = embeddings_model.get_embedding(word)
            
            # Store word and its vector
            word_vectors[word] = vector
            
            # Print vector verification
            print(f"\nWord: {word}")
            print(f"Vector dimension: {len(vector)}")
            print(f"Sample values (first 5): {np.array(vector[:5]).round(4)}...")
            
        except Exception as e:
            print(f"Error processing word '{word}': {e}")
    
    return word_vectors

if __name__ == "__main__":
    print("Converting paragraph to word vectors...\n")
    word_vectors = convert_paragraph_to_vectors()
    print(f"\nSuccessfully converted {len(word_vectors)} words to vectors")
    
    # Get user input for word search
    while True:
        search_word = input("\nEnter a word to see its vector (or 'quit' to exit): ").strip()
        if search_word.lower() == 'quit':
            break
            
        # Initialize embedding model for the search word
        embeddings_model = LocalEmbeddingGenerator()
        
        try:
            # Generate vector for the search word
            vector = embeddings_model.get_embedding(search_word)
            print(f"\nWord: {search_word}")
            print(f"Vector dimension: {len(vector)}")
            print(f"Sample values (first 5): {np.array(vector[:5]).round(4)}...")
            
            # Add the word temporarily to word_vectors for finding neighbors
            word_vectors[search_word] = vector
            find_nearest_neighbors(word_vectors, search_word)
            
            # Remove the temporary word if it wasn't in the original paragraph
            if search_word not in word_vectors:
                del word_vectors[search_word]
                
        except Exception as e:
            print(f"Error processing word '{search_word}': {e}")