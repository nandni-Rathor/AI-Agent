import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

class LocalEmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_embedding(self, text):
        return self.model.encode(text)

def find_nearest_neighbors(word_vectors, target_word, k=5, threshold=0.8):
    # Expand stop words list for better question processing
    stop_words = {'what', 'is', 'the', 'how', 'many', 'why', 'are', 'was', 'were', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'tell', 'me', 'about', 'describe'}
    
    # Improved question preprocessing
    processed_question = target_word.lower().strip()
    # Remove punctuation
    processed_question = ''.join(char for char in processed_question if char.isalnum() or char.isspace())
    question_words = [word for word in processed_question.split() if word not in stop_words]
    processed_question = ' '.join(question_words)

    embeddings_model = LocalEmbeddingGenerator()

    try:
        question_vector = embeddings_model.get_embedding(processed_question)
        question_vector = np.array(question_vector).reshape(1, -1)

        keys = list(word_vectors.keys())
        vectors = np.array([word_vectors[key] for key in keys])

        if len(vectors) == 0:
            print("No semantic vectors available for matching.")
            return

        knn = NearestNeighbors(n_neighbors=min(k, len(vectors)), metric='cosine', algorithm='brute')
        knn.fit(vectors)
        distances, indices = knn.kneighbors(question_vector)

        similarity_scores = 1 - distances[0]
        results = [(keys[idx], score) for idx, score in zip(indices[0], similarity_scores)]

        confident_answers = [r for r in results if r[1] >= threshold]

        print(f"\nQuestion: {target_word}")
        if confident_answers:
            print("Top relevant answers:")
            for answer, score in confident_answers:
                print(f"- {answer} (score: {score:.2%})")
        else:
            print("No highly relevant answer found. Here are the closest matches:")
            for answer, score in results:
                print(f"- {answer} (score: {score:.2%})")

    except Exception as e:
        print(f"Error processing question: {e}")
        return

def convert_paragraph_to_vectors():
    # Sample paragraphs
    paragraphs = [
        "The Amazon rainforest, often referred to as the \"lungs of the Earth,\" is the largest tropical rainforest in the world, covering over 5.5 million square kilometers. It is home to an estimated 390 billion individual trees representing around 16,000 species. The rainforest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide. However, it faces serious threats due to deforestation, illegal logging, and climate change. Conservation efforts are essential to preserve this rich ecosystem for future generations.",
        "The Mughal Empire, established in 1526, was one of the most powerful Islamic dynasties in India. Under rulers like Akbar the Great, the empire witnessed significant cultural synthesis between Hindu and Muslim traditions. The Mughals were known for their magnificent architecture, including the Taj Mahal, and their patronage of arts and literature. They introduced Persian culture and administrative systems, while also adopting many Indian customs and traditions. The empire's decline began in the 18th century, but their cultural influence continues to shape modern India's heritage."
    ]
    
    # Initialize the embedding model
    embeddings_model = LocalEmbeddingGenerator()
    
    # Store vectors for sentences and meaningful phrases
    semantic_vectors = {}
    for paragraph in paragraphs:
        # Improved sentence splitting with better handling of abbreviations
        sentences = [s.strip() for s in paragraph.replace('..', '.').split('.') if s.strip()]
        for sentence in sentences:
            try:
                # Store full sentence embedding with context
                sentence_lower = sentence.lower()
                semantic_vectors[sentence_lower] = embeddings_model.get_embedding(sentence_lower)
                
                # Enhanced phrase extraction
                words = sentence_lower.split()
                # Store meaningful phrases (2-5 words) with context
                for n in range(2, min(6, len(words) + 1)):
                    for i in range(len(words) - n + 1):
                        phrase = ' '.join(words[i:i+n])
                        # Only store phrases that are likely to be meaningful
                        if len(phrase.split()) > 1 and not all(word in stop_words for word in phrase.split()):
                            semantic_vectors[phrase] = embeddings_model.get_embedding(phrase)
                            
                # Store key entities and concepts
                for i, word in enumerate(words):
                    if word not in stop_words and len(word) > 2:  # Skip very short words
                        semantic_vectors[word] = embeddings_model.get_embedding(word)
                        
            except Exception as e:
                print(f"Error processing sentence or phrase: {e}")
                
    return semantic_vectors
    # Process both paragraphs with improved tokenization
    word_vectors = {}
    for paragraph in paragraphs:
        # Improved tokenization
        sentences = paragraph.split('.')
        for sentence in sentences:
            # Clean and tokenize sentence
            cleaned_sentence = sentence.strip().lower()
            if not cleaned_sentence:
                continue
                
            # Generate vector for complete sentence
            try:
                sentence_vector = embeddings_model.get_embedding(cleaned_sentence)
                # Store both individual words and phrases
                words = cleaned_sentence.split()
                for i in range(len(words)):
                    for j in range(i + 1, min(i + 4, len(words) + 1)):
                        phrase = ' '.join(words[i:j])
                        if len(phrase.split()) > 1:  # Only store meaningful phrases
                            word_vectors[phrase] = embeddings_model.get_embedding(phrase)
                    
                # Store individual words
                word = words[i].strip('.,!?')
                if word:
                    word_vectors[word] = embeddings_model.get_embedding(word)
                    
            except Exception as e:
                print(f"Error processing sentence: {e}")
    
    return word_vectors

if __name__ == "__main__":
    print("Amazon Rainforest QA System")
    print("==========================")
    
    # Initialize word vectors and embedding model
    word_vectors = {}
    embeddings_model = LocalEmbeddingGenerator()
    
    try:
        while True:
            print("\nOptions:")
            print("1. Convert paragraph to vectors")
            print("2. Insert new data")
            print("3. Retrieve data")
            print("4. View vector values")
            print("5. Search similar words")
            print("6. Ask a question")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                print("\nConverting paragraph to vectors...")
                word_vectors = convert_paragraph_to_vectors()
                print(f"\nSuccessfully converted {len(word_vectors)} words to vectors")
                
            elif choice == '2':
                text = input("\nEnter the text to insert: ").strip()
                if not text:
                    print("\nError: Empty text not allowed")
                    continue
                try:
                    vector = embeddings_model.get_embedding(text)
                    word_vectors[text] = vector
                    print(f"\nSuccessfully inserted '{text}' with vector dimension: {len(vector)}")
                except Exception as e:
                    print(f"Error inserting text: {e}")
                    
            elif choice == '3':
                if not word_vectors:
                    print("\nNo data available. Please convert paragraph or insert data first.")
                else:
                    print("\nAvailable words:")
                    for i, word in enumerate(word_vectors.keys(), 1):
                        print(f"{i}. {word}")
                        
            elif choice == '4':
                if not word_vectors:
                    print("\nNo vectors available. Please convert paragraph or insert data first.")
                else:
                    word = input("\nEnter the word to view its vector: ").strip()
                    if not word:
                        print("\nError: Please enter a valid word")
                        continue
                    if word in word_vectors:
                        vector = word_vectors[word]
                        print(f"\nVector for '{word}':")
                        print(f"Dimension: {len(vector)}")
                        print(f"Full vector values: {np.array(vector).round(4)}")
                    else:
                        print(f"\nWord '{word}' not found in the database.")
                        
            elif choice == '5':
                if not word_vectors:
                    print("\nNo data available for search. Please convert paragraph or insert data first.")
                else:
                    search_word = input("\nEnter a word to find similar words: ").strip()
                    if not search_word:
                        print("\nError: Please enter a valid word")
                        continue
                    try:
                        vector = embeddings_model.get_embedding(search_word)
                        word_vectors[search_word] = vector
                        find_nearest_neighbors(word_vectors, search_word)
                        if search_word not in word_vectors:
                            del word_vectors[search_word]
                    except Exception as e:
                        print(f"Error processing word '{search_word}': {e}")
                        
            elif choice == '6':
                if not word_vectors:
                    print("\nPlease convert paragraph first (Option 1)")
                    continue
                    
                question = input("\nEnter your question: ").strip()
                if not question:
                    print("\nError: Please enter a valid question")
                    continue
                    
                try:
                    # Enhanced question processing
                    processed_question = question.lower().strip()
                    
                    # Improved context detection
                    context_keywords = {
                        'amazon': ['rainforest', 'trees', 'species', 'conservation', 'climate', 'forest', 'earth', 'carbon', 'dioxide', 'deforestation', 'ecosystem'],
                        'mughal': ['empire', 'india', 'akbar', 'culture', 'traditions', 'architecture', 'taj', 'mahal', 'persian', 'dynasty', 'islamic']
                    }
                    
                    # Weight matches based on context
                    question_context = None
                    context_score = {'amazon': 0, 'mughal': 0}
                    
                    for word in processed_question.split():
                        for context, keywords in context_keywords.items():
                            if word in keywords:
                                context_score[context] += 1
                                
                    # Use context for better matching
                    if max(context_score.values()) > 0:
                        question_context = max(context_score.items(), key=lambda x: x[1])[0]
                        
                    # Adjust threshold based on context confidence
                    threshold = 0.85 if question_context else 0.8
                    
                    # Call find_nearest_neighbors with adjusted parameters
                    find_nearest_neighbors(word_vectors, question, k=5, threshold=threshold)
                    
                except Exception as e:
                    print(f"\nError processing question: {e}")
                    
            elif choice == '7':
                print("\nThank you for using QA System!")
                break
                
            else:
                print("\nInvalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Exiting gracefully...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please try running the program again.")