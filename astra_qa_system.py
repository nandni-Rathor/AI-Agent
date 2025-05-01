from local_embeddings import LocalEmbeddingGenerator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def find_nearest_neighbors(word_vectors, target_word, k=3):
    # Process the target word to handle questions
    processed_question = target_word.lower().strip()
    
    # Preprocess question by removing common stop words
    stop_words = {'what', 'is', 'the', 'how', 'many', 'why', 'are', 'was', 'were'}
    question_words = [word for word in processed_question.split() if word not in stop_words]
    processed_question = ' '.join(question_words)
    
    # Define common question patterns and their answers
    question_patterns = {
        'what is the size': '5.5 million square kilometers',
        'how many trees': '390 billion individual trees',
        'how many species': '16,000 species',
        'what is the role': 'regulating the global climate by absorbing vast amounts of carbon dioxide',
        'what are the threats': 'deforestation, illegal logging, and climate change',
        'what is the amazon': 'the largest tropical rainforest in the world, often referred to as the "lungs of the Earth"',
        'why is it important': 'it plays a crucial role in regulating global climate and preserving biodiversity',
        'what is needed': 'conservation efforts are essential to preserve this rich ecosystem',
        'first major muslim rule': 'The Delhi Sultanate was the first major Muslim rule in India, preceding the Mughal Empire',
        'when was mughal empire established': 'The Mughal Empire was established in 1526',
        'what were mughals known for': 'magnificent architecture, including the Taj Mahal, and their patronage of arts and literature',
        'who was an important ruler': 'Akbar the Great, who promoted cultural synthesis between Hindu and Muslim traditions'
    }
    
    # Initialize embedding model
    embeddings_model = LocalEmbeddingGenerator()
    
    try:
        # Generate embedding for the question
        question_vector = embeddings_model.get_embedding(processed_question)
        question_vector = np.array(question_vector).reshape(1, -1)
        
        # Store sentences for KNN matching
        sentences = [
            ("size", "The Amazon rainforest covers over 5.5 million square kilometers"),
            ("trees", "It is home to an estimated 390 billion individual trees"),
            ("species", "It contains around 16,000 species"),
            ("role", "The rainforest plays a crucial role in regulating the global climate by absorbing vast amounts of carbon dioxide"),
            ("threats", "It faces serious threats due to deforestation, illegal logging, and climate change"),
            ("description", "The Amazon rainforest is the largest tropical rainforest in the world, often referred to as the lungs of the Earth"),
            ("importance", "It plays a crucial role in regulating global climate and preserving biodiversity"),
            ("conservation", "Conservation efforts are essential to preserve this rich ecosystem for future generations"),
            ("mughal_establishment", "The Mughal Empire was established in 1526 as one of the most powerful Islamic dynasties in India"),
            ("sultan_rule", "The Delhi Sultanate was the first major Muslim rule in India, preceding the Mughal Empire"),
            ("cultural_synthesis", "Under Akbar the Great, the empire witnessed significant cultural synthesis between Hindu and Muslim traditions"),
            ("architecture", "The Mughals were known for their magnificent architecture, including the Taj Mahal"),
            ("cultural_influence", "The Mughals introduced Persian culture and administrative systems while adopting Indian customs")
        ]
        
        # Generate embeddings for all sentences
        sentence_vectors = []
        for _, sentence in sentences:
            vector = embeddings_model.get_embedding(sentence)
            sentence_vectors.append(vector)
        
        # Convert to numpy array
        sentence_vectors = np.array(sentence_vectors)
        
        # Initialize and fit KNN
        # Initialize and fit KNN with more neighbors
        knn = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
        knn.fit(sentence_vectors)
        
        # Find nearest neighbors
        distances, indices = knn.kneighbors(question_vector)
        
        # Get the most relevant answers
        similarity_scores = 1 - distances[0]  # Convert distances to similarities
        best_matches = [(sentences[idx][1], score) for idx, score in zip(indices[0], similarity_scores)]
        
        # Use weighted voting for answers
        if max(similarity_scores) > 0.7:  # Increased threshold for better accuracy
            # Sort by similarity score
            best_matches.sort(key=lambda x: x[1], reverse=True)
            print("\nQuestion:", target_word)
            print("Best Answer:", best_matches[0][0])
            print(f"Confidence: {best_matches[0][1]:.2%}")
            
            # Show alternative answers if they're close in confidence
            if len(best_matches) > 1 and best_matches[1][1] > 0.6:
                print("\nAlternative Answers:")
                for answer, score in best_matches[1:]:
                    if score > 0.6:  # Only show high-confidence alternatives
                        print(f"- {answer} (Confidence: {score:.2%})")
            return  # Move return inside the if block
            
        # If no good sentence match, try pattern matching
        for pattern, answer in question_patterns.items():
            if pattern in processed_question or any(word in processed_question for word in pattern.split()):
                print(f"\nQuestion: {target_word}")
                print(f"Answer: {answer}")
                return
        
        # If no matches found, show related words
        print("\nNo direct answer found. Here are some related words:")
        word_similarities = {}
        for word, vector in word_vectors.items():
            sim = cosine_similarity([question_vector], [vector])[0][0]
            word_similarities[word] = sim
        
        top_words = sorted(word_similarities.items(), key=lambda x: x[1], reverse=True)[:3]
        for word, sim in top_words:
            print(f"- {word} (similarity: {sim:.2%})")
                
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
                    
                question = input("\nEnter your question : ").strip()
                if not question:
                    print("\nError: Please enter a valid question")
                    continue
                    
                try:
                    # Process the question
                    processed_question = question.lower().strip()
                    
                    # Add context keywords for better matching
                    context_keywords = {
                        'amazon': ['rainforest', 'trees', 'species', 'conservation', 'climate'],
                        'mughal': ['empire', 'india', 'akbar', 'culture', 'traditions']
                    }
                    
                    # Determine context of question
                    question_context = None
                    for context, keywords in context_keywords.items():
                        if any(keyword in processed_question for keyword in keywords):
                            question_context = context
                            break
                    
                    # Call find_nearest_neighbors with the processed question
                    find_nearest_neighbors(word_vectors, question)
                    
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