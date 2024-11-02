#with stemming and top_n keyword argument
#01/11 last update
import os
import time
import PyPDF2
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tabula
import numpy as np
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Initialize spaCy model, Sentence-BERT model, and Porter Stemmer for NLP processing
nlp = spacy.load('en_core_web_sm')
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
stemmer = PorterStemmer()


def read_job_description(job_description_file):
    with open(job_description_file, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_pdf(file_path):
    text = ''
    try:
        
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        for table in tables:
            text += table.to_string()

        
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return text

# 3.1 Preprocess text by removing stop words, lemmatizing, and stemming
def preprocess_text(text):
    doc = nlp(text.lower())
    
    # Lemmatize and stem each token if it's not a stop word and is alphabetic
    processed_tokens = []
    for token in doc:
        if not token.is_stop and token.is_alpha:
            # Lemmatize, then stem the word
            lemma = token.lemma_
            stemmed = stemmer.stem(lemma)
            processed_tokens.append(stemmed)
    
    return processed_tokens

# 3.2 Function to extract keywords using TF-IDF with handling empty resumes
def extract_keywords(text, top_n=20):
    # Ensure text is not empty after preprocessing
    # this is to fix the error if theres no text in resume
    if not text.strip():
        return []
    
    # Use TF-IDF to find the top keywords
    vectorizer = TfidfVectorizer(max_features=top_n)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
        keywords = vectorizer.get_feature_names_out()
    except ValueError:
        # Return empty list if TF-IDF fails due to empty vocabulary # this is to fix the error if theres no text in resume
        keywords = []
    return list(keywords)

# 3.3 Rank and extract keywords 
def rank_and_extract_keywords(resume_text, top_n=20):
    preprocessed_text = " ".join(preprocess_text(resume_text))
    keywords = extract_keywords(preprocessed_text, top_n=top_n)
    if len(keywords) == 0:  # Check if keywords list is empty
        print("Warning: No keywords found for document. It may contain only stop words")
    return keywords

#3.4 Function to get BERT embeddings for keywords
def extract_bert_embeddings(keywords):
    if not keywords:
        return np.zeros(sbert_model.get_sentence_embedding_dimension())  # Return zero vector if no keywords
    return np.mean(sbert_model.encode(keywords), axis=0)

# 3.5 Function to calculate similarity between job description and resume
def calculate_similarity(job_embedding, resume_embedding):
    return cosine_similarity([job_embedding], [resume_embedding])[0][0]

# Process each resume and calculate similarity using Sentence-BERT
def process_resume(resume_file, job_desc_keywords):
    resume_text = extract_text_from_pdf(resume_file)
    resume_keywords = rank_and_extract_keywords(resume_text, top_n=30)  # top n 30 for now
    
    # Get BERT embeddings for keywords
    resume_embedding = extract_bert_embeddings(resume_keywords)
    job_embedding = extract_bert_embeddings(job_desc_keywords)

    similarity_index = calculate_similarity(job_embedding, resume_embedding)
    return os.path.basename(resume_file), similarity_index, len(resume_text.split())


#############################Not needed for now###################################
# # Visualization for Number of Words vs. Similarity Index
# def plot_words_vs_similarity(results):
#     words_count = [resume['word_count'] for resume in results]
#     similarity_scores = [resume['similarity'] for resume in results]
    
#     plt.figure(figsize=(8, 6))
#     plt.plot(words_count, similarity_scores, marker='o')
#     plt.title('Number of Words vs. Similarity Index')
#     plt.xlabel('Number of Words')
#     plt.ylabel('Similarity Index')
#     plt.grid(True)
#     plt.show()

# # Visualization for Number of Resumes vs. Processing Time
# def plot_resumes_vs_time(resume_count, time_taken):
#     plt.figure(figsize=(8, 6))
#     plt.plot(resume_count, time_taken, marker='o')
#     plt.title('Number of Resumes vs. Processing Time')
#     plt.xlabel('Number of Resumes')
#     plt.ylabel('Processing Time (s)')
#     plt.grid(True)
#     plt.show()

########################################################################
# Main function to process resumes and shortlist candidates
def main(resume_folder, job_description_file, output_file='shortlisted_resumes_new.xlsx'):
    
    print("Starting resume processing...")
    
    # Read and preprocess the Job description
    job_desc_text = read_job_description(job_description_file)
    job_desc_keywords = rank_and_extract_keywords(job_desc_text, top_n=30)  #### Top n words for ranking but n is not comfirmed yet ####
    
    results = []
    time_taken = []
    resume_count = []
    
    print("Processing resumes from folder...")
    start_time = time.time()
    for idx, resume_file in enumerate(os.listdir(resume_folder)):
        if resume_file.endswith('.pdf'):
            resume_path = os.path.join(resume_folder, resume_file)
            resume_name, similarity_index, word_count = process_resume(resume_path, job_desc_keywords)
            results.append({'Resume Name': resume_name, 'Similarity': similarity_index, 'word_count': word_count})
            if (idx + 1) % 10 == 0:
                time_taken.append(time.time() - start_time)
                resume_count.append(idx + 1)
                print(f"Processed {idx + 1} resumes...")
    
    print(f"Processed {len(results)} resumes in total.")
    
    # Normalize similarity scores based on resume lengths
    similarity_scores = [result['Similarity'] for result in results]
    word_counts = [result['word_count'] for result in results]
    
    normalized_similarities = normalize_similarity(similarity_scores, word_counts)
    
    for idx, result in enumerate(results):
        result['Normalized Similarity'] = normalized_similarities[idx]

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    print("Sorting results by similarity score...")
    df_sorted = df.sort_values(by='Normalized Similarity', ascending=False)

    print(f"Saving sorted results to {output_file}...")
    df_sorted.to_excel(output_file, index=False)
    
    print(f"Results saved to {output_file}.")
    print("Program completed successfully!")

# Function to normalize similarity based on resume length (longer resumes dont get higher scores just because of higher number of words)
def normalize_similarity(similarity_scores, resume_lengths):
    similarity_scores = np.array(similarity_scores)
    resume_lengths = np.array(resume_lengths)

    # Normalising resume lengths between 0 and 1
    scaler = MinMaxScaler()
    norm_lengths = scaler.fit_transform(resume_lengths.reshape(-1, 1))

    # Normalising similarity scores based on length
    ## normalising similarity scores is still in question 
    # for now actual cosine similarity score and normaised score both will be saved i'll look further into it
    normalized_scores = similarity_scores / (1 + norm_lengths.flatten())
    return normalized_scores




if __name__ == "__main__":
    # Resume folder
    # set your own path
    resume_folder = r'bla bla'

    # Job description txt file
    job_description_file = r'bla bla'

    
    main(resume_folder, job_description_file)
