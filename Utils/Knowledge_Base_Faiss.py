import csv
import os
import numpy as np
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

input_file = "../Knowledge_Base/Chinese_Knowledge_Base/processed/chinese_processed.csv"
output_file = "../Knowledge_Base/Chinese_Knowledge_Base/processed/chinese_processed_embeddings.csv"
embeddings_file = "../Knowledge_Base/Chinese_Knowledge_Base/processed/chinese_embeddings.npy"
metadata_file = "../Knowledge_Base/Chinese_Knowledge_Base/processed/embeddings_metadata.json"

BATCH_SIZE = 32
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2' # can change to OpenAI-Text-Embedding

def load_model():
    """Load Sentence Transformer model"""
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded")
    return model

def read_csv_data(input_path):
    """Read CSV file and get IDs and statements"""
    print(f"Reading CSV data: {input_path}")
    try:
        df = pd.read_csv(input_path)
        required_columns = ['id', 'statement']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: CSV file missing required column '{col}'")
                raise ValueError(f"CSV file format error, missing column: {col}")
        ids = df['id'].tolist()
        statements = df['statement'].tolist()
        print(f"Successfully read {len(statements)} statements")
        return ids, statements
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

def generate_embeddings(model, statements, batch_size=32):
    """Generate text embeddings in batches"""
    all_embeddings = []
    num_batches = (len(statements) + batch_size - 1) // batch_size
    print(f"Generating embeddings for {len(statements)} statements in {num_batches} batches")
    for i in tqdm(range(0, len(statements), batch_size), desc="Generating embeddings"):
        batch = statements[i:i + batch_size]
        try:
            with torch.no_grad():
                batch_embeddings = model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}/{num_batches}: {e}")
            print("Trying to process this batch one by one...")
            batch_embeddings = []
            for text in batch:
                try:
                    with torch.no_grad():
                        embedding = model.encode(text, convert_to_numpy=True)
                    batch_embeddings.append(embedding)
                except Exception as e2:
                    print(f"Error processing text: {text[:50]}... - {e2}")
                    embedding_dim = model.get_sentence_embedding_dimension()
                    batch_embeddings.append(np.zeros(embedding_dim))
            if batch_embeddings:
                all_embeddings.append(np.vstack(batch_embeddings))
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        print(f"Embeddings generated, shape: {final_embeddings.shape}")
        return final_embeddings
    else:
        print("Error: No embeddings generated")
        return np.array([])

def save_embeddings(embeddings, ids, output_npy, output_metadata):
    """Save embeddings and metadata"""
    try:
        np.save(output_npy, embeddings)
        print(f"Embeddings saved to: {output_npy}")
        id_to_index = {str(id_val): idx for idx, id_val in enumerate(ids)}
        import json
        with open(output_metadata, 'w', encoding='utf-8') as f:
            json.dump({
                "id_to_index": id_to_index,
                "embedding_dim": embeddings.shape[1],
                "num_embeddings": embeddings.shape[0],
                "model_name": MODEL_NAME,
                "creation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, ensure_ascii=False, indent=2)
        print(f"Metadata saved to: {output_metadata}")
        return True
    except Exception as e:
        print(f"Error saving embeddings and metadata: {e}")
        return False

def main():
    """Main function"""
    print("Generating embeddings for Chinese medical statements...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist!")
        return
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    try:
        ids, statements = read_csv_data(input_file)
    except Exception as e:
        print(f"Unable to process input file: {e}")
        return
    embeddings = generate_embeddings(model, statements, BATCH_SIZE)
    if len(embeddings) == 0:
        print("Failed to generate embeddings, exiting")
        return
    success = save_embeddings(embeddings, ids, embeddings_file, metadata_file)
    if success:
        print("\nProcessing complete! Statistics:")
        print(f"- Total processed: {len(statements)}")
        print(f"- Embedding dimension: {embeddings.shape[1]}")
        print(f"- Embedding file size: {os.path.getsize(embeddings_file) / (1024*1024):.2f} MB")
        print("\nRunning a simple similarity test...")
        if len(statements) >= 3:
            from sklearn.metrics.pairwise import cosine_similarity
            import random
            sample_indices = random.sample(range(len(statements)), min(3, len(statements)))
            for i, idx in enumerate(sample_indices):
                sample_text = statements[idx]
                sample_embedding = embeddings[idx].reshape(1, -1)
                similarities = cosine_similarity(sample_embedding, embeddings)[0]
                most_similar_indices = np.argsort(similarities)[::-1][1:6]
                print(f"\nSample {i+1}: {sample_text[:100]}..." if len(sample_text) > 100 else f"\nSample {i+1}: {sample_text}")
                for rank, similar_idx in enumerate(most_similar_indices):
                    similarity = similarities[similar_idx]
                    similar_text = statements[similar_idx]
                    print(f"  Similar text {rank+1} (similarity: {similarity:.4f}): {similar_text[:100]}..." if len(similar_text) > 100 else f"  Similar text {rank+1} (similarity: {similarity:.4f}): {similar_text}")

if __name__ == "__main__":
    main()