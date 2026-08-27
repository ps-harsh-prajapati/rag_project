import ollama
from .chunking import chunk_document , file_path


''' 
    IN this Module we will be embedding the document chunks into a vector database for semantic search and retrieval.
    I am using Ollma's nomic-embed-text-embedding-3-large model for embedding the document chunks into a vector database.
    The embedding model is a large language model that is trained to generate embeddings for text data.
    
    Embedding Dimension: 768
    parameters: 137M
    context window: 2048 tokens
'''
def embed_doc(file_path: str) -> list:
    embedding = []
    chunks = []
    input_text = chunk_document(file_path)
    
    for chunk in input_text:
        response = ollama.embed(
        model='nomic-embed-text',
        input=chunk,
        )
        
        vector = response["embeddings"][0]
        
        if not vector:
            print(f"skipping empty vector for chunks: {chunk[:50]}")
            continue
        
        embedding.append(vector)
        chunks.append(chunk)
    
    return embedding,chunks 
    
def embed_query(query:str ) -> list[float]:
    response = ollama.embed(
        model="nomic-embed-text",
        input = query
    )
    
    return response.embeddings[0]

if __name__ == "__main__":
    embedding , text_chunks  = embed_doc(file_path)
    print(f"Embedding vector for first chunk: {embedding[0][:10]}... (truncated)") # prints the embedding vector for the first chunk    
    print(f"Total number of chunks embedded: {len(embedding)}")
    
    print(type(embedding))          #<class 'list'>
    print(type(embedding[0]))       #<class 'list'>
    print(type(embedding[0][0]))    #<class 'float'>
    print("Embedding dimension:" ,len(embedding[0]))# prints the dimension of the embedding vector