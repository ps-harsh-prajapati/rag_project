import os 
from dotenv import load_dotenv
from groq import Groq  # Using native Groq SDK
from src.retriever import Retriever

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# Initialize the official Groq client
client = Groq(api_key=os.getenv("GROQ_API"))

def build_messages(query: str, context: list) -> list:
    """Builds the message payload for the LLM using the query and retrieved context."""
    formatted_context = "\n\n".join(
        [f"[Document {i+1}]:\n{doc}" for i, doc in enumerate(context)]
    )
    system_prompt = """# ROLE
                    You are a highly precise, analytical expert assistant. Your primary task is to answer the user's question strictly based on the provided context.

                    # INSTRUCTIONS & CONSTRAINTS
                    - STRICT GROUNDING: You must not use prior training knowledge or external information to answer the query. 
                    - THE FALLBACK: If the exact answer cannot be deduced from the context, you must explicitly state: "I cannot answer this based on the provided documents." Do not attempt to guess or hallucinate an answer.
                    - CITATIONS: Whenever possible, cite the specific source document (e.g., "[Document 1]") that supports your claims.
                    - NO HALLUCINATION: Do not infer, assume, or invent details, metrics, or facts that are not explicitly stated in the context.
                    - CLARITY: Structure your response using bullet points, headers, or concise paragraphs for readability.
                    - TONE: Maintain an objective, professional, and helpful tone."""
    user_prompt = user_prompt = f"""# CONTEXT
                                    <context>
                                    {formatted_context}

                                    </context>

                                    # USER QUERY
                                    <query>
                                    {query}
                                    </query>

                                    # TASK
                                    Formulate a comprehensive, accurate answer to the USER QUERY using ONLY the information found in the CONTEXT above."""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def generate_answer(query: str, retriever: Retriever, top_k: int = 6) -> dict:
    results = retriever.retrieve(query, top_k=top_k)
    context = results["text"].tolist()
    messages = build_messages(query, context)
    
    # Native Groq completion call using an active model ID
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0.2,
        max_tokens=1700
    )
    
    return {
        "query": query,
        "context": context,
        "answer": response.choices[0].message.content.strip()
    }
     
if __name__ == "__main__":
    print("initilizing Retriever")

    retriever = Retriever()
    print("=" * 60)
    print("🤖 RAG Assistant Ready!")
    print("Type your questions below. Type '/exit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = str(input("Enter your query: ")).strip()

            if query.lower() == "/exit":
                print("Closing chat. Goodbye!")
                break
            # Skip empty queries
            if not query:
                print("Please enter a valid query.")
                continue
            answer_data = generate_answer(query, retriever, top_k=6)
    
            print("Context:", answer_data["context"])
            print("="*50) 
            print("Query:", answer_data["query"])
            print("="*50)
            print("Answer:", answer_data["answer"])
            print("="*50)
            print("*" * 50)

        except KeyboardInterrupt:
            print("chat close by user, goodby")
            break