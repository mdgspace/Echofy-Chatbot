"""
Main entry point for the RAG system.
"""

# Server setup libraries
from fastapi import FastAPI as server, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configuration and utility libraries
from src.embeddings.models import StellaEmbeddings
from src.generation.llm import generate_response
from src.services.rag_pipeline import create_prompt, initialize_database, retrieve_contexts
from src.utils.helpers import parse_arguments

app = server()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Query(BaseModel):
    prompt: str

@app.websocket("/query")
async def handle_query_websocket(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established.")
    try:
        # if the loop isn't used, then the WS will only accept the 1st request, return the response and close the conn. The loop ensures that the conn stays persistent.
        while True: 
            data = await websocket.receive_json()
            query = Query(**data) # Dictionary unpacking operator.

            response, contexts = orchestratePipeline(query_text=query.prompt)

            # Send the response back to the client as JSON
            await websocket.send_json(
                {"response": response, "sources": contexts, "status": 200}
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Send an error message to the client before closing
        await websocket.send_json(
            {"detail": str(e), "status": 500}
        )
        await websocket.close()

    
def orchestratePipeline(query_text):
    """Main function orchestrating the retrieval and generation process."""
    # Initialize embedding function
    embedding_function = StellaEmbeddings()

    print(f"The query text is: {query_text}")
    
    # Initialize database if needed
    initialize_database(embedding_function)
    
    # Retrieve relevant contexts
    contexts = retrieve_contexts(query_text, embedding_function)
    
    if not contexts:
        print(f"\nUnable to find matching results for '{query_text}'")
        return
    
    print(f"Found {len(contexts)} relevant results. Generating response...")
    
    # Create prompt and generate response
    prompt = create_prompt(query_text, contexts)
    response = generate_response(prompt)
    
    print("\nResponse:")
    print(response)

    return response, contexts


if __name__ == "__main__":
    # Parse query
    query_text = parse_arguments()
    print("Received your prompt! Loading the database and finding matching results...")
    
    # Run main function
    orchestratePipeline(query_text=query_text)