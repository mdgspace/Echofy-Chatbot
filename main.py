"""
Main entry point for the RAG system.
"""

# Server setup libraries
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager 
from pydantic import BaseModel

# Configuration and utility libraries
from src.embeddings.models import StellaEmbeddings
from src.generation.llm import generate_response
from src.services.rag_pipeline import create_prompt, initialize_database, retrieve_contexts
from src.utils.helpers import parse_arguments

# Authentication
from config import HUGGING_FACE_ACCESS_TOKEN, DEBUG
from huggingface_hub import login
import logging

# Define Colored Formatter for Logs
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",   # Blue
        logging.INFO: "\033[92m",    # Green
        logging.WARNING: "\033[93m", # Yellow
        logging.ERROR: "\033[91m",   # Red
        logging.CRITICAL: "\033[95m" # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{log_message}{self.RESET}"

# Configure Root Logger
log_level = logging.DEBUG if DEBUG else logging.INFO
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
logging.basicConfig(level=log_level, handlers=[handler])

# Suppress noisy external loggers
for logger_name in ["filelock", "urllib3", "huggingface_hub", "sentence_transformers", "chromadb", "httpx", "httpcore", "fsspec", "tzlocal"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# This holds the model in RAM so we don't reload it
global_embedding_model = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Executes once when the server starts.KEN:
    """
    try:
        global global_embedding_model
        global_embedding_model = StellaEmbeddings()
        login(token=HUGGING_FACE_ACCESS_TOKEN)
        initialize_database(global_embedding_model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model during startup: {e}")
    yield # Server runs here


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    prompt: str

@app.websocket("/query")
async def handle_query_websocket(websocket: WebSocket):
    await websocket.accept()

    if global_embedding_model is None:
        await websocket.send_json({"detail": "Server is still warming up. Try again in 5s.", "status": 503})
        await websocket.close()
        return

    try:
        while True: 
            data = await websocket.receive_json()
            query = Query(**data)

            # Pass the PRE-LOADED global model
            response, _ = orchestratePipeline(query.prompt, global_embedding_model)

            await websocket.send_json(
                {"response": response, "status": 200}
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await websocket.send_json({"detail": str(e), "status": 500})
        await websocket.close()

    
def orchestratePipeline(query_text, embedding_model):
    """
    Main function. Note: It now accepts the 'embedding_model' as an argument.
    """

    # 1. Retrieve (Uses pre-loaded model)
    contexts = retrieve_contexts(query_text, embedding_model)
    
    if not contexts:
        logger.warning(f"Unable to find matching results for '{query_text}'")
        
    # 2. Generate
    prompt = create_prompt(query_text, contexts)
    response = generate_response(prompt)

    return response, contexts


# --- CLI / INTERACTIVE MODE SETUP ---
if __name__ == "__main__":    
    logger.info("CLI Starting...")
    
    # Load Model MANUALLY for CLI mode (since lifespan only runs for Server)
    if global_embedding_model is None:
        logger.info("Loading Model...")
        global_embedding_model = StellaEmbeddings()
        logger.info("Model loaded.")

    initial_query = parse_arguments()
    
    if initial_query:
        # One-off run
        response, _ = orchestratePipeline(initial_query, global_embedding_model)
        logger.info(f"\nResponse:\n{response}")
    else:
        # Interactive Loop
        logger.info("\n--- Interactive Mode (Type 'exit' to quit) ---")
        while True:
            try:
                user_input = input("\nEnter your prompt: ")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                if not user_input.strip():
                    continue
                
                response, _ = orchestratePipeline(user_input, global_embedding_model)
                logger.info(f"\nResponse:\n{response}")
                
            except KeyboardInterrupt:
                logger.info("\nBye!")
                break