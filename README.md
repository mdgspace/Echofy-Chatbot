# Echofy: MDG Space Knowledge Assistant

## Introduction

Echofy is an intelligent question-answering system built on a Retrieval-Augmented Generation (RAG) workflow. Designed specifically for MDG Space, this chatbot processes knowledge base content, stores it in vector databases, and uses advanced embedding models combined with LLM technology to generate accurate, contextually relevant responses to user queries.

## Features

- **Vector Storage**: Utilizes Chroma DB for efficient vector and metadata storage & text retrieval
- **High-Quality Embeddings**: Powered by NoMic AI's embedding model for superior semantic understanding
- **Context-Aware Responses**: Generates answers using only the most relevant retrieved information
- **Relevance Scoring**: Filters results by confidence score to ensure high-quality responses
- **Modular Architecture**: Easily extendable and maintainable code structure

## Architecture

Echofy employs a modern RAG architecture with the following components:

- **Data Processing**: Loads and processes QA pairs from MDG Space dataset
- **Embedding Generation**: Converts text to semantic vector representations
- **Vector Storage**: Stores embeddings and metadata in Chroma DB
- **Similarity Search**: Finds the most relevant content based on user queries
- **Response Generation**: Uses an LLM to generate natural language responses

## Technologies and Tools Used

Echofy is built using a modern tech stack that combines state-of-the-art tools for embedding, retrieval, and language generation:

- **[nomic-ai/nomic-embed-text-v1](https://docs.nomic.ai/docs/embedding-models/overview)**  
  Used for generating high-quality text embeddings with excellent semantic understanding.

- **[DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1:free)** (via [OpenRouter](https://openrouter.ai/))  
  A powerful 37B parameter LLM used for generating natural, context-aware responses based on the retrieved content.

- **[ChromaDB](https://docs.trychroma.com/)**  
  A fast and scalable open-source vector database used for storing and retrieving text embeddings.

- **[LangChain](https://www.langchain.com/)**  
  Framework for developing applications powered by LLMs; used to manage chaining between embedding, retrieval, and generation components.

These tools work together to provide a robust Retrieval-Augmented Generation (RAG) pipeline, ensuring accurate and relevant answers based on the MDG Space knowledge base.

## Setup Instructions

### Prerequisites

Before you start, ensure you have the following installed:

- Python 3.8 or higher
- `pip` for package management
- `docker` and `docker-compose` for containerized execution

### Clone the Repository

First, clone the repository to your local machine:

```bash
# Using HTTPS
git clone https://github.com/baync180705/Echofy-Chatbot.git

#OR

# Using SSH
git clone git@github.com:baync180705/Echofy-Chatbot.git

cd Echofy-Chatbot
```

### Manual Setup

#### Create a Virtual Environment

For Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Run the Project (APIs)

```bash
# Make the Entrypoint script executable
chmod +x entrypoint.sh

# Run the Entrypoint script
./entrypoint.sh
```

Now you may use Postman, Thunderclient, Curl or any other API testing tool of your choice.
Alternatively, you may also run it on the CLI.

#### Run the Project (CLI)

```bash
# Run with a specific query
python main.py -q 'Tell me about this super cool tech club, MDG Space !'

# Or run interactively
python main.py
```

### Docker Setup (Recommended)

All you need to do is run a single command, and let docker do its magic and get the job done for you !

```bash
docker-compose up --build
```
Now you may use Postman, Thunderclient, Curl or any other API testing tool of your choice.

## Using the APIs

Once the server is running (listening on port 8000 locally), you can interact with the FastAPI application using the /query endpoint. It will listen on **`http://127.0.0.1:8000`** by default.

### Query Endpoint

**URL:**  
`http://127.0.0.1:8000/query`

**Method:**  
`POST`

**Headers:**  
`Content-Type: application/json`

**Request Body Format (Example):**

```json
{
  "prompt": "Tell me about this super cool tech club, MDG Space !"
}
```


## Project Structure

```
echofy/
├── config.py                  # Global configuration and environment variables
├── main.py                    # FastAPI application entry point
├── entrypoint.sh              # Script to initialize and run the application
├── requirements.txt           # Python package dependencies
├── Dockerfile                 # Container configuration
├── docker-compose.yml         # Docker services configuration
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── .dockerignore              # Docker ignore rules
├── README.md                  # Project documentation

├── dataset/                      # Dataset
│   └── mdgspace_dataset.json  # MDG Space QA dataset

├── startup/                   # Startup scripts
│   ├── __init__.py            # Package initialization
│   ├── hf_login.py            # Hugging Face authentication
│   └── db.py                  # Database initialization

├── gunicorn/                  # Gunicorn configuration
│   ├── __init__.py            # Package initialization
│   └── gunicorn_config.py     # Server configuration

└── src/                       # Source code
    ├── __init__.py            # Package initialization

    ├── data/                  # Data handling
    │   ├── __init__.py
    │   └── loader.py          # Dataset loading utilities

    ├── embeddings/            # Text embedding
    │   ├── __init__.py
    │   └── models.py          # Embedding model implementations

    ├── generation/            # Text generation
    │   ├── __init__.py
    │   └── llm.py             # LLM interaction logic

    ├── retrieval/             # Vector search
    │   ├── __init__.py
    │   └── chroma_service.py  # ChromaDB operations

    ├── services/              # Core services
    │   ├── __init__.py
    │   └── rag_pipeline.py    # RAG pipeline implementation

    └── utils/                 # Utilities
        ├── __init__.py
        └── helpers.py         # Helper functions
```

## Requirements

- fastapi>=0.115.9
- uvicorn>=0.34.3
- gunicorn>=23.0.0
- pydantic>=2.11.5
- dotenv>=0.9.9
- requests>=2.32.4
- langchain-core>=0.3.65
- sentence-transformers>=4.1.0
- langchain-chroma>=0.2.4
- numpy>=2.3.0
- einops>=0.8.1

## Advanced Configuration

The system can be configured by modifying parameters in `config.py`:

### Debug Mode
- `DEBUG`: Enable/disable debug mode (default: `True`)

### File Paths
- `CHROMA_PATH`: Path to ChromaDB storage (default: `"chroma"`)
- `JSON_PATH`: Path to dataset (default: `"dataset/mdgspace_dataset.json"`)

### Embedding Configuration
- `DEFAULT_EMBEDDING_MODEL`: Model used for text embeddings (set via environment variable)

### LLM Configuration
- `LLM_MODEL`: Model for text generation (default: `"x-ai/grok-4-fast:free"`)
- `LLM_TEMPERATURE`: Temperature for text generation (default: `0.85`)
- `LLM_TOP_P`: Top-p sampling parameter (default: `0.9`)
- `LLM_TOP_K`: Top-k sampling parameter (default: `10`)

### Retrieval Configuration
- `DEFAULT_RELEVANCE_THRESHOLD`: Minimum similarity score (default: `0.5`)
- `DEFAULT_RETRIEVAL_K`: Number of contexts to retrieve (default: `10`)

### Authentication
- `HUGGING_FACE_ACCESS_TOKEN`: Authentication token for Hugging Face API (set via environment variable)