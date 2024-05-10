import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA

# Set environment variable to disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Instantiate language model and specify data directory
llm = get_hf_llm(temperature=0.9)
genai_docs = r"C:\ALL\AI\PROJECT\rag_langchain\data_source\generative_ai"

# Build RAG chain
genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")

# Create FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using Langchain’s Runnable interfaces",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Route for health check
@app.get("/check")
async def check():
    return {"status": "ok"}

# Route for generative AI task
@app.post("/generative_ai", response_model=OutputQA)
async def generative_ai(inputs: InputQA):
    answer = genai_chain.invoke(inputs.question)
    return {"answer": answer}

# Add Langserve routes for playground
add_routes(
    app,
    genai_chain,
    playground_type="default",
    path="/generative_ai"
)
