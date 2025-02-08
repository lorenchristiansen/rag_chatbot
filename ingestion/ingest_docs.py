from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import bs4
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
model_name = os.getenv("MODEL_NAME")

embeddings = OllamaEmbeddings(model=model_name)
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

print("Successfully stored resources")


# text_file_path = "docs/RemoteWork.txt"

# # Read the text from the file
# with open(text_file_path, "r", encoding="utf-8") as file:
#     text = file.read()
  
# # Chunk the document
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_text(text)

# documents = [Document(page_content=chunk) for chunk in chunks]

# # Create embeddings and store in ChromaDB
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vector_db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# # Add documents to the vector store
# vector_db.add_documents(documents)

# print("Sales documentation successfully ingested into ChromaDB")
