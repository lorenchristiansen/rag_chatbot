from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain import hub
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.load import dumps, loads
from langgraph.graph import START, StateGraph
from langchain.prompts import ChatPromptTemplate
from typing_extensions import List, TypedDict
import json
import os

app = Flask(__name__)
CORS(app)  # Allows frontend to communicate with backend

os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RAG-test"

embeddings = OllamaEmbeddings(model="codegemma")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# Load Ollama LLM
llm = ChatOllama(
    model="codegemma",
    temperature=0,
)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

# Define state for application
class State(TypedDict):
    question: str
    generated_queries: str
    context: List[Document]
    answer: str

def generate_queries(state: State):
    """
    Generates multiple query perspectives from the user's input question.
    
    Args:
        state (State): The current state containing the user question.
    
    Returns:
        List[List[str]]: A list containing a single list of query variations.
    """
    queries = (
        prompt_perspectives
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))  # Splitting into multiple queries
    ).invoke({"question": state["question"]})  # Invoke the chain dynamically
    return {"generated_queries": queries}

# Define application steps
def retrieve(state: State):
    retrieved_docs = [vector_store.similarity_search(query) for query in state["generated_queries"]]
    return {"context": retrieved_docs}  # List[List[Document]]

# def get_unique_union(documents: list[list]):
#     """ Unique union of retrieved docs """
#     # Flatten list of lists, and convert each Document to string
#     flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
#     # Get unique documents
#     unique_docs = list(set(flattened_docs))
#     # Return
#     return [loads(doc) for doc in unique_docs]  

def get_unique_union(state: State):
    """ Unique union of retrieved documents. """
    # Flatten list of lists and serialize each Document object
    flattened_docs = [json.dumps(doc.dict(), sort_keys=True) for sublist in state["context"] for doc in sublist]

    # Get unique serialized documents
    unique_docs = list(set(flattened_docs))
    # Deserialize back into Document objects
    unique_documents = [Document(**json.loads(doc)) for doc in unique_docs]

    # Deserialize back into Document objects
    return {"context": unique_documents}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([generate_queries, retrieve, get_unique_union, generate])
#graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge(START, "generate_queries")
graph = graph_builder.compile()

#####################################START ROUTES################################

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Empty query"}), 400

    response = graph.invoke({"question": query})
    return jsonify(response["answer"])

if __name__ == "__main__":
    app.run(debug=True)
