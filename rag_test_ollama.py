#export LANGSMITH_TRACING="true"
#export LANGSMITH_API_KEY="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"

#from openai import OpenAI

#from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langsmith import traceable
from langchain_core.messages import AIMessage
import os

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "RAG-test"

llm = ChatOllama(
    model="codegemma",
    temperature=0,
)

#openai_client = wrap_openai(OpenAI())

def retriever(query: str):
    results = ["Harrison worked at Kensho"]
    return results


@traceable
def rag(question):
    docs = retriever(question)
    messages = [
    (
        "system",
       """Answer the users question using only the provided information below:
        {docs}""".format(docs="\n".join(docs))
    ),
    ("human", question),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg.content)

if __name__ == "__main__":
    rag("where did harrison work")











