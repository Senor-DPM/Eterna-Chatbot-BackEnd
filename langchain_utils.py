# langchain_utils.py

import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
import getpass

key = getpass.getpass()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ['LANGCHAIN_PROJECT'] = "My First Chatbot"
os.environ["OPENAI_API_KEY"] = key

llm = ChatOpenAI(api_key=key, model="gpt-4o-mini")
DATA_PATH = "database"
CHROMA_PATH = "ChromaDatabases"

def load_docs():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=110,
        chunk_overlap=30,
        length_function = len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def saveToChroma(chunks: list[Document], user_id):

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH, collection_name=user_id
    )
    # db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH} for {user_id}.")

def generate_data_store(user_id):
    documents = load_docs()
    chunks = split_text(documents)
    saveToChroma(chunks, user_id)



PROMPT_TEMPLATE = """
Chat with the user using only the following context. You are already given the person through collections:

{context}
---

Now, with the given context, give a reply as if you were that user. Try your best to mimic their personality, even if the response may be slightly negative: and try not to just repeat the question inside the reply unless it makes sense to. Use emojis as much as that person does. Give single line replies. If a question was askeed about past messages, give the answer if available. Reply to each message with the best possible response."{question}
"""

def query_database(query_text, user_id):

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_name=user_id)

    results = db.similarity_search_with_relevance_scores(query_text, k=50)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
