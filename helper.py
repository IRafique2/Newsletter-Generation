from dotenv import find_dotenv, load_dotenv
import json
import requests
from langchain import LLMChain, OpenAI, PromptTemplate

from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.utilities import GoogleSerperAPIWrapper
from app import * 
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

#import model


SERPER_API_KEY = os.getenv("SERPER_API_KEY")

load_dotenv(find_dotenv())

embeddings = HuggingFaceEmbeddings(model_name="distilbert-base-nli-stsb-mean-tokens")

def search_serp(query):
    search = GoogleSerperAPIWrapper(k=5, type="search")
    response = search.results(query)
    print("Search Results:", response)  # Log the response
    return response


#pick main model
def pick_best_articles_urls(response, query):
    if 'organic' in response and len(response['organic']) > 0:
        # Extract the URLs directly from the 'organic' search results
        urls = [result['link'] for result in response['organic']]
        return urls
    model = OllamaLLM(model="gemma3:1b", temperature=0.7)
    prompt_template = """
        You are a world class journalist, researcher, tech, Software Engineer, Developer, and online course creator.
        You are amazing at finding the most interesting and relevant, useful articles in certain topics.

        QUERY RESPONSE: (response_str)

        Above is the list of search results for the query (query).

        Please choose the best 3 articles from the list and return ONLY an array of the urls.

        Do not include anything else.

        return ONLY an array of the urls.

        Also make sure the articles are recent and not too old.

        If the file, or URL is invalid, show www.google.com.
    """
    prompt = PromptTemplate(
        input_variables=["response_str", "query"],
        template=prompt_template
    )

    llm_chain = LLMChain(
        llm=model,
        prompt=prompt
    )

    urls = llm_chain.run(response_str=response_str, query=query)
    url_list = json.loads(urls)
    return url_list


# 3. Function to get content for each article from URLs and make summaries
def extract_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)

    return db


# 4. Function to summarize the articles
def summarizer(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
    template = """
      Template for summarizing articles...
      {docs} and {query} are placeholders for actual data.
    """
    prompt_template = PromptTemplate(
        input_variables=["docs", "query"], template=template
    )
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = summarizer_chain.run(docs=docs_page_content, query=query)

    return response.replace("\n", "")


# 5. Function to turn summarization into a newsletter
def generate_newsletter(summaries, query):
    summaries_str = str(summaries)
    llm = OllamaLLM(model="gemma3:1b", temperature=0.7)
    template = """
      Template for generating a newsletter...
      {summaries_str} and {query} are placeholders for actual data.
    """
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"], template=template
    )
    news_letter_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    news_letter = news_letter_chain.predict(summaries_str=summaries_str, query=query)

    return news_letter