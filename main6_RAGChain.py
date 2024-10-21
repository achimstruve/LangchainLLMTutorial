from openai import OpenAI
import os
import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.redis import Redis
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import textwrap
import redis



dotenv.load_dotenv()

# get API KEYs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
REDIS_ENDPOINT = os.getenv("REDIS_ENDPOINT")
REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_PORT = os.getenv("REDIS_PORT")

# connect to the models
llm_claude3 = ChatAnthropic(model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY)
llm_gpt4 = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=UkZQWNT5scU", add_video_info=False)

# Load the vidoe transcript as documents
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs_split = text_splitter.split_documents(docs)

# setup redis database
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)

# clean database
r.flushdb()

embeddings = HuggingFaceEmbeddings()

rds = Redis.from_documents(
    docs_split,
    embeddings,
    redis_url=REDIS_URL,
    index_name="youtube",
)

# retrieve the top 10 most similar documents
retriever = rds.as_retriever(
    search_type="similarity", search_kwargs={"k":10}
)


template = """
Answer the question based only on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": (lambda x: x["question"]) | retriever,
     "question": (lambda x: x["question"])}
     | prompt
     | llm_gpt4
     | StrOutputParser()
)

answer = chain.invoke({"question": "What is the most important thing to know about token design?"})

print(answer)

""" # save the result in a formated .txt file
with open("output.txt", "w") as f:
    f.write(answer) """