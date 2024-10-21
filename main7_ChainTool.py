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
from langchain_community.tools import YouTubeSearchTool
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

youtube_tool = YouTubeSearchTool()

youtube_tool.run("Crypto Token Design")

# Bind the YouTube tool to the LLM
llm_with_tools = llm_gpt4.bind_tools([youtube_tool])

chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]["query"]) | youtube_tool

print(chain.invoke("Find some videos about token design"))
