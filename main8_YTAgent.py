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
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
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

prompt = hub.pull("hwchase17/openai-tools-agent")

@tool
def transcribe_video(video_url:str) -> str:
    "Extract transcript from YT video"
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    docs=loader.load()
    return docs

tools = [youtube_tool, transcribe_video]

agent = create_tool_calling_agent(llm_gpt4, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({
    "input": "What are the key aspects of token design for crypto startups?"
})
