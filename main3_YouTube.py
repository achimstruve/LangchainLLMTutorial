from openai import OpenAI
import os
import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader
import textwrap


dotenv.load_dotenv()

# get API KEYs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# connect to the models
llm_claude3 = ChatAnthropic(model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY)
llm_gpt4 = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=UkZQWNT5scU", add_video_info=False)

# Load the vidoe transcript as documents
docs = loader.load()

transcript=docs[0].page_content

# We can now use the transcript in a chain
prompt_template = """
You are a helpful assistant that explains Crypto topics. Given the following video transcript: {video_transcript}
Give a summary.
"""

# Create the prompt
prompt = PromptTemplate(
    input_variables=["video_transcript"],
    template=prompt_template,
)

# Assemble the chain using the pipe operator "|"
chain = prompt | llm_gpt4

print(chain.invoke({"video_transcript":docs}).content)