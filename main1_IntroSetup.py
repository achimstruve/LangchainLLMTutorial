from openai import OpenAI
import os
import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import textwrap

dotenv.load_dotenv()

# get API KEYs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# connect to the models
llm_claude3 = ChatAnthropic(model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY)
llm_gpt4 = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# verify the connection
system_prompt = "You explain things to people like they are five year old."
user_prompt = "What is LangChain?"

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
]

response = llm_gpt4.invoke(messages)
answer = textwrap.fill(response.content, width=100)

print(answer)