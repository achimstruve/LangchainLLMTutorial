from openai import OpenAI
import os
import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import textwrap


dotenv.load_dotenv()

# get API KEYs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# connect to the models
llm_claude3 = ChatAnthropic(model="claude-3-opus-20240229", api_key=ANTHROPIC_API_KEY)
llm_gpt4 = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

prompt_template = """
You are a helpful assistant that explains AI topics. Given the following input: {topic}
Provide an explanation of the given topic.
"""

# Create the prompt form the prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template=prompt_template,
)

# Assemble the chain using the pipe operator "|"
chain = prompt | llm_gpt4

print(chain.invoke({"topic":"What is LangChain?"}).content)

