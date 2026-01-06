import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv(override=True)
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Custom instruction for the LLM
prompt_template = """Write a detailed summary of the following audio transcript. 
Focus on the main arguments and provide the result in bullet points:
"{text}"
SUMMARY:"""

summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

