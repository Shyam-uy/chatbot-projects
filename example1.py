# Importing packages
import os
from constants import opneai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

# Integrate our OpenAI key
os.environ["OPENAI_API_KEY"] = "sk-*****1szfyO1M9wcZvMT3BlbkFJJr22yYNcULHxQv####"

# Using steramlit instead of flask (for textbox UI)
import streamlit as st

# streamlit framework
st.title("Celebrity Search Results")
input_text = st.text_input("Search the topic you want")

# Prompt Templates (Prompt Engineering)
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about this celebrity {name}"
)

# Memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
events_memory = ConversationBufferMemory(input_key='dob', memory_key='events_history')

# For every prompt template, we will be having a LLM Chain
# OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

# Let's build our 2nd templace
second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {name} born?"
)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

# Let's build our 3rd templace
third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "5 important events happened around {dob} "
)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='events', memory=events_memory)

# Joining both the llms
parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob', 'events'], verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    
    with st.expander('Major Events'):
        st.info(events_memory.buffer)

