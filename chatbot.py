import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableMap
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.agents import tool
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers.openai_functions import (
    JsonOutputFunctionsParser,
    JsonKeyOutputFunctionsParser,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
import time
from langchain.agents import AgentExecutor
from langchain.utilities import SQLDatabase

db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"))


class appSystemReqs(BaseModel):
    """System requirements of an software"""

    appv: str = Field(description="software version")
    oprSys: List[str] = Field(description="list of operating system available")
    processor: List[str] = Field(description="list of processor available")
    memory: List[int] = Field(description="list of memory capacities, unit is GB")
    disp: str = Field(description="screen display information")
    dispCard: str = Field(description="display card information")


class Info(BaseModel):
    """Information to extract"""

    reqs: List[appSystemReqs]


class BestFitInfo(BaseModel):
    """three system requirements for a software that most match the user's requirement or information"""

    firstReq: Optional[appSystemReqs] = Field(description="first system requirements")
    secondReq: Optional[appSystemReqs] = Field(description="second system requirements")
    thirdReq: Optional[appSystemReqs] = Field(description="third system requirements")
    query: str = Field(description="the requirement or the information of the user")


class UserInput(BaseModel):
    """The question or the query of the user"""

    query: str = Field(description="the question or the query of the user")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


@tool(args_schema=UserInput)
def search_system_requirements(query: UserInput) -> None:
    """Search system requirements of an application (could be game/app/...)"""

    functions = [convert_pydantic_to_openai_function(Info)]

    extraction_prompt = PromptTemplate.from_template(
        """A article will be passed to you. Extract from it most 3 system requirements for an application that are mentioned by this article match the requirement of user. 

    If user not provide enough information to you to extracting, just ask user.

    Do not extract the name of the article itself. If no system requirements are mentioned that's fine - you don't need to extract any! Just return an empty list.

    Do not make up or guess ANY extra information. Only extract what exactly is in the text.

    Article below:
    {input}"""
    )

    extraction_model = ChatOpenAI(
        engine="gpt-35-turbo-16k",  # engine = "deployment_name"
    ).bind(functions=functions, function_call={"name": "Info"})
    loader = WebBaseLoader(
        "https://www.designmaster.biz/support/autocad-system-requiremen.html"
    )
    documents = loader.load()
    doc = documents[0]
    splits = text_splitter.split_text(doc.page_content)

    prep = RunnableLambda(lambda x: [{"input": doc} for doc in splits])

    extraction_chain = (
        extraction_prompt
        | extraction_model
        | JsonKeyOutputFunctionsParser(key_name="reqs")
    )
    flatten_chain = prep | extraction_chain.map() | flatten
    result = flatten_chain.invoke({"input": doc.page_content})
    print("\n---1", result)

    choice_prompt = PromptTemplate.from_template(
        """You will be given a list of system requirements for a specific product, as well as the user's requirements and an example answer. 
        
        Your task is to choose up to three requirements from the list that best match the user's needs and provide an answer using the example answer as a template. 

        If no suitable information are mentioned that's fine - you don't need to do any! Just return an empty list.
        
        It is important not to make up or guess any additional information.

        User requirements:
        {user_input}.

        Information:
        {info}

        Please note that the following example is not an official answer, but it demonstrates the format for providing an answer. Let's call it Application X. You can add or delete attributes based on the provided information:
        Example Answer:
            Here are the system requirements I recommend for Application X:
            1/ 
            X Version: 1999
            Operating System: Windows 2 (11-bit), Windows 3 (12-bit), or Windows 7 SP1 (64-bit)
            Processor: 1 GHz (1+ GHz recommended)
            Memory: 1 GB RAM (11 GB recommended)
            Screen Display: 123 x 123 resolution with True Color
            Display Card: 4 GB GPU with 29 GB/s Bandwidth
            Pointing Device: MS-Mouse compliant
            Network: Internet connection for installation and licensing
            2/ 
            X Version: 2000
            Operating System: Windows 2 (11-bit), Windows 3 (12-bit), or Windows 7 SP1 (64-bit)
            Processor: 1 GHz (1+ GHz recommended)
            Memory: 1 GB RAM (11 GB recommended)
            Screen Display: 123 x 123 resolution with True Color
            Display Card: 4 GB GPU with 29 GB/s Bandwidth
            Pointing Device: MS-Mouse compliant
            Network: Internet connection for installation and licensing
            3/ 
            X Version: 3000
            Operating System: Windows 2 (11-bit), Windows 3 (12-bit), or Windows 7 SP1 (64-bit)
            Processor: 1 GHz (1+ GHz recommended)
            Memory: 1 GB RAM (11 GB recommended)
            Screen Display: 123 x 123 resolution with True Color
            Display Card: 4 GB GPU with 29 GB/s Bandwidth
            Pointing Device: MS-Mouse compliant
            Network: Internet connection for installation and licensing
        """
    )

    choice_model = ChatOpenAI(
        engine="gpt-35-turbo-16k",  # engine = "deployment_name"
    )

    choice_chain = choice_prompt | choice_model
    new_list = [{"system info": d} for d in result]
    info = "".join(str(item) for item in new_list)
    res = choice_chain.invoke({"user_input": query, "info": info})
    print("\n---2", res)

    return res.content


@tool
def answer_about_yourself(query: str) -> None:
    """Answer user question about introduce yourself or greetings """
    print("----- Call qa chain -----")

    qa_model = ChatOpenAI(engine="gpt-35-turbo-16k")
    qa_prompt = PromptTemplate.from_template(
        """You will be given information and the user's question, you will answer that question base on provided information.
        If user greets you, greets them back then answer the question "who are you ?". 
        It is important not to make up or guess any additional information.

    Question: {input}
    
    Information:
    You are an assistant of BKTechStore website. You will help the customer with their questions about the BKTechStore website which sells computer and related devices."""
    )

    qa_chain = qa_prompt | qa_model
    res = qa_chain.invoke({"input": query})
    return res.content


tools_functions = [
    format_tool_to_openai_function(f)
    for f in [answer_about_yourself, search_system_requirements]
]


tools_model = ChatOpenAI(
    engine="gpt-35-turbo-16k",  # engine = "deployment_name"
).bind(functions=tools_functions)

agent_prompt = PromptTemplate.from_template(
    """You are helpful but sassy assistant. You answer user's question.
        User's question is: {input}.
        Intermediate step: {agent_scratchpad}."""
)

preprocess_agent_chain = agent_prompt | tools_model | OpenAIFunctionsAgentOutputParser()

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | preprocess_agent_chain
)

tools = [answer_about_yourself, search_system_requirements]

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

print(
    "\n --- ",
    agent_executor.invoke({"input": "tell me about you"}),
    " --- \n",
)

class chatbot:
    db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"))

    def __init__(self):
        self.agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

    def chat_public(self, query):
        result = self.agent_executor.invoke({"input": query})
        print("--- result: ", result)

        return result["output"]


# class chatbot:
#     db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"))

#     keyword_templ = """Below is a history of the conversation so far, and an input question asked by the user that needs to be answered by querying relevant company documents.
# Do not answer the question. Output queries must be in both English and Vietnamese and MUST strictly follow this format: (<Vietnamese queries>) | (<English queries>).
# Examples are provided down below:

# EXAMPLES


# Chat history:{context}

# Question:
# {question}

# Search query:
# """

#     #     chat_template = """<|im_start|>system
#     # Assistant helps the online website with their questions about the BKTechStore website which sells computer and related devices. Your answer must adhere to the following criteria:
#     # You MUST follow this rule:
#     # - If question is in English, answer in English. If question is in Vietnamese, answer in Vietnamese.
#     # - Be friendly in your answers. You may use the provided sources to help answer the question. If there isn't enough information, say you don't know. If asking a clarifying question to the user would help, ask the question.
#     # - If the user greets you, respond accordingly and tell user about yourself.
#     # - You were given a database. Please answer all relevant information with this database.

#     # {user_info}

#     # Sources:
#     # {summaries}
#     # <|im_end|>

#     # Chat history:{context}

#     # <|im_start|>user
#     # {question}
#     # <|im_end|>
#     # <|im_start|>assistant
#     # """

#     suffix = """<|im_start|>system
# Assistant helps the online website with their questions about the BKTechStore website which sells computer and related devices. Your answer must adhere to the following criteria:
# You MUST follow this rule:
# - If question is in English, answer in English. If question is in Vietnamese, answer in Vietnamese.
# - If the user greets you, respond accordingly and tell user about yourself without query anything from given database.
# - Be friendly in your answers. You may use the provided sources to help answer the question. If there isn't enough information, say you don't know. If asking a clarifying question to the user would help, ask the question.
# - You do not need to query database if question of the user not relevant to electronic gadgets.
# <|im_end|>


# Relevant pieces of previous conversation:
# {history}
# (You do not need to use these pieces of information if not relevant)

# Only use below table:
# Use pcs table with name as the table primary key.
# Do not use other table.

# Question: {input}
# Thought: I was given a database. I will answer all relevant information if user ask about electronic gadgets product with this database.
# Though: if user do not required me finding products, I do not query database.
# Thought: if user ask about electronic gadgets, I should look at the tables which is allowed in the database to see what I can query.  Then I should query the schema of the most relevant tables.
# Thought: If no suitable product satisfy the answer, I will answer no suitable product.
# Thought: if answer was not a block of text, I convert it to block of text before parse out.
# {agent_scratchpad}
# """

#     def __init__(self):
#         self.llm = AzureChatOpenAI(
#             openai_api_type="azure",
#             openai_api_base= os.getenv("OPENAI_API_BASE"),
#             openai_api_version="2023-03-15-preview",
#             deployment_name="gpt-35-turbo-16k",
#             openai_api_key= os.getenv("OPENAI_API_KEY"),
#             temperature=0.5,
#             max_tokens=3000,
#         )

#         self.llm2 = AzureChatOpenAI(
#             openai_api_type="azure",
#             openai_api_base= os.getenv("OPENAI_API_BASE"),
#             openai_api_version="2023-03-15-preview",
#             deployment_name="gpt-35-turbo-16k",
#             openai_api_key= os.getenv("OPENAI_API_KEY"),
#             temperature=0.7,
#             max_tokens=1000,
#         )

#         self.llm3 = AzureChatOpenAI(
#             openai_api_type="azure",
#             openai_api_base= os.getenv("OPENAI_API_BASE"),
#             openai_api_version="2023-03-15-preview",
#             deployment_name="gpt-35-turbo-16k",
#             openai_api_key= os.getenv("OPENAI_API_KEY"),
#             temperature=0.0,
#             max_tokens=1000,
#         )
#         # self.memory = ConversationBufferMemory(input_key="question", memory_key="context")
#         self.memory = ConversationSummaryMemory(
#             llm=self.llm3, input_key="input", memory_key="history"
#         )
#         # self.qa_chain = load_qa_with_sources_chain(
#         #     llm=self.llm, chain_type="stuff", prompt=PromptTemplate.from_template(self.chat_template))
#         # self.keywordChain = LLMChain(
#         #     llm=self.llm3, prompt=PromptTemplate.from_template(self.keyword_templ))
#         # self.qaChain = LLMChain(
#         #     llm=self.llm3, prompt=PromptTemplate.from_template(self.chat_template), memory = self.memory)
#         self.qaChain = create_sql_agent(
#             handle_parsing_errors=True,
#             llm=self.llm3,
#             toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm3),
#             verbose=True,
#             agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#             input_variables=["input", "agent_scratchpad", "history"],
#             suffix=self.suffix,  # must have history as variable,
#             agent_executor_kwargs={"memory": self.memory},
#         )

#     def chat_public(self, query):
#         # result = self.qaChain(
#         #     {"summaries": "", "input": query, "history": "", "user_info": ""}
#         # )["text"]
#         if not query.endswith("."):
#             query += "."

#         result=""
#         try:
#             result = self.qaChain(
#                 {"summaries": "", "input": query, "history": "", "user_info": ""}
#             )["output"]
#         except Exception as e:
#             response = str(e)
#             if "An output parsing error occurred" in response:
#                 print("into result")
#                 result = response.removeprefix("An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: `").removesuffix("`")

#         return result
