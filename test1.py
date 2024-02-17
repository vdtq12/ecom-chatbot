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
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.document_loaders import PyPDFLoader

# db = SQLDatabase.from_uri("postgresql://trstaikn:x7EKrFNMiv0m1zs03b-QZPRXwG3dd0S_@rosie.db.elephantsql.com/trstaikn")


class appSystemReqs(BaseModel):
    """System requirements of an software"""

    name: str = Field(description="sofware name")
    softwareVersion: str = Field(description="software version")
    operatingSystem: List[str] = Field(description="list of operating system available")
    processor: List[str] = Field(description="list of processor available")
    memory: List[int] = Field(description="list of memory capacities, unit is GB")
    screenDisplay: str = Field(description="screen display information")
    displayCard: str = Field(description="display card information")


class Info(BaseModel):
    """Information to extract"""

    reqs: List[appSystemReqs]


class CustomerInput(BaseModel):
    """The question or the query of the customer"""

    query: str = Field(description="the question or the query of the customer")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


@tool(args_schema=CustomerInput)
def search_system_requirements(query: CustomerInput):
    """Search system requirements of an application (could be game/app/...)"""

    functions = [convert_pydantic_to_openai_function(Info)]

    extraction_prompt = PromptTemplate.from_template(
        """A article will be passed to you. Extract from it most 3 system requirements for an application that are mentioned by this article match the requirement of customer. 

    If customer not provide enough information to you to extracting, just ask customer.

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
    # test_chain = extraction_prompt | extraction_model |  JsonKeyOutputFunctionsParser(key_name="reqs")
    # test_result = test_chain.invoke({"input": splits[2]})
    # print('\n ---test result: ', test_result)
    result = flatten_chain.invoke({"input": doc.page_content})
    print("\n---1", result)

    choice_prompt = PromptTemplate.from_template(
        """You will be given a list of system requirements for a specific product, as well as the customer's requirements and an example answer. 
        
        Your task is to choose up to three requirements from the list that best match the customer's needs and provide an answer using the example answer as a template. 

        If no suitable information are mentioned that's fine - you don't need to do any! Just return an empty list.
        
        It is important not to make up or guess any additional information.

        Customer requirements:
        {customer_input}.

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
    res = choice_chain.invoke({"customer_input": query, "info": info})
    print("\n---2", res)

    return res


@tool(args_schema=CustomerInput)
def answer_about_yourself(query: CustomerInput):
    """Answer customer question about introduce yourself or greetings as hi/hello/..."""

    qa_model = ChatOpenAI(engine="gpt-35-turbo-16k")
    qa_prompt = PromptTemplate.from_template(
        """You will be given information and the customer's question, you will answer that question base on provided information.
        If customer greets you, greets them back and introduce to customer who you are base on the provided information. 
        It is important not to make up or guess any additional information.

    Question: {input}
    
    Information:
    You are an assistant of BKTechStore website. You will help the customer with their questions about the BKTechStore website which sells computer and related devices."""
    )

    qa_chain = qa_prompt | qa_model | OpenAIFunctionsAgentOutputParser()
    res = qa_chain.invoke({"input": query})
    return res

@tool(args_schema=CustomerInput)
def answer_about_bktechstore(query: CustomerInput):
    """Answer customer question about information of the BKTechStore policies such as business information, privacy, transaction term, online shopping guide, warranty policy, contact information, payment methods."""
    
    loader = PyPDFLoader('./static/chatbotref.pdf')
    docs = loader.load_and_split()

    doc_list = [doc.page_content for doc in docs]

    retriever = BM25Retriever.from_texts(doc_list)
    retriever.k = 3

    result = retriever.get_relevant_documents(query)

    policies_info = [x.page_content for x in result]

    print('\n--- result: ', result)
    print('\n--- policy info: ', policies_info)

    policy_model = ChatOpenAI(engine="gpt-35-turbo-16k")
    policy_prompt = PromptTemplate.from_template(
        """You will be given information about BKTechStore policies, answer template and the customer's question, you will answer that question base on provided information.
        If no suitable information are mentioned that's fine - you answer that currently no information about this problem and tell them to contact customer care for exact details.
        It is important not to make up or guess any additional information.
        Do not contain the phrase "Based on the provided information" in your answer.

    Question: {input}
    
    Information of policies:
    {information}
    """
    )

    policy_chain = policy_prompt | policy_model | OpenAIFunctionsAgentOutputParser()
    res = policy_chain.invoke({"input": query, "information": policies_info})
    return res

# @tool(args_schema=CustomerInput)
# def find_product(query: CustomerInput):
#     """Answer customer question about finding a product in BKTechStore with customer requirement"""

#     find_model = ChatOpenAI(
#         engine="gpt-35-turbo-16k",  # engine = "deployment_name"
#         temperature=0
#     )

#     chain1 = create_sql_agent(
#         llm=find_model,
#         toolkit=SQLDatabaseToolkit(db=db, llm=find_model),
#         verbose=True,
#         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         input_variables=["query", "agent_scratchpad"],
#         handle_parsing_errors=True,
#     )

#     result = chain1.run("""Join the product table and product_line table and exclude 2 attribute / column: images, description. (join by product_line of product table and product_id of product_line table).
#                         From the join result, answer user question.
                        
#                         User question: {query}.""")
    
#     return result


tools_functions = [
    format_tool_to_openai_function(f)
    for f in [answer_about_yourself, search_system_requirements, answer_about_bktechstore]
]



tools_model = ChatOpenAI(
    engine="gpt-35-turbo-16k",  # engine = "deployment_name"
).bind(functions=tools_functions)

# agent_prompt = PromptTemplate.from_template(
#     """ You work as an assistant for the BKTechStore website and your role is to respond to customer inquiries.
#         If a question is not relevant to the website or if there are no matching tools available, you should inform the customer that you are unable to answer their question.

#         Use the following format:
#         Question: the input question you must answer
#         Thought: you should always think about what to do
#         Action: the action to take, should be one of the tool name
#         Action Input: the input to the action
#         ... (this Thought/Action/Action Input/Observation can repeat N times)
#         Thought: I now know the final answer.
#         Final Answer: the final answer to the original input question

#         Customer's question: {input}.

#         Intermediate step: {agent_scratchpad}."""
# )

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You work as an assistant only for the BKTechStore website and your role is to respond to customer inquiries. 
            If a question is not relevant to the BKTechStore website or if there are no matching tools available, you should inform the customer that you are unable to answer their question.
            
            Here are something you can do if user ask:
            - Answer questions about the products available on the website.
            - Provide information about the BKTechStore policies such as business information, privacy, transaction terms, online shopping guide, warranty policy, and contact information.
            - Assist with for system requirements of specific applications or games.
            - Provide general information or guidance about the website and its features.""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

preprocess_agent_chain = agent_prompt | tools_model | OpenAIFunctionsAgentOutputParser()
# intermediate_steps = []

agent_chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
    )
    | preprocess_agent_chain
)

tools = [answer_about_yourself, search_system_requirements, answer_about_bktechstore]
# agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)


# print(
#     "\n --- ",
#     agent_executor.invoke({"input": "hi"}),
#     " --- \n",
# )

# print(
#     "\n --- ",
#     agent_executor.invoke({"input": "what can you do ?"}),
#     " --- \n",
# )

# print(
#     "\n --- ",
#     agent_executor.invoke({"input": "I want to buy a laptop to use AutoCAD. What is the recommendation system requirements ?"}),
#     " --- \n",
# )

# print("--- final answer: ", agent_executor({"input": "How long is the warranty period for products at your store?"}))

class chatbot:
    def __init__(self):
        self.agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

    def chat_public(self, query):
        result = self.agent_executor.invoke({"input": query})
        print("--- result: ", result)

        return result["output"]




