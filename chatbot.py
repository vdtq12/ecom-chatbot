from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    ConversationSummaryMemory
)
from langchain_experimental.sql import SQLDatabaseChain
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import create_sql_agent,SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

load_dotenv()

class chatbot:
    db = SQLDatabase.from_uri(os.getenv("DATABASE_URI"))

    keyword_templ = """Below is a history of the conversation so far, and an input question asked by the user that needs to be answered by querying relevant company documents.
Do not answer the question. Output queries must be in both English and Vietnamese and MUST strictly follow this format: (<Vietnamese queries>) | (<English queries>).
Examples are provided down below:

EXAMPLES


Chat history:{context}

Question:
{question}

Search query:
"""

    #     chat_template = """<|im_start|>system
    # Assistant helps the online website with their questions about the BKTechStore website which sells computer and related devices. Your answer must adhere to the following criteria:
    # You MUST follow this rule:
    # - If question is in English, answer in English. If question is in Vietnamese, answer in Vietnamese.
    # - Be friendly in your answers. You may use the provided sources to help answer the question. If there isn't enough information, say you don't know. If asking a clarifying question to the user would help, ask the question.
    # - If the user greets you, respond accordingly and tell user about yourself.
    # - You were given a database. Please answer all relevant information with this database.

    # {user_info}

    # Sources:
    # {summaries}
    # <|im_end|>

    # Chat history:{context}

    # <|im_start|>user
    # {question}
    # <|im_end|>
    # <|im_start|>assistant
    # """

    suffix = """<|im_start|>system
Assistant helps the online website with their questions about the BKTechStore website which sells computer and related devices. Your answer must adhere to the following criteria:
You MUST follow this rule:
- If question is in English, answer in English. If question is in Vietnamese, answer in Vietnamese. 
- If the user greets you, respond accordingly and tell user about yourself without query anything from given database.
- Be friendly in your answers. You may use the provided sources to help answer the question. If there isn't enough information, say you don't know. If asking a clarifying question to the user would help, ask the question.
- You do not need to query database if question of the user not relevant to electronic gadgets.
<|im_end|>


Relevant pieces of previous conversation:
{history}
(You do not need to use these pieces of information if not relevant)

Only use below table:
Use pcs table with name as the table primary key.
Do not use other table.

Question: {input}
Thought: I was given a database. I will answer all relevant information if user ask about electronic gadgets product with this database.
Though: if user do not required me finding products, I do not query database.
Thought: if user ask about electronic gadgets, I should look at the tables which is allowed in the database to see what I can query.  Then I should query the schema of the most relevant tables.
Thought: If no suitable product satisfy the answer, I will answer no suitable product.
Thought: if answer was not a block of text, I convert it to block of text before parse out. 
{agent_scratchpad}
"""

    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_type="azure",
            openai_api_base= os.getenv("OPENAI_API_BASE"),
            openai_api_version="2023-03-15-preview",
            deployment_name="gpt-35-turbo-16k",
            openai_api_key= os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            max_tokens=3000,
        )

        self.llm2 = AzureChatOpenAI(
            openai_api_type="azure",
            openai_api_base= os.getenv("OPENAI_API_BASE"),
            openai_api_version="2023-03-15-preview",
            deployment_name="gpt-35-turbo-16k",
            openai_api_key= os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=1000,
        )

        self.llm3 = AzureChatOpenAI(
            openai_api_type="azure",
            openai_api_base= os.getenv("OPENAI_API_BASE"),
            openai_api_version="2023-03-15-preview",
            deployment_name="gpt-35-turbo-16k",
            openai_api_key= os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=1000,
        )
        # self.memory = ConversationBufferMemory(input_key="question", memory_key="context")
        self.memory = ConversationSummaryMemory(
            llm=self.llm3, input_key="input", memory_key="history"
        )
        # self.qa_chain = load_qa_with_sources_chain(
        #     llm=self.llm, chain_type="stuff", prompt=PromptTemplate.from_template(self.chat_template))
        self.keywordChain = LLMChain(
            llm=self.llm3, prompt=PromptTemplate.from_template(self.keyword_templ))
        # self.qaChain = LLMChain(
        #     llm=self.llm3, prompt=PromptTemplate.from_template(self.chat_template), memory = self.memory)
        self.qaChain = create_sql_agent(
            handle_parsing_errors=True,
            llm=self.llm3,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm3),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            input_variables=["input", "agent_scratchpad", "history"],
            suffix=self.suffix,  # must have history as variable,
            agent_executor_kwargs={"memory": self.memory},
        )

    def chat_public(self, query):
        # result = self.qaChain(
        #     {"summaries": "", "input": query, "history": "", "user_info": ""}
        # )["text"]
        if not query.endswith("."):
            query += "."

        result=""
        try:
            result = self.qaChain(
                {"summaries": "", "input": query, "history": "", "user_info": ""}
            )["output"]
        except Exception as e:
            response = str(e)
            if "An output parsing error occurred" in response:
                print("into result")
                result = response.removeprefix("An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor. This is the error: Could not parse LLM output: `").removesuffix("`")
    
        return result
