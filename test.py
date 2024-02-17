from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate


import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_version = "2023-07-01-preview"
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")

db = SQLDatabase.from_uri("postgresql://trstaikn:x7EKrFNMiv0m1zs03b-QZPRXwG3dd0S_@rosie.db.elephantsql.com/trstaikn")
print(db)
model = ChatOpenAI(
        engine="gpt-35-turbo-16k",  # engine = "deployment_name"
        temperature=0
    )


qaChain = create_sql_agent(
    llm=model,
    toolkit=SQLDatabaseToolkit(db=db, llm=model),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    input_variables=["input", "agent_scratchpad"],
    handle_parsing_errors=True,
)

# result = qaChain.run("""Join the product_item table and product table and exclude 2 attribute / column: images, description. (join by product_line of product_item table and product_id of product table). 
#                   How many data row are there from the join result ?""")

# result = qaChain.run("""Join the product table and product_line table and exclude 2 attribute / column: images, description. (join by product_line of product table and product_id of product_line table). 
#                     From the join result, find me 3 highest price laptop.""")
p = """Join the product table and product_line table and exclude 2 attribute / column: images, description. (join by product_line of product table and product_id of product_line table).
                        From the join result, answer user question.
                        
                        User question: """.join('Any laptop at your store under 25000 ?')
result = qaChain({ "input": p})["output"]

print('--- result: ', result)

# qa_model = ChatOpenAI(engine="gpt-35-turbo-16k")

# qa_prompt = PromptTemplate.from_template(
#         """You will be given requirement information and product information. Compare the given information and show me if they meet the system requirements or not. 
#             If none of them match, then just say that "No product match your requirement".
#             If more than 2 products meet the requirement, choose the best 2 products.

#     Product Information:
#     {info}

#     Requirement Information:
#                 Here are the system requirements for AutoCAD:
#                 1. AutoCAD 2022:
#                 - Operating System: 64-bit Windows 10
#                 - Processor: 2.5 GHz (3+ GHz recommended)
#                 - Memory: 8 GB or 16 GB RAM
#                 - Screen Display: 1920 x 1080 resolution with True Color

#                 2. AutoCAD 2021:
#                 - Operating System: 64-bit Microsoft Windows 10 or 8.1
#                 - Processor: 2.5 GHz (3+ GHz recommended)
#                 - Memory: 8 GB or 16 GB RAM
#                 - Screen Display: 1920 x 1080 resolution with True Color

#                 3. AutoCAD 2020:
#                 - Operating System: Microsoft Windows 10 (64-bit only), 8.1 (64-bit only), or 7 SP1 (64-bit only)
#                 - Processor: 2.5 GHz (3+ GHz recommended)
#                 - Memory: 8 GB or 16 GB RAM
#                 - Screen Display: 1920 x 1080 resolution with True Color

#                 Please note that these are the recommended system requirements and may vary depending on the specific version of AutoCAD you are using."""
#     )

# chain = qa_prompt | qa_model

# print(chain.invoke({"info": result}))


