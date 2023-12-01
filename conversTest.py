from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


import os

# first initialize the large language model
llm = AzureChatOpenAI(
    openai_api_type="azure",
    openai_api_base="https://openai-nois-intern.openai.azure.com/",
    openai_api_version="2023-03-15-preview",
    deployment_name="gpt-35-turbo-16k",
    openai_api_key="400568d9a16740b88aff437480544a39",
    temperature=0.5,
    max_tokens=3000,
)

# now initialize the conversation chain
# conversation = ConversationChain(llm=llm)
# print("conversation: ")
# print(conversation.prompt.template)
# Current conversation:
# {history}
# Human: {input}
# AI:











# CONVERSATION BUFFER 

# conversation_buf = ConversationChain(
#     llm=llm,
#     memory=ConversationBufferMemory()
# )
# conversation_buf("Good morning AI!")
# conversation_buf("I wanna buy a Collie dog")
# conversation_buf("Can it be raised in a tropical climate?")
# print("\nconversation_buf: ")
# print(conversation_buf.memory.buffer)












# CONVERSATION BUFFER 

# conversation_sum = ConversationChain(
#     llm=llm,
#     memory=ConversationSummaryMemory(llm=llm)
# )
# conversation_sum("Good morning AI!")
# conversation_sum("I wanna buy a Collie dog")
# conversation_sum("Can it be raised in a tropical climate?")
# print("\nconversation_sum: ")
# print(conversation_sum.memory.buffer)













# CONVERSATION ENTITY MEMORY

conversation = ConversationChain(
	llm=llm, 
	verbose=True,
	prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
	memory=ConversationEntityMemory(llm=llm)
)













# CONVERSATION KNOWLEDGE GRAPH MEMORY


# memory=ConversationKGMemory(llm=llm)
# conversation_kg = ConversationChain(
#     llm=llm, 
#     memory=memory
# )
# memory.save_context({"input": "say hi to sam"}, {"output": "who is sam"})
# memory.save_context({"input": "sam is a friend"}, {"output": "okay"})
# print(conversation_kg.memory.kg.get_triples()) #[('say hi', 'sam', 'to'), ('sam', 'friend', 'is a')]
# print('endline')
# print(memory.load_memory_variables({"input": 'who is sam'})) #{'history': 'On sam: sam is a friend.'}
# print(memory.get_knowledge_triplets("who is sam")) #[KnowledgeTriple(subject='Sam', predicate='is', object_='a friend')]


# template = """The following is a conversation between a human and an AI where user can ask for their computer recommendation. 
# If the AI does not know the answer to a question, it truthfully says it does not know. 
# The AI also uses information contained in the "Relevant Information" section to know more about user needs.

# Relevant Information:

# {history}

# Conversation:
# Human: {input}
# AI:"""
# prompt = PromptTemplate(input_variables=["history", "input"], template=template)
# conversation_with_kg = ConversationChain(
#     llm=llm, verbose=True, prompt=prompt, memory=ConversationKGMemory(llm=llm)
# )
# conversation_with_kg("I am a designer and i want to buy a computer which can use Photoshop smoothly")
# conversation_with_kg("Can you recommend me any computer ?")
# conversation_with_kg("What do you know about me?")
# conversation_with_kg("What computer was you recommend ?")
# print(conversation_with_kg.memory.kg.get_triples()) 
# print(conversation_with_kg.memory.kg())

# template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
# If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

# Relevant Information:

# {history}

# Conversation:
# Human: {input}
# AI:"""
# prompt = PromptTemplate(input_variables=["history", "input"], template=template)
# memory = ConversationKGMemory(llm=llm)
# conversation_with_kg = LLMChain(
#     llm=llm, verbose=True, prompt=prompt, memory=memory
# )
# print(conversation_with_kg.predict(
#     input="My name is James and i want to buy a computer"
# ))
# print(conversation_with_kg.predict(
#     input="Recommend me the one which i can use photoshop"
# ))
# print(conversation_with_kg.predict(input="What do you know about me ?"))
# print(memory.load_memory_variables({"input": 'What do you know about me ?'})) #
















# prompt_template = """
# Answer user's question of adopting any {animal}

# Chat history: 
# {history}
# """
# prompt = PromptTemplate(input_variables=["animal", "history"], template=prompt_template)
# conversationllm = LLMChain(llm=llm, prompt=prompt, memory=ConversationBufferMemory())
# conversationllm("Good morning AI!")
# conversationllm("I wanna buy a Collie dog")
# conversationllm("Can it be raised in a tropical climate?")
# print("\nconversationllm:")
# print(conversationllm.prompt.template)
# print(conversationllm.memory.buffer)

