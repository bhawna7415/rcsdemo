from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from urllib.parse import urlparse
from decouple import config
import os
import re
import openai
from flask import Flask, render_template, request
import openai
import pinecone
import os
from langchain.chains.conversation.memory import ConversationBufferMemory
from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.prompts import ChatPromptTemplate
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

# from pinecone import Pinecone

OPENAI_API_KEY =  config("OPENAI_API_KEY")
PINECONE_API_KEY =  config("PINECONE_API_KEY")
ENVIRONMENT =  config("ENVIRONMENT")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
index_name = "langchainvector"




model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# # initialize pinecone
pc = pinecone.Pinecone(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment="gcp-starter",  # next to api key in console
)

index = pc.Index("langchainvector") 

text_field = "text"
vectorstore = Pinecone(
    index, embed, text_field
)

query = "Provide me answers from vector storage, if answers is not available in vector storage than please provide answers from openai."

vectorstore.similarity_search(
    query,  
    k=3  
)
# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.1,
    max_tokens=500
)
retriever  = vectorstore.as_retriever(search_kwargs={"k": 1})

# Always remember that - You are a chatbot of ecommerce store, never loose your character, you are supposed to answer users questions about ecommerce product. you are not suppose to provide answer of any question that is not related to ecommerce bussiness. User is not able to see context so don't mention context word in your response, only provide response in english language

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an ecommerce store chatbot. Answer customer questions related to products, orders, shipping, and returns. Do not answer questions that are not related to ecommerce. Always stay in character, focusing on customer service and product information. Provide responses only in English.

{context}
Always remember that - You are a chatbot of ecommerce store, never loose your character, you are supposed to answer users questions about ecommerce product. you are not suppose to provide answer of any question that is not related to ecommerce bussiness. User is not able to see context so don't mention context word in your response
Question: {question}
Answer:
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo',max_tokens=300)


def answer_with_gpt(query: str, context=None):
    # system_prompt =  """
    #     you will provide responses to user queries based on the given content, ensuring that the response is relevant to the query.
    #     """
    system_prompt =  """
        Given a user query and specific content, generate a response that accurately addresses the query while maintaining coherence and relevance to the content.
        """
    messages = [
        {"role": "system", "content":system_prompt }
    ]
    context = context + '\n\n --- \n\n + ' + query
    messages.append({"role": "user", "content": context})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=messages
    )
    return '\n' + response.choices[0].message.content.strip()
def extract_urls(text):
    url_pattern = r"""(?:(?:https?://|ftp://|www\.)\S+[^\s+\)]+)| (?://(?:[A-Z0-9_\.]+\.[A-Z]{2,})+)"""
    # potential_urls = re.findall(r'\bhttps?://\S+\b', text)
    potential_urls =  re.findall(url_pattern, text, flags=re.VERBOSE)

    # Filter out potential URLs using urlparse to ensure validity
    urls = [url for url in potential_urls if urlparse(url).scheme in ('http', 'https')]
    url = ''
    if len(urls)>0:
        url +=  urls[0]
    return urls

def extract_content(text_with_urls):
    url_pattern = r'https?://\S+'

    # URLs ko remove karna
    text_without_urls = re.sub(url_pattern, '', text_with_urls)
    return text_without_urls



@app.route('/')
def index():
    return render_template('chat.html')

chat_history = []
@app.route('/query',methods=['GET',"POST"])
def query():
    if request.method == 'GET':
        user_input = str(request.args.get('text'))    
        try:
            result  = conversational_qa_chain.invoke(
                {
                    "question": user_input,
                    "chat_history": chat_history,
                }
            )
            chat_history.append(HumanMessage(content= user_input))
            chat_history.append(AIMessage(content= result.content))

            print("====== successfully retrieve the data ============", result.content)           

            urls = extract_urls(result.content)    
            response =  extract_content(str(result.content))    

            url_pattern = r'\[Image\]\((.*?)\)'

            response = re.sub(url_pattern, '', response)
            
            print("response for product is=========: ",response)
            print("urls for product is=========: ",urls)
           
            return {'response': response,"urls":urls} # , 
        except Exception as e:
            result = "Unfortunately, information is not currently accessible."
            return result
        
if __name__ == '__main__':
    app.run(debug=True)

