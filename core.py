from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
import os
from time import sleep
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import json
from json import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# Azure OpenAI Embeddings setup
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",
                                    azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] ,
                                    openai_api_version="2023-05-15",
                                    deployment="hackathon-emb-emb3l-team-23-9uyav")
# Azure Chat OpenAI setup
llm = AzureChatOpenAI(verbose=True,
                        azure_deployment="gpt-4o-mini",  # or your deployment
                        api_version="2024-10-21",  # or your api version
                        temperature=0)

INDEX_NAME = "adidb"
def get_retriever():
    # loader = DirectoryLoader("./new_data2", glob="**/*.json", loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'text_content': False})
    # docs = loader.load()
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
    # docs = text_splitter.split_documents(docs)
    # for i in range(0, len(docs), 50):
    #     print(i)
    #     vector_store.add_documents(documents=docs[i:i+50])
    #     sleep(30)
    
    return vector_store.as_retriever(search_kwargs={"k": 10})

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = doc.page_content
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (doc, score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def get_retrieval_chain_rag_fusion(retriever):
    template = template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)
    generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )
    return generate_queries | retriever.map() | reciprocal_rank_fusion | RunnableLambda(lambda x: "\n\n".join([doc[0] for doc in x]) )


def run_llm_for_search_tool(query: str, number_of_ans=3):
    template = """You are an assistant for question-answering tasks. \n
    You are helping a customer to find the best matching product for their requirements. The customer is looking for a product that meets the following requirements: \n
    {question}\n
    Use the following retrieved products to answer the question.\n
    Retrieved Products:
    {context}
    \n
    If you don't know the answer, just say that you don't know. Return {number_of_ans} best matching products with their key details.\n
    You have to compare the product details as detailed as possible. Format the output as an HTML table that shows the comparison of products. You should generate a structured comparison table and add datasheetlink to the table for each product. 
    You can flexibly design the table format based on the product details. The row headers should be products' names. The table must look professional and nice. \n
    Now, provide the <table> with the very detailed comparison of the products, no other comments are needed.
    """
    retrieval_qa_chat_prompt = PromptTemplate.from_template(template)
    retriever = get_retriever()
    retrieval_chain_rag_fusion = get_retrieval_chain_rag_fusion(retriever)
    # print("added vectors to pinecone")
    chain = {"context": retrieval_chain_rag_fusion, "question": RunnablePassthrough(),  "number_of_ans": RunnablePassthrough()} | retrieval_qa_chat_prompt | llm
    result = chain.invoke({"question":query, "number_of_ans":number_of_ans})
    
    return result


# if __name__ == "__main__":
#     res = run_llm_for_search_tool(query="My customer is looking for a high speed low noise adc driver. The max voltage is unknown. The max supply current can be unknown. The temperature conditions are unkwown. The customer is also interested in a part with good low power performance.")
#     print(res)
