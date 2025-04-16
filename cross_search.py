from dotenv import load_dotenv
import re
import requests
import os
from time import sleep
import shutil
from core import reciprocal_rank_fusion
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()


from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.summarize import load_summarize_chain
from langchain.utils.math import cosine_similarity
# Azure OpenAI Embeddings setup
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",
                                    azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"] ,
                                    openai_api_version="2023-05-15",
                                    deployment="hackathon-emb-emb3l-team-23-9uyav")
# Azure Chat OpenAI setup
llm = AzureChatOpenAI(verbose=True,
                        model="gpt-4o-mini",  # or your deployment
                        api_version="2024-10-21",  # or your api version
                        temperature=0)

INDEX_NAME = "adidb"

vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

def clean_extracted_text(text):
    """Cleans extracted text by removing excessive dots, headers, and unwanted characters."""
    
    # Remove excessive dots (e.g., ".......", ".....")
    text = re.sub(r"\.{2,}", ".", text)  # Replace multiple dots with a single dot
    
    # Remove excessive spaces and newlines
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    text = re.sub(r"\n{2,}", "\n", text)  # Replace multiple newlines with a single newline
    
    # Remove special characters like \xa0 (non-breaking space)
    text = text.replace("\xa0", " ").strip()
    
    return text

##  Define the summarize pdf function
# Define the main function that will take pdf file path as an input and generate a summary of the file.
def summarize_pdf(pdf_file_path, chunk_size, chunk_overlap):
    # map_prompt = PromptTemplate(
    #     input_variables=["text"],
    #     template="""
    #             Write a summary of this chunk of text that includes the main points and any important details.
    #                   {text}
    #             """
    # )
    combine_prompt = PromptTemplate(
    template="""
            Write a summary of the following text delimited by triple backquotes.
            Return your response which covers the key points of the text such as product category, description, key features, specifications, important electrical characteristics and any other important information about the product.
            ```{text}```
            """,
            input_variables=["text"]
    )
    #Load PDF file
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()[:20] #only load the fisrt 20 pages
    #remove table pf content page and clean the text
    docs_raw_text = [doc.page_content.strip() for doc in docs_raw if not re.search(r"table of contents", doc.page_content, re.IGNORECASE)]
    docs_raw_text = [clean_extracted_text(doc) for doc in docs_raw_text]
    #Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs_chunks = text_splitter.create_documents(docs_raw_text)

    #Summarize the chunks
    # chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=combine_prompt)
    #Return the summary
    return chain.invoke(docs_chunks)["output_text"]


# Step 1: Extract and embed text from both PDFs
# pdf1_path = "ads9228.pdf"
# pdf2_path = "product2.pdf"

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to dictionary with relevant fields
    flattened_docs = [(doc[0].page_content, doc[1]) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    
    return unique_docs

def find_most_comparable_part(summary):
    relevant_docs = []
    #Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=40)
    docs_chunks = text_splitter.split_text(summary)
    for chunk in docs_chunks:
        relevant_docs.append(retriever.invoke(chunk, k=10))
    relevant_docs = reciprocal_rank_fusion(relevant_docs)
    top_docs_content = "\n\n".join([doc[0] for doc in relevant_docs[:10]]) #use only top 10 relevant docs
    template = PromptTemplate(
        input_variables=["query"],
        template="""You are a smart assistant that can help users find the most similar product from Analog Devices company's database to a given product that the user inputs. 
        Given the following products from the database: \n
        {context}
        \n
        Find the most similar parts in the database for the following product summary. You should focus on the specifications, not on the products's name when searching for similar products.  \n
        {query}\n
        Generate an answer with maximum two similar products. Here is an example of the answer format:\n
        LTC2201 -- https://www.analog.com/media/en/technical-documentation/data-sheets/2201fd.pdf\n
        LTC2202 -- https://www.analog.com/media/en/technical-documentation/data-sheets/2202fd.pdf\n

        If there is no similar product, please answer "No similar product found"."""
    )
    chain = template | llm
    result = chain.invoke({"query": summary, "context": top_docs_content}).content
    # print(result)
    if result == "No similar product found":
        return result
    # Extract PDF URLs using regex
    pdf_urls = re.findall(r'https?://[^\s\)\]]+\.pdf', result)
    download_folder = "downloaded_pdfs"
    # Remove the folder if it exists
    if os.path.exists(download_folder):
        shutil.rmtree(download_folder)
    # Create the folder
    os.makedirs(download_folder)
    # Download PDFs
    pdf_paths = []
    part_names = []
    for url in pdf_urls:
        pdf_name = url.split("/")[-1]  # Extract filename from URL
        pdf_path = os.path.join(download_folder, pdf_name)  # Save path
        part_names.append(pdf_name) #save part names
        print(f"Downloading {pdf_name}...")
        response = requests.get(url, stream=True)  # Use streaming to handle large files
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"{pdf_name} downloaded successfully!")
            pdf_paths.append(pdf_path)  # Store paths for loading
        else:
            print(f"Failed to download {pdf_name}, Status Code: {response.status_code}")
    #  Load PDFs using PyPDFLoader
    adipart_document_summarizations = []
    for pdf in pdf_paths:
        adipart_document_summarizations.append(summarize_pdf(pdf, 2000, 100))
    
    adi_product_summaries = "\n\n".join(doc for doc in adipart_document_summarizations)
    
    template = PromptTemplate(
        input_variables=["pairs"],
        template="""
        You are a smart assistant that can help users compare products of Analog Devices company and other companies' products. 
        You are given the list of product summaries: a summary of user input product and the summaries of one or two products from Analog Devices that are similar to the user input product. You have to compare the product details and generate a structured comparison table. Please highlight the advantages of Analog Devices' products that the user input product does not have. You can flexibly design the table structure based on the product details.
        Here is the list product summary of the product that the user input: \n

        {summary_list1}\n\n

        Here are the product summaries of the most similar parts in the Analog Devices company's database, wrapped inside the triple backquotes: \n
        ```{summary_list2}```\n

        Output the structured comparison in HTML table format. The table must look professional and nice. The column headers must be "Feature and the products' name only. The "feature" column should include information about product category, description and all features and specifications of the products\n
        Please also provide the merits of using similar products from Analog Devices, do not talk about the advatanges of user input product. If Analog Devices' products do not have any advantages, please do not provide any information about the merits of using the products from Analog Devices.
        First, provide the comparison table, then provide the merits of using the products from Analog Devices. THe answer for reasons must start with <h3>Merits of Using the Products from Analog Devices</h3> and then list the reasons.
        """
    )
    chain = template | llm
    result = chain.invoke({"summary_list1": summary, "summary_list2": adi_product_summaries}).content
    return result,pdf_paths,part_names #modiefied this line to return pdf_paths and part_names by Kana

# if __name__ == "__main__":
    
#     pdf1_summary = summarize_pdf(pdf1_path, 2000, 100)
#     # print("pdf1 sum:" ,pdf1_summary)
#     most_comparable_part = find_most_comparable_part(pdf1_summary)
#     print("Most similar parts are: ", most_comparable_part)

