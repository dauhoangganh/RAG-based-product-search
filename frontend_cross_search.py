from openai import OpenAI
import streamlit as st
from io import BytesIO
import tempfile
from cross_search import summarize_pdf, find_most_comparable_part
import re
import os
import time
from bs4 import BeautifulSoup

##################################################
# Animation function
##################################################
def message_stream(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)
##################################################


st.title("Cross Reference Search")
st.caption("🚀 AI Hackathon 2025")


with st.chat_message("assistant",avatar="images/amplogo.png"):
    st.write("Upload a PDF datasheet of your current product to find Analog Devices' comparable parts.")
    time.sleep(0.5)

##################################################
# Beautify the output table with hover effect
##################################################
def add_hover_effect(html_table):
    soup = BeautifulSoup(html_table, 'html.parser')
    
    # Add the hover effect style directly to the table
    style_tag = soup.new_tag('style')
    style_tag.string = """
    table tr:hover {
        background-color: #b8e2f4;
    }
    """
    
    # Insert the style tag into the body if head is not present
    if soup.head is None:
        if soup.body is None:
            soup.insert(0, style_tag)
        else:
            soup.body.insert(0, style_tag)
    else:
        soup.head.append(style_tag)
    
    return str(soup)

# PDF file upload
uploaded_file = st.file_uploader("Upload a PDF File", type=["pdf"])

if uploaded_file is not None:
    with st.chat_message("assistant",avatar="images/amplogo.png"):
        st.write_stream(message_stream("Comparing the product you uploaded with ADI's products."))
        with st.spinner("Analyzing your input.."):
            # binary to BytesIO
            pdf_data = BytesIO(uploaded_file.read())
            
            # save PDF as temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_data.read())
                temp_file_path = temp_file.name
                st.toast('Extracting key features of your products', icon="🆗")
                pdf1_summary = summarize_pdf(temp_file_path, 2000, 100)
                st.toast('Searching for comparable parts', icon="🔍")
                most_comparable_part,pdf_lists,part_names = find_most_comparable_part(pdf1_summary)
                cleaned_comp_part = re.sub(r'^.*```.*$\n?', '', most_comparable_part, flags=re.MULTILINE) 
                st.toast('Generating answer', icon="✅")
                html_table = re.search(r'<table.*</table>', cleaned_comp_part, re.DOTALL)
                if html_table:
                    html_table = html_table.group()
                    html_table = add_hover_effect(html_table)
                html_merit_list = re.search(r'<h3.*</ul>', cleaned_comp_part, re.DOTALL)
                if html_merit_list:
                    html_merit_list = html_merit_list.group()
                    html_merit_list = re.sub(r'<h3', '<h3 style="color: #4194cb;"', html_merit_list)
            
    with st.chat_message("assistant",avatar="images/amplogo.png"):
        st.write_stream(message_stream("Here are the results:"))
        time.sleep(0.5)
        if html_table and html_merit_list:
            st.html(html_table)
            st.html(html_merit_list)
            st.markdown("**Datasheets of the most comparable parts:**")
        else:
            st.html(cleaned_comp_part)

    if len(pdf_lists) != 0:
        for i, pdf_path in enumerate(pdf_lists):
            if os.path.exists(pdf_path):  
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                
                pdf_name = os.path.basename(pdf_path)  
                st.download_button(
                    label=f"Download {pdf_name}",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    key=f"download_{i}",
                    on_click="ignore"
                )
            else:
                st.warning(f"File not found: {pdf_path}")


##################################################
