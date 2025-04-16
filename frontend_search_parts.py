from openai import OpenAI
import streamlit as st
from frontend_cross_search import add_hover_effect
from dotenv import load_dotenv
import os
load_dotenv()
import re

from core import run_llm_for_search_tool
import time


##################################################
# Animation function
##################################################

def message_stream(message):
    for word in message.split():
        yield word + " "
        time.sleep(0.05)
##################################################

##################################################
# style change
##################################################
st.markdown("""
    <style>
        .stForm {
            background-color: #0067B9;  /* ADI BLUE*/
            padding: 20px;
            border-radius: 10px;
        }
        .stForm label {
            color: #FFFFFF; 
        }
        .stForm input, .stForm select, .stForm textarea {
            background-color: #FFFFFF;  
            border-radius: 5px;
        }
        .stSelectbox select {
            background-color: #FFFFFF;  
            border-radius: 5px;
            padding: 10px;
            color: #000000;  /
        }
    </style>
""", unsafe_allow_html=True)
##################################################

st.title("Product Search Tool")
st.caption("🚀 AI Hackathon 2025")

with st.chat_message("assistant",avatar="images/amplogo.png"):
    st.write_stream(message_stream("Input the necessary information below so that I can help you find the most suitable parts."))
    time.sleep(0.5)
##################################################
# Form create
##################################################

with st.form(key="search_form"):
    application = st.text_area("Application/Purpose",placeholder="ex: I want to monitor blood presure, I am looking for a power supply control IC, etc")
    category = st.selectbox("Category", ["undefined","A/D Converters (ADC)", "Audio Products", "Clock and Timing", "D/A Converters (DAC)", "Embedded Security", "High Speed Logic and Data Path Management", "iButton and Memory", "Industrial Ethernet Solutions", "Interface and Isolation", "Power Monitor, Control, and Protection", "Motor and Motion Control", "Optical Communications and Sensing", "Power Management", "Processors and Microcontrollers", "RF and Microwave", "Sensors and MEMS", "Switches and Multiplexers", "Video Products"])

    vleft,vright = st.columns(2, vertical_alignment="bottom")
    min_voltage = vleft.text_input("Min Voltage", placeholder="ex: 1.8V", key="v_min")
    max_voltage = vright.text_input("Max Voltage", placeholder="ex: 3.3V", key="v_max")

    isupply = st.text_input("Supply curent max limit (ex: 1mA )")

    tleft,tright = st.columns(2, vertical_alignment="bottom")
    min_temperature = tleft.text_input("Min Temperature range", placeholder= "ex: 0℃", key="t_min")
    max_temperature = tright.text_input("Max Temperature range", placeholder= "ex: 100℃", key="t_max")

    package = st.text_input("Package", placeholder="ex: 8-lead SOIC, 16-lead LFCSP, etc")
    notes = st.text_area("Other requirements", placeholder="ex: I need a part with a low power consumption, etc")

    number_of_ans = st.number_input("How many recommendations do you need?", min_value=3, max_value=10, step=1, value=3)

    # submit button
    submit_button = st.form_submit_button(label="submit")
##################################################

##################################################
# Process for submitting form
##################################################

#transform user input to a more descriptive prompt
def generate_prompt(frontend_data):
    app = frontend_data.get('application', 'unknown')
    product_category= frontend_data.get('category', 'unknown')
    min_voltage = frontend_data.get('min_voltage', 'unknown')
    max_voltage = frontend_data.get('min_voltage', 'unknown')
    current = frontend_data.get('supply_current', 'unknown')
    min_temperature = frontend_data.get('min_temperature', 'unknown')
    max_temperature = frontend_data.get('max_temperature', 'unknown')
    package = frontend_data.get('package', 'unknown')
    notes = frontend_data.get('notes', 'nothing')

    user_prompt = (
        f"My customer wants to find a product for the following application/purpose: {app}."
        f"My customer wants to search for a product in {product_category}."
        f"The min voltage is {min_voltage}."
        f"The max voltage is {max_voltage}."
        f"The max supply current can be {current}."
        f"The min temperature condition is {min_temperature} degree Celsius."
        f"The max temperature condition is {max_temperature} degree Celsius."
        f"The package is {package}."
        f"The customer is also interested in a part with the following requirements: {notes}.\n"       
    )
    # print(user_prompt)
    
    return user_prompt

if submit_button:
    if not application:  
        st.error("Please fill in the 'Application/Purpose' field.")
    else:
        with st.chat_message("assistant",avatar="images/amplogo.png"):
            st.write_stream(message_stream("Sure, let me find the most suitable parts for you."))
            with st.spinner("Processing... Please wait."):
                frontend_data = {
                    'application': application if application != "" else "unknown",
                    'category': category if category != "" else "unknown",
                    'min_voltage': min_voltage if min_voltage != "" else "unknown",
                    'max_voltage': max_voltage if max_voltage != "" else "unknown",
                    'supply_current': isupply if isupply != "" else "unknown",
                    'min_temperature': min_temperature if min_temperature != "" else "unknown",
                    'max_temperature': max_temperature if max_temperature != "" else "unknown",
                    'package': package if package != "" else "unknown",
                    'notes': notes if notes != "" else "nothing special"
                }
                prompt_sentence = generate_prompt(frontend_data)
                #st.info(prompt_sentence)
                result = run_llm_for_search_tool(prompt_sentence, number_of_ans)
                cleaned_output = re.sub(r'^.*```.*$\n?', '', result.content, flags=re.MULTILINE)
                html_table = re.search(r'<table.*</table>', cleaned_output, re.DOTALL)
                if html_table:
                    html_table = html_table.group()
                    html_table = add_hover_effect(html_table)
        with st.chat_message("assistant",avatar="images/amplogo.png"):
            st.write_stream(message_stream("Here are the results:"))
            time.sleep(0.5)
            if html_table:
                st.html(html_table)
            else:
                st.html(cleaned_output)
##################################################

