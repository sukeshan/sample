import os
import fasttext
import streamlit as st
#from scrapper import scrap
from utils import check_time ,update_time
from engine import train

@st.cache_resource
def load_model():

    global model
    model = fasttext.load_model(r"DB\weigth.bin")
    return model

# Set the page title
st.title("Streamlit Search Engine")
if check_time(r'DB\time.txt') is 'Re_Train':
    st.write('Model Needs to Retrain plz wait a moment server is down as of now :(')
    update_time(r'DB\time.txt')
    train()
    model = load_model()

model = load_model()


# Get the search text from the user
search_text = st.text_input("Enter your search text:")

# Create a button to search
if st.button("Search"):
    # If the user clicks the button, search for the text

    st.write(f"The search text is: {model.get_nearest_neighbors(search_text)}")


