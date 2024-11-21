import requests
from bs4 import BeautifulSoup
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import streamlit as st

# Directly assigning API keys as variables
SERP_API_KEY = "5e915515d9acb05403f1432fceeeeb9fa90def5082017e6d1783ff02f8e35f84"
OPENAI_API_KEY = "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"

def search_and_process(query):
    """Search the web using SerpAPI and process the results."""
    api_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
    }

    response = requests.get(api_url, params=params)
    if response.status_code != 200:
        raise Exception(f"Search API failed: {response.json().get('error', 'Unknown error')}")
    data = response.json()
    # Extract top results' snippets
    snippets = [result["snippet"] for result in data.get("organic_results", []) if "snippet" in result]
    return "\n".join(snippets[:10])

def generate_response(content, query):
    """Generate a response using LangChain."""
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=OPENAI_API_KEY)

    prompt_template = PromptTemplate(
        input_variables=["content", "query"],
        template="Based on the following content:\n{content}\n\nAnswer the query: {query}"
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
    
    # Generate the response
    response = llm_chain.run({"content": content, "query": query})
    return response

# Streamlit interface
st.set_page_config(page_title="Web Query System", layout="wide")
st.title("🌐 Web Query System")

# User query input
user_query = st.text_input("Enter your query:")

if st.button("Search and Generate Response"):
    if not user_query.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Processing..."):
            try:
                # Step 1: Retrieve and process web content
                content = search_and_process(user_query)

                # Step 2: Generate response using the LLM
                response = generate_response(content, user_query)

                # Display results
                st.subheader("Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
