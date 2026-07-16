import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from utils.path import obtener_ruta

ruta_env = obtener_ruta(".env")
load_dotenv(dotenv_path=ruta_env)

load_dotenv()
'''
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=5,
    verbose=False
)
'''
llm_groq = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    max_retries=5,
    verbose=False
)

llm = llm_groq
#llm = llm_gemini