import subprocess
import sys
import os

# Clone the context_data repository
if not os.path.exists("context_data"):
    subprocess.check_call(["git", "clone", "https://github.com/edfutemple/context_data.git"])

# Install necessary packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("llama-index")
install("langchain")

from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True: 
        query = input("What do you want to ask? ")
        response = index.query(query)
        print(f"Response: {response.response}")  # Replaced IPython's display with a print statement

os.environ["OPENAI_API_KEY"] = input("Paste your OpenAI key here and hit enter:")
construct_index("context_data/data")
ask_ai()
