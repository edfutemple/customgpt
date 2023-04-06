import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_api_key", methods=["POST"])
def set_api_key():
    api_key = request.form["api_key"]
    os.environ["OPENAI_API_KEY"] = api_key
    construct_index("context_data/data")
    return redirect(url_for("ask"))

@app.route("/ask")
def ask():
    return render_template("ask.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    query = request.form["query"]
    index = GPTSimpleVectorIndex.load_from_disk("index.json")
    response = index.query(query)
    return jsonify({"response": response.response})

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


def create_index_if_not_exist():
    if not os.path.exists("index.json"):
        construct_index("path/to/your/data/folder")

if __name__ == "__main__":
    create_index_if_not_exist()
    app.run(debug=True)
