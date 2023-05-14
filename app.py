from flask import Flask, render_template, request
import openai
import pinecone

app = Flask(__name__)

OPENAI_KEY = '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'
openai.api_key = OPENAI_KEY

MODEL = "text-embedding-ada-002"
index_name = 'semantic-search'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key='3d5090d6-ab47-47a1-b4c9-7916be47d027',
    environment="us-west1-gcp-free"  # find next to api key in   console
)

# check if index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536) # 1536 is the output dimension of ada model
# connect to index
index = pinecone.Index(index_name)

def search_article(search_term):
    metadata = []
    
    xq = openai.Embedding.create(input=search_term, engine=MODEL)['data'][0]['embedding']
    res = index.query([xq], top_k=5, include_metadata=True)
    
    for i in range(len(res['matches'])):
        metadata.append(res['matches'][i]['metadata'])
        
    return metadata

search_results = []
@app.route("/",methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        query = request.form.get("search_query")
        if query == "":
            search_results = ""
        else:
            search_results = search_article(query)
    return render_template('index.html', search_results = search_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
