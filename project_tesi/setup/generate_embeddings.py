import json
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Carica le variabili dal file .env
load_dotenv()
# Carica il JSON esportato da Neo4j
with open("setup/graph_filtered.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Estrai il contenuto della chiave "graph"
graph_data = data.get("graph", [])

# Liste per i nodi e le relazioni
nodes = {}
relationships = []

print("sto estaendo i dati...")
# Estrazione nodi e relazioni
for entry in graph_data:
    if "n" in entry:  # Nodo iniziale
        nodes[entry["n"]["id"]] = entry["n"]
    if "m" in entry:  # Nodo finale
        nodes[entry["m"]["id"]] = entry["m"]
    if "r" in entry:  # Relazione
        relationships.append(entry["r"])

# Funzione per estrarre testo significativo dai nodi
def extract_text_from_node(node):
    properties = node.get("properties", {})
    return " ".join(str(value) for value in properties.values())

# Inizializzare il modello di embedding
embeddings_model = OpenAIEmbeddings()

# Generare embedding per ogni nodo
node_embeddings = {
    node_id: embeddings_model.embed_query(extract_text_from_node(node))
    for node_id, node in nodes.items()
}

# Salvare gli embedding in un file JSON
with open("node_embeddings.json", "w", encoding="utf-8") as outfile:
    json.dump(node_embeddings, outfile, indent=4)

print("✅ Embedding generati e salvati in node_embeddings.json!")
