from neo4j import GraphDatabase
import json
import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()
neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Connessione a Neo4j
driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))

# Carica gli embeddings dal file JSON
with open("node_embeddings.json", "r", encoding="utf-8") as file:
    node_embeddings = json.load(file)

# Funzione per aggiornare i nodi con gli embeddings
def update_node_embeddings(tx, node_id, embedding):
    # Assicuriamoci che l'embedding sia un array di float
    embedding = [float(x) for x in embedding]

    query = """
    MATCH (n)
    WHERE ID(n) = $node_id
    SET n.embedding = $embedding
    """
    tx.run(query, node_id=int(node_id), embedding=embedding)

# Aggiorna i nodi nel database
with driver.session() as session:
    for node_id, embedding in node_embeddings.items():
        try:
            session.write_transaction(update_node_embeddings, int(node_id), embedding)
        except Exception as e:
            print(f"Errore con il nodo {node_id}: {e}")

print("✅ Embeddings salvati nei nodi di Neo4j correttamente!")

# Chiudi la connessione
driver.close()
