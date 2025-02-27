import os
from dotenv import load_dotenv

# Carica variabili d'ambiente
load_dotenv()
neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")
