import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

def create_vector_indices():
    # Carica variabili d'ambiente
    load_dotenv()
    
    # Configurazione Neo4j
    neo4j_url = os.getenv("NEO4J_URL")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    # Inizializza il modello di embedding
    embeddings = OpenAIEmbeddings()
    
    # Definizione della struttura dei nodi e delle loro proprietà
    node_structure = [
        {
            "label": ["Exercise"],
            "properties": ["name", "exercise_id", "type", "primary_muscle", "description"]
        },
        {
            "label": ["Food"],
            "properties": ["KnowledgeBase", "food_id", "name", "category", "calories", "proteins", "carbohydrates", "fats"]
        },
        {
            "label": ["Meal"],
            "properties": ["type", "meal_id"]
        },
        {
            "label": ["MealDay"],
            "properties": ["name", "mealday_id"]
        },
        {
            "label": ["MealPlan"],
            "properties": ["duration", "focus", "meal_plan_id"]
        },
        {
            "label": ["User"],
            "properties": ["name", "activity_level", "age", "goal", "goalAchieved", "height", "preferences", "user_id", "weight"]
        },
        {
            "label": ["WorkoutDay"],
            "properties": ["name", "workoutday_id"]
        },
        {
            "label": ["WorkoutPlan"],
            "properties": ["duration", "focus", "workout_plan_id"]
        }
    ]
    
    # Inizializza connessione a Neo4j
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_password))
    
    # Per ogni tipo di nodo
    for node_info in node_structure:
        label = node_info["label"][0]  # Prendi il primo elemento dell'array label
        properties = node_info["properties"]
        
        # Seleziona le proprietà testuali che possono essere utilizzate per l'embedding
        text_properties = [prop for prop in properties if prop.lower() not in ["food_id", "exercise_id", "meal_id", "mealday_id", "meal_plan_id", "user_id", "workoutday_id", "workout_plan_id", "calories", "proteins", "carbohydrates", "fats", "age", "height", "weight", "duration"]]
        
        print(f"Elaborazione di {label}...")
        print(f"Proprietà testuali selezionate: {text_properties}")
        
        # Elimina l'indice se esiste
        try:
            with driver.session() as session:
                # Controlla se l'indice esiste
                result = session.run(f"SHOW INDEXES WHERE name = 'vector_{label}'")
                if result.peek():
                    print(f"Eliminazione dell'indice esistente per {label}...")
                    session.run(f"DROP INDEX vector_{label}")
                    print(f"Indice per {label} eliminato con successo")
        except Exception as e:
            print(f"Errore durante l'eliminazione dell'indice per {label}: {str(e)}")
        
        # Crea il nuovo indice
        try:
            if text_properties:  # Verifica che ci siano proprietà testuali da utilizzare
                Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    index_name=f"vector_{label}",
                    node_label=label,
                    text_node_properties=text_properties,
                    embedding_node_property="embedding"
                )
                print(f"Indice creato con successo per {label}")
            else:
                print(f"Nessuna proprietà testuale trovata per {label}, indice non creato")
        except Exception as e:
            print(f"Errore nella creazione dell'indice per {label}: {str(e)}")
    
    # Chiudi la connessione
    driver.close()

if __name__ == "__main__":
    create_vector_indices()