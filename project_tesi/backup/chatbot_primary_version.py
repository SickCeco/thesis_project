import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from cypher_examples import CypherExamples

def initialize_vector_stores(embeddings, neo4j_url, neo4j_username, neo4j_password):

    vector_stores = {}
    
    # Define the property to use as "text" for each label
    label_to_text_property = {
        "Food": "name",  
        "Exercise": "description",
        "User": "name",
        "MealPlan": "focus",
        "MealDay": "name",
        "Meal": "type",
        "WorkoutPlan": "focus",
        "WorkoutDay": "name"
    }
    
    for label, text_property in label_to_text_property.items():
        try:
            vector_stores[label] = Neo4jVector.from_existing_index(
                embedding=embeddings,
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                index_name=f"vector_{label}",
                node_label=label,
                text_node_property=text_property,  
                embedding_node_property="embedding"
            )
            print(f"Vector store for {label} successfully loaded using {text_property} as text_node_property")
        except Exception as e:
            print(f"Warning: Unable to load vector store for {label}: {e}")
    
    return vector_stores

class GraphRAGBot:
    def __init__(self):
        load_dotenv()
        
        self.neo4j_url = os.getenv("NEO4J_URL")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")

        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password
        )

        # Load Cypher query examples into a dictionary
        self.cypher_examples = CypherExamples.get_all_examples()

        self.embeddings = OpenAIEmbeddings()
        self.vector_stores = initialize_vector_stores(self.embeddings, self.neo4j_url, self.neo4j_username, self.neo4j_password)

        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

        # Initialize the chain with example queries in dictionary format
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,
            qa_llm=self.llm,
            validate_cypher=True,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_examples=self.cypher_examples  
        )

    
    def get_response(self, query):
        """
        Generates a response based on the user's query using Neo4j and vector search.
        If the query matches predefined ones, it directly uses the example Cypher query.
        """
        all_vector_results = []
        for label, store in self.vector_stores.items():
            try:
                results = store.similarity_search(query, k=2)
                all_vector_results.extend(results)
            except Exception as e:
                print(f"Warning: Error in vector search for {label}: {e}")
        
        # Check if the user's query matches one of our predefined queries
        cypher_query = None
        for key, cypher in self.cypher_examples.items():
            if key in query.lower():  # Simple check, can be improved with NLP
                cypher_query = cypher
                break
        
        if cypher_query:
            print(f"Using a predefined query: {cypher_query}")
            try:
                graph_response = self.graph.query(cypher_query)
            except Exception as e:
                print(f"Warning: Error in predefined graph query: {e}")
                graph_response = "I couldn't find specific information in the graph."
        else:
            try:
                graph_response = self.cypher_chain.invoke(query)
            except Exception as e:
                print(f"Warning: Error in graph query: {e}")
                graph_response = "I couldn't find specific information in the graph."
        
        context = f"""
        Information from the graph: {graph_response}
        
        Additional information from vector search:
        {' '.join([doc.page_content for doc in all_vector_results])}
        """
        
        final_prompt = f"""
        Based on the following context, answer the user's question in a natural and informative way: {query}
        
        Context:
        {context}
        
        Make sure to:
        1. Provide a complete and coherent answer
        2. Include relevant details from the context
        3. Maintain a conversational tone
        """
        
        final_response = self.llm.invoke(final_prompt)
        return final_response

def chat_loop():
    print("Initializing chatbot...")
    bot = GraphRAGBot()
    print("ChatBot ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        try:
            print("\nProcessing your response...")
            response = bot.get_response(user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_loop()