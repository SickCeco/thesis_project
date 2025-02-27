from neo4j import GraphDatabase
from typing import Optional, List, Dict, Any, Tuple

class Neo4jConnector:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
        
    def query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Execute a Cypher query with optional parameters.
        
        Args:
            query: The Cypher query to execute
            parameters: Optional dictionary of parameters to pass to the query
            
        Returns:
            List of dictionaries containing the query results
        """
        with self.driver.session() as session:
            try:
                # Se non ci sono parametri, usa una query semplice
                if parameters is None:
                    result = session.run(query)
                else:
                    # Altrimenti usa la query con i parametri
                    result = session.run(query, parameters)
                
                # Converti il risultato in una lista di dizionari
                return [dict(record) for record in result]
            except Exception as e:
                print(f"Error executing query: {e}")
                raise

    def verify_connection(self) -> bool:
        """Verify that the connection to Neo4j is working."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                return result.single()[0] == 1
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Execute a Cypher query and return success status with message."""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                result.consume()  # Ensure query execution completes
                return True, "Query executed successfully"
        except Exception as e:
            return False, str(e)
    
    def session(self):
        """Return a new Neo4j session."""
        return self.driver.session()