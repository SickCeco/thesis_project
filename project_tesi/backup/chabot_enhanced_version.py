import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from cypher_examples import CypherExamples

class GraphQuery(BaseModel):
    query: str = Field(
        ...,
        description="Primary Cypher query to be executed against the Neo4j database."
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Break down the main query into smaller, focused sub-queries."
    )

def clean_neo4j_response(response: Any) -> Any:
    """Remove embedding vectors and clean the Neo4j response."""
    if isinstance(response, dict):
        return {k: clean_neo4j_response(v) for k, v in response.items() if k != 'embedding'}
    elif isinstance(response, list):
        return [clean_neo4j_response(item) for item in response]
    return response

def format_vector_result(doc) -> str:
    """Format a vector search result, excluding embeddings."""
    try:
        content = doc.page_content
        metadata = {k: v for k, v in doc.metadata.items() if k != 'embedding'}
        return f"Content: {content}, Relevant properties: {metadata}"
    except Exception as e:
        print(f"Error formatting vector result: {e}")
        return str(doc)

def tool_example_to_messages(example: Dict) -> List[BaseMessage]:
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    
    for tool_call in example["tool_calls"]:
        query_obj = GraphQuery(
            query=tool_call["query"],
            sub_queries=tool_call["sub_queries"]
        )
        
        openai_tool_calls.append({
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {
                "name": "GraphQuery",
                "arguments": query_obj.model_dump_json()
            },
        })
    
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    
    tool_outputs = example.get("tool_outputs") or ["Query generated successfully."] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    
    return messages

class GraphRAGBot:
    def __init__(self):
        print("Initializing GraphRAGBot...")
        load_dotenv()
        
        self.neo4j_url = os.getenv("NEO4J_URL")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        print("Connecting to Neo4j...")
        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        
        print("Loading Cypher examples...")
        self.cypher_examples_instance = CypherExamples()  # Create instance
        self.cypher_examples = self.cypher_examples_instance.get_all_examples()
        self.example_conversations = self.cypher_examples_instance.get_example_conversations()
        
        print("Initializing embeddings and LLM...")
        self.embeddings = OpenAIEmbeddings()
        self.vector_stores = self._initialize_vector_stores()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        
        print("Setting up Cypher chain...")
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at writing Cypher queries for Neo4j. 
            
            The graph has the following schema:
            {schema}
            
            Use the following examples to understand how to structure your queries:
            
            Examples:
            {examples}
            
            When generating a query:
            1. Review the schema to understand available nodes, properties, and relationships
            2. Look at the similar examples provided
            3. Follow the same structure and patterns
            4. Use the correct relationship directions as specified in the schema
            5. Include all necessary relationships and properties
            6. Make sure property names match exactly with the schema
            
            Current question: {query}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        
        examples_str = self._format_examples_for_prompt()
        schema_str = self._format_schema_for_prompt()
        
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,
            qa_llm=self.llm,
            validate_cypher=True,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt
        )
        
        print("GraphRAGBot initialization complete!")

    def _format_schema_for_prompt(self) -> str:
        """Format the schema information into a string for the prompt."""
        schema = self.cypher_examples_instance.get_schema_info()  # Use instance here
        formatted_schema = ["Graph Structure:"]
        
        for node, info in schema["nodes"].items():
            node_str = [f"\nNode: {node}"]
            node_str.append("Properties: " + ", ".join(info["properties"]))
            
            if "relationships" in info:
                rel_str = []
                for rel, target in info["relationships"].items():
                    if isinstance(target, dict):
                        direction = "←" if target["type"] == "incoming" else "→"
                        rel_str.append(f"{target['from']} {direction}[{rel}]→ {node}")
                    elif isinstance(target, str):
                        rel_str.append(f"{node} →[{rel}]→ {target}")
                    elif isinstance(target, list):
                        for t in target:
                            rel_str.append(f"{node} →[{rel}]→ {t}")
                            
                node_str.append("Relationships:\n  " + "\n  ".join(rel_str))
            
            formatted_schema.extend(node_str)
        return "\n".join(formatted_schema)
    
    def _format_examples_for_prompt(self) -> str:
        """Format the example conversations into a string for the prompt."""
        formatted_examples = []
        for ex in self.example_conversations:
            example_str = f"""
            Question: {ex['input']}
            Query: {ex['tool_calls'][0]['query']}
            Steps:
            {chr(10).join('- ' + step for step in ex['tool_calls'][0]['sub_queries'])}
            """
            formatted_examples.append(example_str)
        
        return "\n".join(formatted_examples)
    
    def get_response(self, user_query: str) -> str:
        """Generate a response based on the user query using Neo4j and vector search."""
        print(f"\nProcessing query: {user_query}")
        try:
            print("\nExecuting GraphCypherQAChain...")
            # Include examples in the chain input and use 'query' as the key
            chain_response = self.cypher_chain({
            "query": user_query,
            "examples": self._format_examples_for_prompt(),
            "schema": self._format_schema_for_prompt(),
            "history": []
        })
            # Clean the response to remove embeddings
            cleaned_chain_response = clean_neo4j_response(chain_response)
            print("\nCypher chain execution complete")
            
            print("\nPerforming vector search...")
            vector_results = self.perform_vector_search(user_query)
            print("Vector search complete")
            
            context = f"""
            Graph Query Results:
            {cleaned_chain_response}
            
            Additional Context from Vector Search:
            {' '.join(vector_results[:3])}
            """
            
            print("\nGenerating final response...")
            final_prompt = f"""
            Based on the following information, provide a clear and concise answer to: {user_query}
            
            Context:
            {context}
            
            Focus on providing relevant details while maintaining a natural conversational tone.
            """
            
            final_response = self.llm.invoke(final_prompt)
            print("Final response generated")
            
            return final_response.content
            
        except Exception as e:
            error_msg = f"An error occurred while processing your request: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg
   
    def _initialize_vector_stores(self):
        print("Initializing vector stores...")
        vector_stores = {}
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
                print(f"Loading vector store for {label}...")
                vector_stores[label] = Neo4jVector.from_existing_index(
                    embedding=self.embeddings,
                    url=self.neo4j_url,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    index_name=f"vector_{label}",
                    node_label=label,
                    text_node_property=text_property,
                    embedding_node_property="embedding"
                )
                print(f"Vector store for {label} successfully loaded")
            except Exception as e:
                print(f"Warning: Unable to load vector store for {label}: {e}")
        
        return vector_stores

    def perform_vector_search(self, query: str, max_results: int = 2) -> List[str]:
        """Perform vector search across all stores with limited results."""
        print(f"\nPerforming vector search for: {query}")
        vector_results = []
        for label, store in self.vector_stores.items():
            try:
                print(f"Searching vector store for {label}...")
                results = store.similarity_search(query, k=max_results)
                formatted_results = [format_vector_result(doc) for doc in results]
                vector_results.extend(formatted_results[:max_results])
                print(f"Found {len(formatted_results)} results for {label}")
                print(formatted_results)
            except Exception as e:
                print(f"Warning: Error in vector search for {label}: {e}")
        
        final_results = vector_results[:max_results * 2]
        print(f"Total vector results selected: {len(final_results)}")
        return final_results

def chat_loop():
    print("Initializing improved chatbot...")
    bot = GraphRAGBot()
    print("ChatBot ready! (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            break
            
        try:
            print("\nProcessing response...")
            response = bot.get_response(user_input)
            print("\nResponse:", response)
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    chat_loop()