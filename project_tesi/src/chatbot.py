import os
from typing import List, Optional, Dict, Any, Tuple
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
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        
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
            7. DO NOT use LIMIT clauses unless the user explicitly asks for a limited number of results
            
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
                        # Handle relationship with additional properties
                        if "from" in target and "type" in target and target["type"] == "incoming":
                            # Incoming relationship
                            rel_str.append(f"{target['from']} →[{rel}]→ {node}")
                        elif "to" in target:
                            # Outgoing relationship with properties
                            rel_str.append(f"{node} →[{rel}]→ {target['to']}")
                            if "properties" in target:
                                props = ", ".join(target["properties"])
                                rel_str[-1] += f" (Properties: {props})"
                    elif isinstance(target, str):
                        # Simple outgoing relationship
                        rel_str.append(f"{node} →[{rel}]→ {target}")
                    elif isinstance(target, list):
                        # Multiple outgoing relationships
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

    def check_question_feasibility(self, question: str) -> Tuple[bool, str]:
        """
        Verifica se la domanda può essere risposta usando il knowledge graph,
        includendo analisi di pattern e combinazioni.
        """
        print("\nChecking question feasibility...")
        try:
            system_prompt = """You are a Neo4j graph database expert for a fitness and nutrition system.
            Your task is to determine if a Cypher query could answer a given question using our knowledge graph.
            
            Our knowledge graph contains:
            
            Nodes:
            - User: [userId, name, weight, height, age, gender, activityLevel, goal, preferences]
            - MealPlan: [mealPlanId, focus, duration]
            - MealDay: [mealDayId, name]
            - Meal: [mealId, type]
            - Food: [foodId, name, category, calories, proteins, carbohydrates, fats, knowledgeBase]
            - WorkoutPlan: [workoutPlanId, focus, duration]
            - WorkoutDay: [workoutDayId, name]
            - Exercise: [exerciseId, name, type, primaryMuscle, description]
            
            Relationships:
            - User -[FOLLOWS]-> MealPlan
            - User -[FOLLOWS]-> WorkoutPlan
            - MealPlan -[HAS_MEAL_DAY]-> MealDay
            - MealDay -[HAS_MEAL]-> Meal
            - Meal -[CONTAINS {quantity}]-> Food
            - WorkoutPlan -[HAS_WORKOUT_DAY]-> WorkoutDay
            - WorkoutDay -[HAS_EXERCISE {sets, repetitions}]-> Exercise
            
            IMPORTANT: A question is answerable if we can construct a valid Cypher query that extracts the needed information from the graph, even if that requires analyzing patterns or combinations.
            
            Types of answerable questions include:
            
            1. Simple property queries:
            Example: "List all exercises of type 'bodyweight'"
            MATCH (e:Exercise) WHERE e.type = 'bodyweight' RETURN e
            
            2. Relationship queries:
            Example: "What meal plans do users with weight loss goals follow?"
            MATCH (u:User)-[f:FOLLOWS]->(mp:MealPlan) WHERE u.goal = 'weight loss' RETURN mp
            
            3. Multi-hop path queries:
            Example: "What foods are in meals followed by users who prefer vegetarian food?"
            MATCH (u:User)-[:FOLLOWS]->(mp:MealPlan)-[:HAS_MEAL_DAY]->(md:MealDay)-[:HAS_MEAL]->(m:Meal)-[:CONTAINS]->(f:Food)
            WHERE u.preferences CONTAINS 'vegetarian' RETURN f
            
            4. Aggregation queries:
            Example: "What's the average protein content of foods in breakfast meals?"
            MATCH (m:Meal)-[:CONTAINS]->(f:Food) WHERE m.type = 'breakfast' RETURN avg(f.proteins)
            
            5. Pattern analysis and combinations:
            Example: "What are the most common food combinations in lunch meals?"
            MATCH (m:Meal)-[:CONTAINS]->(f1:Food), (m)-[:CONTAINS]->(f2:Food)
            WHERE m.type = 'lunch' AND f1.foodId < f2.foodId
            RETURN f1.name, f2.name, count(*) AS frequency
            ORDER BY frequency DESC
            
            6. Complex analytical queries:
            Example: "Which exercise and food combinations are most common in plans for weight loss?"
            MATCH (u:User)-[:FOLLOWS]->(wp:WorkoutPlan)-[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)-[:HAS_EXERCISE]->(e:Exercise),
                    (u)-[:FOLLOWS]->(mp:MealPlan)-[:HAS_MEAL_DAY]->(md:MealDay)-[:HAS_MEAL]->(m:Meal)-[:CONTAINS]->(f:Food)
            WHERE u.goal = 'weight loss'
            RETURN e.name, f.name, count(*) AS frequency
            ORDER BY frequency DESC
            
            Questions that CANNOT be answered:
            
            1. Questions requiring external domain knowledge not in the graph
            Example: "What meals should I eat to lower cholesterol?" (medical advice)
            
            2. Questions about relationships that don't exist in our schema
            Example: "Which exercises directly improve digestion of certain foods?" (no relationship between Exercise and Food)
            
            3. Questions requiring subjective judgments or ranking not based on data
            Example: "What are the most enjoyable exercises?" (subjective criteria not in our data)
            
            IMPORTANT: For pattern analysis questions, as long as the data exists in the graph and can be queried, the question IS answerable, even if it requires complex aggregations or counting of co-occurrences.
            
            Your answer MUST start with:
            - "YES" if the question can be answered with data in our graph
            - "NO" if the question cannot be answered
            
            Then briefly explain how it could be answered with a Cypher-like query or why it cannot be answered.
            """
            
            feasibility_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nCan this question be answered using our knowledge graph data? Answer YES or NO and explain:"
            response = self.llm.invoke(feasibility_prompt)
            
            response_text = response.content
            can_answer = response_text.upper().startswith(("YES", "SI"))
            explanation = response_text.split("\n", 1)[1] if "\n" in response_text else response_text
            
            print(f"Feasibility check result: {'Can answer' if can_answer else 'Cannot answer'}")
            print(f"FEASIBILITY: {explanation}")
            
            return can_answer, explanation
            
        except Exception as e:
            print(f"Error in feasibility check: {e}")
            return False, f"Error checking feasibility: {str(e)}"
        
    def validate_response(self, question: str, response: str, graph_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Step 4: Valida la risposta generata
        """
        print("\nValidating response...")
        try:
            validation_prompt = f"""
            You are a validator for responses generated from a knowledge graph about fitness and nutrition.
            
            Original Question: {question}
            Generated Response: {response}
            
            Graph Data Used:
            {graph_data}
            
            Please validate the response based on:
            1. Accuracy - Does it match the graph data?
            2. Completeness - Does it address all aspects of the question?
            3. Relevance - Is the information relevant to the question?
            4. Consistency - Are there any contradictions?

            Provide your assessment with VALID or INVALID, if its INVALID try to make a answer based on the context.
            """
            
            validation_result = self.llm.invoke(validation_prompt)
            result_text = validation_result.content
            
            is_valid = result_text.upper().startswith("VALID")
            explanation = result_text.split("\n", 1)[1] if "\n" in result_text else result_text
            
            return is_valid, explanation
            
        except Exception as e:
            print(f"Error in response validation: {e}")
            return False, f"Validation error: {str(e)}"

    def get_response(self, user_query: str) -> str:
        """Generate a response based on the user query using Neo4j and vector search."""
        print(f"\nProcessing query: {user_query}")
        try:
            # Step 1: Verifica la fattibilità
            can_answer, feasibility_explanation = self.check_question_feasibility(user_query)
            if not can_answer:
                return f"I cannot provide a complete answer to this question. {feasibility_explanation}"
            
            # Step 2: Interroga il grafo (usando il codice esistente)
            print("\nExecuting GraphCypherQAChain...")
            chain_response = self.cypher_chain({
                "query": user_query,
                "examples": self._format_examples_for_prompt(),
                "schema": self._format_schema_for_prompt(),
                "history": []
            })
            cleaned_chain_response = clean_neo4j_response(chain_response)
            print("\nCypher chain execution complete")
            
            print("\nPerforming vector search...")
            vector_results = self.perform_vector_search(user_query)
            print("Vector search complete")
            
            # Step 3: Genera la risposta (usando il codice esistente)
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
            
            initial_response = self.llm.invoke(final_prompt)
            initial_response_text = initial_response.content
            
            # Step 4: Valida la risposta
            is_valid, validation_explanation = self.validate_response(
                user_query, 
                initial_response_text, 
                context
            )
            
            if not is_valid:
                print(f"Response validation failed: {validation_explanation}")
                improved_prompt = f"""
                {final_prompt}
                
                The previous response was invalid because: {validation_explanation}
                Please generate an improved response addressing these issues.
                """
                improved_response = self.llm.invoke(improved_prompt)
                return improved_response.content  # Here we properly extract the content
            
            return initial_response_text  # This is already a string
            
        except Exception as e:
            error_msg = f"An error occurred while processing your request: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg

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