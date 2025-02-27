from enum import Enum
import os
import re
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel
import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"
    LIGHT = "Light"
    MODERATE = "Moderate"
    ACTIVE = "Active"
    VERY_ACTIVE = "Very Active"

class Goal(str, Enum):
    WEIGHT_LOSS = "Weight Loss"
    WEIGHT_GAIN = "Weight Gain"
    MAINTENANCE = "Maintenance"

class UserParameters(BaseModel):
    username: str
    name: str
    age: int
    weight: float  # in kg
    height: float  # in cm
    goal: Goal
    activity_level: ActivityLevel
    preferences: List[str] = []
    gender: str

class WorkoutPlanTextFormatter:
    """Handler for converting between text and JSON formats for workout plans"""
    
    @staticmethod
    def create_text_prompt(categorized_exercises: Dict[str, List[Dict]], 
                        user_params: Dict[str, Any]) -> str:
        """Crea un prompt per l'LLM per generare un piano di allenamento con esercizi già categorizzati"""
        
        # Formatta gli esercizi categorizzati in modo leggibile per l'LLM
        formatted_exercises = ""
        for muscle_category, exercises in categorized_exercises.items():
            formatted_exercises += f"\n## {muscle_category.upper()} EXERCISES:\n"
            for ex in exercises:
                name = ex.get("name", "Unnamed exercise")
                desc = ex.get("description", "No description available")
                formatted_exercises += f"- {name}: {desc[:100]}...\n"
        
        prompt = f"""
        Create a complete multi-day workout plan (3-5 days) based on the user's parameters and available exercises.
        
        IMPORTANT GUIDELINES:
        - Each day should focus on AT MOST 3 DIFFERENT MUSCLE GROUPS. This is critical!
        - Use the provided categorized exercises which are already grouped by muscle target.
        - Create workouts that target complementary muscles on the same day.
        - Typical pairings: Chest+Triceps, Back+Biceps, Legs+Core, Shoulders+Arms.
        - Never create full-body workouts that target all major muscle groups in one session.
        
        FORMAT YOUR RESPONSE EXACTLY AS SHOWN BELOW YOU MUST INCLUDE Goal and Activity Level:
        
        === WORKOUT PLAN ===
        Goal: {user_params['goal']}
        Activity Level: {user_params['activity_level']}
        
        DAY 1 (Target Muscles: [list max 3 specific muscle groups]):
        - [exercise name]  
        Sets: [number], Reps: [number], Rest: [seconds]s  
        Notes: [form cues or specific instructions]
        
        - [Another exercise]  
        Sets: [number], Reps: [number], Rest: [seconds]s  
        Notes: [form cues or specific instructions]
        
        DAY 2 (Target Muscles: [list max 3 different specific muscle groups]):
        - [exercise name]  
        Sets: [number], Reps: [number], Rest: [seconds]s  
        Notes: [form cues or specific instructions]
        
        [Continue for all days, 3-5 days total]
        === END WORKOUT PLAN ===
        
        Available categorized exercises:
        {formatted_exercises}
        
        User parameters:
        {user_params}
        
        IMPORTANT: Each workout day MUST specify exactly which muscle groups (maximum 3) are being targeted.
        Each day should have 4-6 exercises that collectively target the specified muscle groups.
        """
        
        return prompt

class WorkoutPlanGraphRAG:
    def __init__(self):
        logger.info("Initializing WorkoutPlanGraphRAG...")
        load_dotenv()
        
        self.neo4j_url = os.getenv("NEO4J_URL")
        self.neo4j_username = os.getenv("NEO4J_USERNAME")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        logger.info("Connecting to Neo4j...")
        self.graph = Neo4jGraph(
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password
        )
        
        logger.info("Initializing OpenAI components...")
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        
        logger.info("Initializing vector stores...")
        self.vector_stores = self._initialize_vector_stores()
        
        logger.info("Initializing Cypher chain...")
        self._initialize_cypher_chain()
        logger.info("Initialization complete!")

    def _initialize_vector_stores(self) -> Dict[str, Any]:
        """Initializes vector stores for workout-related nodes"""
        vector_stores = {}
        nodes_config = {
            "WorkoutPlan": {
                "text_key": "description",
                "metadata_fields": ["id", "description"]
            },
            "Exercise": {
                "text_key": "description",
                "metadata_fields": ["name", "description"]
            },
            "User": {
                "text_key": "name",
                "metadata_fields": ["username", "name", "age", "weight", "height", "goal", "activityLevel", "gender"]
            }
        }
        
        for label, config in nodes_config.items():
            try:
                vector_stores[label] = Neo4jVector.from_existing_index(
                    embedding=self.embeddings,
                    url=self.neo4j_url,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    index_name=f"vector_{label}",
                    node_label=label,
                    text_node_property=config["text_key"],
                    embedding_node_property="embedding",
                    metadata_node_properties=config["metadata_fields"]  # Changed from properties to metadata_node_properties
                )
            except Exception as e:
                logger.error(f"Failed to initialize vector store for {label}: {e}")
        
        return vector_stores

    def create_personalized_workout_plan(self, user_params: Union[UserParameters, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Creates a personalized workout plan based on user parameters"""
        try:
            # Convert user_params to dict if it's a Pydantic model, or use as is if already a dict
            user_params_dict = user_params.dict() if hasattr(user_params, 'dict') else user_params
            
            # Find similar workout plans and exercises, già categorizzati per gruppo muscolare
            categorized_workouts = self._find_similar_users_workouts(user_params_dict)
            if not categorized_workouts:
                logger.error("No similar workouts found")
                return None
                
            # Expand exercises with vector search while maintaining muscle categories
            expanded_categorized_exercises = self._expand_exercises_with_vector_search(categorized_workouts)
            if not expanded_categorized_exercises:
                logger.error("Failed to expand exercises with vector search")
                return None

            # Store expanded_exercises for later use in _get_exercise_description
            self._last_expanded_exercises = expanded_categorized_exercises

            # Create and format the workout plan
            text_formatter = WorkoutPlanTextFormatter()
            prompt = text_formatter.create_text_prompt(
                categorized_exercises=expanded_categorized_exercises,
                user_params=user_params_dict
            )

            # Get LLM response and parse
            messages = [
                {"role": "system", "content": "You are a professional fitness trainer specialized in creating structured multi-day workout plans (3-5 days)."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.invoke(messages)
            response_content = response.content
            
            # Check if this is just the raw workout plan (like your example)
            if not re.search(r"===\s*WORKOUT\s*PLAN\s*===", response_content):
                # If it's just the raw plan without headers, add them for parsing
                response_content = f"=== WORKOUT PLAN ===\nGoal: {user_params_dict.get('goal', 'Custom')}\nActivity Level: {user_params_dict.get('activity_level', 'Custom')}\n\n{response_content}"
                if not response_content.strip().endswith("=== END WORKOUT PLAN ==="):
                    response_content += "\n=== END WORKOUT PLAN ==="
            
            # Parse the response and validate
            workout_plan = self._parse_workout_plan(response_content)
            if workout_plan:
                return workout_plan
            else:
                # Try direct parsing if standard parsing fails
                workout_plan = self._direct_parse_workout(response_content)
                return workout_plan

        except Exception as e:
            logger.error(f"Error creating workout plan: {e}")
            return None
        
    def _direct_parse_workout(self, text_output: str) -> Optional[Dict[str, Any]]:
        """Direct parser for when regular parsing fails - specifically for the format shown in your example"""
        try:
            days = []
            day_pattern = r"DAY\s+(\d+):(.*?)(?=DAY\s+\d+:|===\s*END\s*WORKOUT\s*PLAN\s*===|\Z)"
            day_matches = list(re.finditer(day_pattern, text_output, re.DOTALL | re.IGNORECASE))
            
            if not day_matches:
                # If no day pattern found, try for a plan without day markers
                exercise_pattern = r"-\s*(.*?)\n\s*Sets:\s*(\d+),\s*Reps:\s*([\d-]+),\s*Rest:\s*([\d]+)s\n\s*Notes:\s*(.*?)(?=(?:\n\s*-|\Z))"
                exercise_matches = list(re.finditer(exercise_pattern, text_output, re.DOTALL))
                
                if exercise_matches:
                    # Create a single day with all exercises
                    exercises = []
                    for ex_match in exercise_matches:
                        name = ex_match.group(1).strip()
                        sets = int(ex_match.group(2))
                        reps_str = ex_match.group(3)
                        rest = int(ex_match.group(4))
                        notes = ex_match.group(5).strip()
                        
                        # Handle ranges in reps
                        if "-" in reps_str:
                            reps = reps_str  # Keep as string
                        else:
                            reps = int(reps_str)
                        
                        exercise_description = self._get_exercise_description(name)
                        
                        exercise = {
                            "name": name,
                            "sets": sets,
                            "reps": reps,
                            "rest_seconds": rest,
                            "notes": notes,
                            "description": exercise_description,
                            "target_muscles": self._infer_target_muscles(name, exercise_description)
                        }
                        exercises.append(exercise)
                    
                    if exercises:
                        day_data = {
                            "day": 1,
                            "muscle_groups": self._identify_muscle_groups(exercises),
                            "exercises": exercises
                        }
                        days.append(day_data)
                
                if not days:
                    logger.error("Failed to parse workout plan in direct parsing")
                    return None
            else:
                # Process each day
                for day_match in day_matches:
                    day_num = int(day_match.group(1))
                    day_content = day_match.group(2).strip()
                    
                    exercise_pattern = r"-\s*(.*?)\n\s*Sets:\s*(\d+),\s*Reps:\s*([\d-]+),\s*Rest:\s*([\d]+)s\n\s*Notes:\s*(.*?)(?=(?:\n\s*-|\Z))"
                    exercise_matches = list(re.finditer(exercise_pattern, day_content, re.DOTALL))
                    
                    day_exercises = []
                    for ex_match in exercise_matches:
                        name = ex_match.group(1).strip()
                        sets = int(ex_match.group(2))
                        reps_str = ex_match.group(3)
                        rest = int(ex_match.group(4))
                        notes = ex_match.group(5).strip()
                        
                        # Handle ranges in reps
                        if "-" in reps_str:
                            reps = reps_str  # Keep as string
                        else:
                            reps = int(reps_str)
                        
                        exercise_description = self._get_exercise_description(name)
                        
                        exercise = {
                            "name": name,
                            "sets": sets,
                            "reps": reps,
                            "rest_seconds": rest,
                            "notes": notes,
                            "description": exercise_description,
                            "target_muscles": self._infer_target_muscles(name, exercise_description)
                        }
                        day_exercises.append(exercise)
                    
                    if day_exercises:
                        day_data = {
                            "day": day_num,
                            "muscle_groups": self._identify_muscle_groups(day_exercises),
                            "exercises": day_exercises
                        }
                        days.append(day_data)
            
            # Create workout plan
            if days:
                workout_plan = {
                    "summary": {
                        "goal": "Custom Workout",  # Default when direct parsing
                        "fitness_level": "Custom",
                        "generated_at": datetime.datetime.now().isoformat(),
                        "days_count": len(days)
                    },
                    "days": days
                }
                return workout_plan
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in direct parsing: {e}")
            return None
        
    def _find_similar_users_workouts(self, user_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Estrae dati di allenamento da utenti simili nel Knowledge Graph, già divisi per categoria muscolare"""
        logger.info(f"Ricerca di allenamenti con parametri: {user_data}")
        
        cypher_query = """
        MATCH (u:User)
        WHERE 
            u.goal = $goal
            AND u.activityLevel = $activity_level
            AND abs(u.weight - $weight) <= 20
            AND abs(u.age - $age) <= 10
        
        MATCH (u)-[:FOLLOWS]->(wp:WorkoutPlan)
            -[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)
            -[:HAS_EXERCISE]->(e:Exercise)
        
        // Cerca anche di ottenere le categorie muscolari dagli esercizi
        OPTIONAL MATCH (e)-[:TARGETS]->(m:MuscleGroup)
        
        RETURN 
            CASE 
                WHEN m IS NOT NULL THEN m.name 
                ELSE 'Other' 
            END AS muscle_category,
            e.name AS exercise_name, 
            COLLECT(DISTINCT e.description) AS descriptions
        """
        
        try:
            results = self.graph.query(cypher_query, {
                "goal": user_data.get('goal'), 
                "activity_level": user_data.get('activity_level'),
                "weight": float(user_data.get('weight', 0)),
                "age": int(user_data.get('age', 0))
            })
            
            # Raggruppa per categoria muscolare
            workouts_by_muscle = {}
            for result in results:
                muscle_category = result["muscle_category"]
                exercise_name = result["exercise_name"]
                
                if muscle_category not in workouts_by_muscle:
                    workouts_by_muscle[muscle_category] = {}
                    
                if exercise_name not in workouts_by_muscle[muscle_category]:
                    workouts_by_muscle[muscle_category][exercise_name] = []
                    
                workouts_by_muscle[muscle_category][exercise_name].extend([
                    {"name": exercise_name, "description": desc, "muscle_category": muscle_category} 
                    for desc in result["descriptions"]
                ])
            
            # Trasforma in formato finale
            categorized_workouts = {}
            for muscle, exercises in workouts_by_muscle.items():
                categorized_workouts[muscle] = []
                for exercise_name, exercise_details in exercises.items():
                    categorized_workouts[muscle].extend(exercise_details)
            
            logger.info(f"Trovati esercizi per {len(categorized_workouts)} categorie muscolari")
            return categorized_workouts
                
        except Exception as e:
            logger.error(f"Errore nella ricerca di allenamenti di utenti simili: {e}")
            return {}
        
    def _expand_exercises_with_vector_search(self, categorized_workouts: Dict[str, List[Dict]], k: int = 5) -> Dict[str, List[Dict]]:
        """Espande le opzioni di esercizi usando la ricerca vettoriale mantenendo le categorie muscolari"""
        expanded_categorized_workouts = {}
        
        for muscle_category, exercises in categorized_workouts.items():
            expanded_exercises = set()
            
            for exercise in exercises:
                exercise_desc = f"Exercise for {muscle_category} similar to {exercise['name']}"
                
                # Cerca esercizi simili che targettizzano lo stesso gruppo muscolare
                similar_exercises = self._perform_vector_search_with_filter(
                    exercise_desc, 
                    "Exercise", 
                    filter_category=muscle_category,
                    k=k
                )
                
                expanded_exercises.add(json.dumps(exercise))
                for similar in similar_exercises:
                    if similar["metadata"].get("name") != exercise["name"]:
                        # Aggiungi la categoria muscolare al metadata
                        metadata = similar["metadata"]
                        metadata["muscle_category"] = muscle_category
                        expanded_exercises.add(json.dumps(metadata))
            
            expanded_categorized_workouts[muscle_category] = [json.loads(e) for e in expanded_exercises]
        
        return expanded_categorized_workouts

    def _perform_vector_search_with_filter(self, query: str, node_type: str, filter_category: str = None, k: int = 5) -> List[Dict]:
        """Esegue una ricerca vettoriale per esercizi simili con filtro opzionale per categoria muscolare"""
        if node_type not in self.vector_stores:
            return []
            
        try:
            # Modifica la query Cypher per filtrare per categoria muscolare
            custom_where_clause = ""
            if filter_category and filter_category != "Other":
                custom_where_clause = f"""
                MATCH (node)-[:TARGETS]->(:MuscleGroup {{name: "{filter_category}"}})
                """
            
            results = self.vector_stores[node_type].similarity_search_with_score(
                query, 
                k=k,
                custom_where_clause=custom_where_clause
            )
            
            formatted_results = []
            for doc, score in results:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Errore nella ricerca vettoriale: {e}")
            return []
        
    def _initialize_cypher_chain(self):
        """Initializes the Cypher chain for graph queries"""
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating Cypher queries for workout plan recommendation.
            Create queries that find similar workout plans and consider user parameters."""),
            ("human", "{query}")
        ])
        
        self.cypher_chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=self.llm,
            qa_llm=self.llm,
            validate_cypher=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True
        )

    def _parse_workout_plan(self, text_output: str) -> Optional[Dict[str, Any]]:
        """Analizza l'output LLM in un piano di allenamento strutturato con più giorni e categorie muscolari specifiche"""
        try:
            # Extract overall plan information
            goal_match = re.search(r"Goal:\s*(\w+(?:\s+\w+)*)", text_output)
            activity_level_match = re.search(r"Activity Level:\s*(\w+(?:\s+\w+)*)", text_output)
            
            if not goal_match:
                logger.warning("Could not find goal information, using defaults")
                goal = "Default Goal"
            else:
                goal = goal_match.group(1)
                
            activity_level = activity_level_match.group(1) if activity_level_match else "Unknown"

            # Pattern più flessibile per analizzare le sezioni giornaliere, includendo i gruppi muscolari target
            day_pattern = r"DAY\s+(\d+)(?:\s*\(Target Muscles:\s*([^)]+)\))?\s*(?:\s*:\s*(?:[^-\n]*))?(.*?)(?=DAY\s+\d+|===\s*END\s*WORKOUT\s*PLAN\s*===|\Z)"
            day_matches = list(re.finditer(day_pattern, text_output, re.DOTALL | re.IGNORECASE))
            
            if not day_matches:
                logger.error("No day sections found in the workout plan")
                return None
                
            days = []
            for day_match in day_matches:
                day_num = int(day_match.group(1))
                
                # Estrai i gruppi muscolari target dal pattern (nuova funzionalità)
                target_muscles_text = day_match.group(2).strip() if day_match.group(2) else ""
                target_muscles = [muscle.strip() for muscle in target_muscles_text.split(",")] if target_muscles_text else []
                
                day_content = day_match.group(3).strip()
                
                # Il resto del codice per l'analisi degli esercizi rimane simile...
                # [Codice precedente per exercise_pattern, ecc.]
                
                exercise_pattern = (
                    r"-\s*(.*?)\s*\n?"  # Exercise name, potentially followed by newline
                    r"\s*Sets:\s*(\d+),\s*Reps:\s*([\d-]+(?:\s*s\s*hold)?),\s*Rest:\s*(\d+)s\s*\n"
                    r"\s*Notes:\s*(.*?)(?=\s*-|\Z)"  # Notes section until next exercise or end
                )

                # Fallback pattern for cases where exercises are on the same line as the day header
                fallback_pattern = (
                    r"-\s*(.*?)\s+"  # Exercise name
                    r"Sets:\s*(\d+),\s*Reps:\s*([\d-]+(?:\s*s\s*hold)?),\s*Rest:\s*(\d+)s\s+"
                    r"Notes:\s*(.*?)(?=\s+-|\Z)"  # Notes section until next exercise or end
                )

                exercise_matches = list(re.finditer(exercise_pattern, day_content, re.DOTALL))
                
                # If no exercises found with primary pattern, try fallback pattern
                if not exercise_matches:
                    exercise_matches = list(re.finditer(fallback_pattern, day_content, re.DOTALL))
                    
                day_exercises = []
                for ex_match in exercise_matches:
                    name = ex_match.group(1).strip()
                    sets = int(ex_match.group(2))
                    reps = ex_match.group(3).strip()
                    rest = int(ex_match.group(4))
                    notes = ex_match.group(5).strip()
                    
                    # Get the description from the expanded exercises data
                    exercise_description = self._get_exercise_description(name)
                    
                    # Non serve più inferire i muscoli target, poiché sono già specificati
                    exercise = {
                    "name": name,
                    "sets": sets,
                    "reps": reps,
                    "rest_seconds": rest,
                    "notes": notes,
                    "description": exercise_description,
                    "target_muscles": self._infer_target_muscles(name, exercise_description)  # Add this line
                }
                    day_exercises.append(exercise)
                
                if day_exercises:  # Only add days that have exercises
                    # Usa i muscoli target estratti dal pattern
                    day_data = {
                        "day": day_num,
                        "muscle_groups": target_muscles if target_muscles else ["General"],
                        "exercises": day_exercises
                    }
                    days.append(day_data)
                else:
                    logger.warning(f"No exercises found for day {day_num}, skipping")
            
            # Create final structure
            if days:
                workout_plan = {
                    "summary": {
                        "goal": goal,
                        "fitness_level": activity_level,
                        "generated_at": datetime.datetime.now().isoformat(),
                        "days_count": len(days)
                    },
                    "days": days
                }
                    
                return workout_plan
            else:
                logger.error("No days with exercises found in the workout plan")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing workout plan: {e}")
            logger.error(f"Text output: {text_output[:200]}...")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        
    def _infer_target_muscles(self, exercise_name, description):
        """Infer target muscles from exercise name and description"""
        muscle_targets = {
            "triceps": ["tricep", "extension", "close-grip"],
            "chest": ["chest", "bench press", "push-up", "dumbbell press"],
            "back": ["back", "row", "pull-up", "deadlift"],
            "shoulders": ["shoulder", "overhead", "lateral raise"],
            "biceps": ["bicep", "curl"],
            "legs": ["leg", "squat", "lunge", "calf"],
            "abs": ["ab", "core", "crunch", "plank"]
        }
        
        text = (exercise_name + " " + description).lower()
        targets = []
        
        for muscle, keywords in muscle_targets.items():
            if any(keyword in text for keyword in keywords):
                targets.append(muscle)
        
        return targets if targets else ["general"]

    def _identify_muscle_groups(self, exercises, max_groups=3):
        """Identify muscle groups based on exercise names and descriptions, limiting to max_groups"""
        muscle_group_keywords = {
            "Chest": ["chest", "pectoral", "bench press", "push-up", "fly"],
            "Back": ["back", "lat", "row", "pull-up", "pulldown"],
            "Legs": ["leg", "squat", "lunge", "hamstring", "quad", "calf"],
            "Shoulders": ["shoulder", "deltoid", "press", "lateral raise"],
            "Arms": ["arm", "bicep", "tricep", "curl"],
            "Core": ["core", "abs", "abdominal", "plank"]
        }
        
        # Count occurrences of each muscle group
        group_counts = {group: 0 for group in muscle_group_keywords}
        
        for exercise in exercises:
            text_to_check = (exercise.get("name", "") + " " + 
                            exercise.get("description", "") + " " + 
                            exercise.get("notes", "")).lower()
            
            for group, keywords in muscle_group_keywords.items():
                if any(keyword in text_to_check for keyword in keywords):
                    group_counts[group] += 1
        
        # Select top muscle groups based on count
        top_groups = sorted([(group, count) for group, count in group_counts.items() if count > 0], 
                            key=lambda x: x[1], reverse=True)[:max_groups]
        
        return [group for group, _ in top_groups] if top_groups else ["General"]

    def _get_exercise_description(self, exercise_name):
        """Fetch the description for an exercise based on its name"""
        try:
            # First try direct lookup in Neo4j
            cypher_query = """
            MATCH (e:Exercise)
            WHERE e.name = $name
            RETURN e.description AS description
            LIMIT 1
            """
            
            results = self.graph.query(cypher_query, {"name": exercise_name})
            
            if results and results[0].get("description"):
                return results[0]["description"]
            
            # If not found, try to get it from the expanded exercises used during plan creation
            # This would require storing expanded_exercises as an instance variable during create_personalized_workout_plan
            if hasattr(self, '_last_expanded_exercises'):
                for _, exercises in self._last_expanded_exercises.items():
                    for exercise in exercises:
                        if exercise.get("name") == exercise_name and exercise.get("description"):
                            return exercise["description"]
            
            return "No description available"
            
        except Exception as e:
            logger.error(f"Error getting exercise description: {e}")
            return "Description unavailable"
    
    def _create_vector_store(self, label: str, config: Dict[str, Any]) -> Any:
        """Creates a vector store for a specific node type"""
        try:
            return Neo4jVector.from_existing_index(
                embedding=self.embeddings,
                url=self.neo4j_url,
                username=self.neo4j_username,
                password=self.neo4j_password,
                index_name=f"vector_{label}",
                node_label=label,
                text_node_property=config["text_prop"],
                embedding_node_property="embedding",
                metadata_node_properties=config["metadata_fields"]
            )
        except Exception as e:
            logger.error(f"Error creating vector store for {label}: {e}")
            raise


#aggiungere parte di presa delle istruzioni, revisionare il dataset
  

