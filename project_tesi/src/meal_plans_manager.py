import datetime
from enum import Enum
import re
from typing import List, Dict, Any, Optional, Tuple
import uuid
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"  # Little or no exercise
    LIGHT = "Light"          # Light exercise 1-3 times/week
    MODERATE = "Moderate"    # Moderate exercise 3-5 times/week
    ACTIVE = "Active"        # Hard exercise 6-7 times/week
    VERY_ACTIVE = "Very Active"  # Very hard exercise & physical job or training twice per day

class Goal(str, Enum):
    WEIGHT_LOSS = "Weight Loss"
    MAINTAIN = "maintain"
    WEIGHT_GAIN = "Weight Gain"

class UserParameters(BaseModel):
    name: str
    age: int
    weight: float  # in kg
    height: float  # in cm
    goal: Goal
    activity_level: ActivityLevel
    dietary_restrictions: List[str] = []
    gender: str

class CalorieCalculator:
    """Helper class to manage calorie and portion calculations"""
    
    ACTIVITY_MULTIPLIERS = {
        ActivityLevel.SEDENTARY: 1.2,
        ActivityLevel.LIGHT: 1.375,
        ActivityLevel.MODERATE: 1.55,
        ActivityLevel.ACTIVE: 1.725,
        ActivityLevel.VERY_ACTIVE: 1.9
    }
    
    GOAL_ADJUSTMENTS = {
        Goal.WEIGHT_LOSS: -500,    # Caloric deficit for weight loss
        Goal.MAINTAIN: 0,          # No adjustment for maintenance
        Goal.WEIGHT_GAIN: 500      # Caloric surplus for weight gain
    }
    
    MEAL_DISTRIBUTION = {
        "breakfast": 0.25,
        "lunch": 0.35,
        "dinner": 0.30,
        "snack": 0.10
    }
    
    @staticmethod
    def calculate_bmr(user: UserParameters) -> float:
        """
        Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation
        
        For men: BMR = (10 × weight) + (6.25 × height) - (5 × age) + 5
        For women: BMR = (10 × weight) + (6.25 × height) - (5 × age) - 161
        """
        base_bmr = (10 * user.weight) + (6.25 * user.height) - (5 * user.age)
        if user.gender.lower() == "male":
            bmr = base_bmr + 5
        else:
            bmr = base_bmr - 161
            
        return round(bmr, 2)
    
    @staticmethod
    def calculate_daily_calories(user: UserParameters) -> float:
        """
        Calculate total daily calorie needs based on:
        1. BMR
        2. Activity level
        3. Goal (weight loss, maintenance, or gain)
        """
        # Calculate BMR
        bmr = CalorieCalculator.calculate_bmr(user)
        
        # Apply activity multiplier
        activity_multiplier = CalorieCalculator.ACTIVITY_MULTIPLIERS[user.activity_level]
        tdee = bmr * activity_multiplier  # Total Daily Energy Expenditure
        
        # Adjust based on goal
        goal_adjustment = CalorieCalculator.GOAL_ADJUSTMENTS[user.goal]
        daily_calories = tdee + goal_adjustment
        
        return round(daily_calories, 2)
    
    @staticmethod
    def calculate_meal_calories(daily_calories: float) -> Dict[str, float]:
        """Calculate target calories for each meal"""
        return {
            meal_type: round(daily_calories * percentage, 2)
            for meal_type, percentage in CalorieCalculator.MEAL_DISTRIBUTION.items()
        }
    
    @staticmethod
    def calculate_portion_size(food: Dict[str, Any], target_calories: float) -> float:
        """
        Calculate portion size in grams needed to achieve target calories
        
        Args:
            food: Dictionary containing food information including calories per 100g
            target_calories: Desired calories for this portion
            
        Returns:
            Portion size in grams
        """
        calories_per_100g = food['calories']
        if calories_per_100g == 0:
            return 0
        return (target_calories / calories_per_100g) * 100

    @staticmethod
    def adjust_portions_for_meal(foods: List[Dict[str, Any]], target_calories: float) -> List[Dict[str, Any]]:
        """
        Adjust portion sizes for a list of foods to meet target calories while maintaining ratios
        """
        total_calories = sum(food['calories'] for food in foods)
        adjusted_foods = []
        
        for food in foods:
            calorie_ratio = food['calories'] / total_calories if total_calories > 0 else 0
            food_target_calories = target_calories * calorie_ratio
            
            portion_size = CalorieCalculator.calculate_portion_size(food, food_target_calories)
            
            adjusted_food = food.copy()
            adjusted_food['portion_grams'] = round(portion_size, 1)
            adjusted_food['portion_calories'] = round(food_target_calories, 1)
            
            # Calculate macronutrients for the portion
            for macro in ['proteins', 'carbohydrates', 'fats']:
                if macro in food:
                    adjusted_food[f'portion_{macro}'] = round(
                        (food[macro] * portion_size) / 100, 1
                    )
                    
            adjusted_foods.append(adjusted_food)
            
        return adjusted_foods

class MealPlanTextFormatter:
    """Handler for converting between text and JSON formats for meal plans"""
    
    @staticmethod
    def create_text_prompt(expanded_meals: Dict[str, List[Dict]], 
                        daily_calories: float,
                        meal_calories: Dict[str, float],
                        user_params: Dict[str, Any]) -> str:
        """Creates a prompt that will guide the LLM to generate a properly formatted text output"""
        
        prompt = f"""
        Create a 7-day meal plan using the provided food options and calorie targets.
        
        FORMAT YOUR RESPONSE EXACTLY AS SHOWN BELOW:
        
        === MEAL PLAN ===
        Daily Calorie Target: [total calories]
        
        --- DAY [number] ---
        
        BREAKFAST ([target calories] calories):
        - [food name] ([portion]g) ([calories] cal, [proteins]g protein, [carbs]g carbs, [fats]g fat)
        - [next food...]
        
        [Other meals...]
        
        === END MEAL PLAN ===
        
        Available foods by meal type:
        {expanded_meals}
        
        Daily calorie target: {daily_calories} kcal
        
        Meal calorie targets:
        Breakfast: {meal_calories['breakfast']} kcal
        Lunch: {meal_calories['lunch']} kcal
        Dinner: {meal_calories['dinner']} kcal
        Snack: {meal_calories['snack']} kcal
        
        User parameters:
        {user_params}
        
        Requirements:
        1. Follow the exact format shown above
        2. For each food item:
        - Calculate the portion size in grams needed to meet the meal's calorie target
        - The sum of calories from all portions must match the meal's target
        - Use this formula: portion_grams = (target_calories / calories_per_100g) * 100
        - Example: if a food has 200 cal/100g and you need 300 cal, use 150g
        3. Stay within 5% of calorie targets for each meal
        4. Main meals should follow these macronutrient ratios:
        - Carbohydrates: 25-35% of calories
        - Proteins: 20-30% of calories
        - Healthy fats: 20-35% of calories
        5. Lunch and Dinner must contain exactly one item from each category: 
        - Carbohydrates: 30-40% of meal calories
        - Proteins: 30-40% of meal calories
        - Fats: 20-30% of meal calories
        - Vegetables: remaining calories
        6. Consider dietary restrictions: {user_params.get('dietary_restrictions', [])}
        """
        
        return prompt

    @staticmethod
    def parse_text_to_json(text_output: str, meal_calories: Dict[str, float]) -> Dict[str, Any]:
        """Converts the formatted text output into the required JSON structure"""
        
        def parse_food_line(line: str) -> Optional[Dict[str, float]]:
            """Parse a single food line into a dictionary"""
            line = line.strip()
            if not line or line == '-':
                return None
                    
            pattern = r"-\s*(.*?)\s*\((\d+\.?\d*)g\)\s*\((\d+\.?\d*)\s*cal,\s*(\d+\.?\d*)g\s*protein,\s*(\d+\.?\d*)g\s*carbs,\s*(\d+\.?\d*)g\s*fat\)"
            match = re.match(pattern, line)
            if not match:
                logger.warning(f"Invalid food line format: {line}")
                return None
                    
            name, portion, calories, proteins, carbs, fats = match.groups()
            return {
                "name": name.strip(),
                "portion": float(portion),
                "calories": float(calories),
                "proteins": float(proteins),
                "carbohydrates": float(carbs),
                "fats": float(fats)
            }

        def parse_meal_section(text: str, meal_type: str) -> Optional[Dict[str, Any]]:
            """Parse a meal section into a dictionary"""
            # Extract the section between meal type header and the next meal or end of text
            pattern = f"{meal_type}\s*\(.*?\):(.*?)(?=(?:BREAKFAST|LUNCH|DINNER|SNACK|\Z))"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                return None
                
            section = match.group(1).strip()
            foods = []
            total_calories = 0
            total_proteins = 0
            total_carbs = 0
            total_fats = 0
            
            for line in section.split('\n'):
                if line.strip().startswith('-'):
                    food = parse_food_line(line)
                    if food:
                        foods.append(food)
                        total_calories += food['calories']
                        total_proteins += food['proteins']
                        total_carbs += food['carbohydrates']
                        total_fats += food['fats']
            
            if foods:
                return {
                    "foods": foods,
                    "total_calories": round(total_calories, 1),
                    "total_proteins": round(total_proteins, 1),
                    "total_carbohydrates": round(total_carbs, 1),
                    "total_fats": round(total_fats, 1)
                }
            return None

        try:
            # Extract daily calorie target
            daily_target_match = re.search(r"Daily\s+Calorie\s+Target:\s*(\d+\.?\d*)", text_output)
            if not daily_target_match:
                logger.error("Could not find daily calorie target")
                return None
            daily_calorie_target = float(daily_target_match.group(1))

            # Split into days using a more precise pattern
            day_pattern = r"---\s*DAY\s*(\d+)\s*---\s*(.*?)(?=(?:---\s*DAY|===\s*END\s*MEAL\s*PLAN|$))"
            day_matches = re.finditer(day_pattern, text_output, re.DOTALL)
            
            days = []
            for day_match in day_matches:
                day_number = int(day_match.group(1))
                day_content = day_match.group(2).strip()
                
                meals = {}
                day_total_calories = 0
                
                # Process each meal type
                for meal_type in ['BREAKFAST', 'LUNCH', 'DINNER', 'SNACK']:
                    meal_data = parse_meal_section(day_content, meal_type)
                    if meal_data:
                        meals[meal_type.lower()] = meal_data
                        day_total_calories += meal_data['total_calories']
                
                if meals:
                    days.append({
                        "day": day_number,
                        "meals": meals,
                        "total_calories": round(day_total_calories, 1)
                    })
            
            # Create final structure
            meal_plan = {
                "days": days,
                "summary": {
                    "daily_calorie_target": daily_calorie_target,
                    "meal_calorie_targets": meal_calories,
                    "generated_at": datetime.datetime.now().isoformat()
                }
            }
            
            logger.info(f"Successfully parsed {len(days)} days")
            return meal_plan

        except Exception as e:
            logger.error(f"Error parsing meal plan text: {str(e)}")
            logger.error(f"Text output: {text_output[:200]}...")
            raise

class MealPlanGraphRAG:
    def __init__(self):
        logger.info("Initializing MealPlanGraphRAG...")
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
        # Fix model name and add some temperature for variety while maintaining coherence
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
        
        logger.info("Initializing vector stores...")
        self.vector_stores = self._initialize_vector_stores()
        
        logger.info("Initializing Cypher chain...")
        self._initialize_cypher_chain()
        logger.info("Initialization complete!")

    def create_personalized_meal_plan(self, user_params: UserParameters) -> Optional[Dict[str, Any]]:
        """Creates a personalized meal plan with calculated calories and portions"""
        print("\n=== CREATING PERSONALIZED MEAL PLAN ===")
        try:
            # Calculate daily caloric needs
            daily_calories = CalorieCalculator.calculate_daily_calories(user_params)
            print(f"\nCalculated daily calories: {daily_calories}")
            
            # Calculate target calories for each meal
            meal_calories = CalorieCalculator.calculate_meal_calories(daily_calories)
            print(f"\nMeal calorie targets: {json.dumps(meal_calories, indent=2)}")
            
            # Get and expand meals
            meals_by_type = self._find_similar_users_meals(user_params)
            if not meals_by_type:
                logger.error("No similar meals found")
                print("\nDEBUG: No similar meals found in database")
                return None
                
            # Log the meals found
            print("\nDEBUG: Found meals by type:")
            print(json.dumps(meals_by_type, indent=2))
                
            expanded_meals = self._expand_foods_with_vector_search(meals_by_type)
            if not expanded_meals:
                logger.error("Failed to expand meals with vector search")
                print("\nDEBUG: Failed to expand meals with vector search")
                return None

            # Log the expanded meals
            print("\nDEBUG: Expanded meals:")
            print(json.dumps(expanded_meals, indent=2))

            # Create the text prompt with more explicit formatting requirements
            text_formatter = MealPlanTextFormatter()
            prompt = text_formatter.create_text_prompt(
                expanded_meals=expanded_meals,
                daily_calories=daily_calories,
                meal_calories=meal_calories,
                user_params=user_params.dict()
            )

            # Log the prompt being sent
            print("\nDEBUG: Sending prompt to LLM:")
            print(prompt)

            # Enhanced system message
            messages = [
                {"role": "system", "content": """You are a meal planning assistant specialized in creating structured meal plans.
                You MUST follow these rules:
                1. Always start with "=== MEAL PLAN ==="
                2. Include "Daily Calorie Target: [number]"
                3. For each day (DAY 1 through DAY 7):
                - Start with "--- DAY [number] ---"
                - Include all four meal types (BREAKFAST, LUNCH, DINNER, SNACK)
                - Format each food item exactly as: "- [food name] ([calories] cal, [proteins]g protein, [carbs]g carbs, [fats]g fat)"
                4. End with "=== END MEAL PLAN ==="
                
                Do not deviate from this format or add any additional text."""},
                {"role": "user", "content": prompt}
            ]

            print("\nSending prompt to LLM...")
            response = self.llm.invoke(messages)

            # Log the complete raw response
            print("\nDEBUG: Raw LLM response:")
            print(response.content)
            logger.debug(f"Raw LLM response:\n{response.content}")

            # Parse the text response into JSON with detailed error logging
            try:
                # First, check if the response contains the required markers
                if "=== MEAL PLAN ===" not in response.content:
                    logger.error("Response missing starting marker")
                    print("\nDEBUG: Response missing '=== MEAL PLAN ===' marker")
                    return None
                    
                if "=== END MEAL PLAN ===" not in response.content:
                    logger.error("Response missing ending marker")
                    print("\nDEBUG: Response missing '=== END MEAL PLAN ===' marker")
                    return None

                meal_plan = text_formatter.parse_text_to_json(
                    response.content,
                    meal_calories
                )
                
                if not meal_plan:
                    logger.error("Meal plan parsing returned None")
                    print("\nDEBUG: Meal plan parsing returned None")
                    return None
                    
                if not meal_plan.get('days'):
                    logger.error("Meal plan parsing produced no days")
                    print("\nDEBUG: Meal plan parsing produced no days")
                    return None
                    
                # Validate the meal plan
                if len(meal_plan['days']) != 7:
                    logger.error(f"Expected 7 days, got {len(meal_plan['days'])} days")
                    print(f"\nDEBUG: Expected 7 days, got {len(meal_plan['days'])} days")
                    return None

                logger.info(f"Successfully created meal plan with {len(meal_plan['days'])} days")
                return meal_plan
                
            except Exception as e:
                logger.error(f"Failed to parse meal plan: {str(e)}")
                logger.error(f"Raw response: {response.content[:200]}...")
                print(f"\nDEBUG: Parsing error: {str(e)}")
                print(f"First 200 chars of response: {response.content[:200]}")
                return None
                
        except Exception as e:
            print(f"\nERROR in meal plan creation: {str(e)}")
            logger.error(f"Error creating meal plan: {e}")
            return None
        
    def _initialize_cypher_chain(self):
        cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating Cypher queries for meal plan recommendation.
            Create queries that find similar meal plans and consider user parameters."""),
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

    def _perform_vector_search(self, query: str, node_type: str, k: int = 5) -> List[Dict]:
        """Versione migliorata della ricerca vettoriale"""
        if node_type not in self.vector_stores:
            return []
            
        try:
            results = self.vector_stores[node_type].similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                # Get the node's identifier from metadata
                node_identifier = doc.metadata.get("node_id")  # Neo4j typically uses this
                if not node_identifier:
                    continue
                    
                # Query to get full node properties using the internal Neo4j ID
                cypher_query = """
                MATCH (n)
                WHERE ID(n) = $node_id
                RETURN properties(n) as props
                """
                
                try:
                    node_data = self.graph.query(cypher_query, {"node_id": node_identifier})
                    
                    if node_data and len(node_data) > 0:
                        result = {
                            "content": doc.page_content,
                            "metadata": {**doc.metadata, **node_data[0]["props"]},
                            "score": float(score)
                        }
                        formatted_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error fetching node properties for ID {node_identifier}: {e}")
                    # If we can't get the full properties, at least return what we have from the vector search
                    result = {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    def _initialize_vector_stores(self) -> Dict[str, Neo4jVector]:
        """Inizializza gli store vettoriali con la corretta configurazione"""
        vector_stores = {}
        nodes_config = {
            "MealPlan": {
                "text_prop": "focus",
                "metadata_fields": ["id", "name", "type", "calories", "proteins", "carbohydrates", "fats"]
            },
            "Food": {
                "text_prop": "name",
                "metadata_fields": ["name", "calories", "proteins", "carbohydrates", "fats", "category"]
            },
            "User": {
                "text_prop": "name",
                "metadata_fields": ["name", "age", "weight", "height", "goal", "activity_level", "gender"]
            }
        }
        
        for label, config in nodes_config.items():
            try:
                logger.info(f"Initializing vector store for {label}...")
                vector_stores[label] = Neo4jVector.from_existing_index(
                    embedding=self.embeddings,
                    url=self.neo4j_url,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    index_name=f"vector_{label}",
                    node_label=label,
                    text_node_property=config["text_prop"],
                    embedding_node_property="embedding",
                    metadata_node_properties=config["metadata_fields"]  # Specifica i campi da includere nei metadata
                )
                logger.info(f"Successfully initialized vector store for {label}")
            except Exception as e:
                logger.error(f"Failed to initialize vector store for {label}: {e}")
        
        return vector_stores
    
    def _find_similar_users_meals(self, user_params: UserParameters) -> Dict[str, List[Dict]]:
        """Estrae i dati dei pasti da utenti simili nel Knowledge Graph"""
        cypher_query = """
        MATCH (u:User)
        WHERE 
            u.goal = $goal
            AND abs(u.weight - $weight) <= 20 
            AND abs(u.age - $age) <= 20
            AND u.gender = $gender
            AND u.activityLevel = $activity_level
        
        // Ottiene i loro piani alimentari e cibi
        MATCH (u)-[:FOLLOWS]->(mp:MealPlan)
            -[:HAS_MEAL_DAY]->(md:MealDay)
            -[:HAS_MEAL]->(m:Meal)
            -[:CONTAINS]->(f:Food)
        
        // Organizza i cibi per tipo di pasto
        WITH m.type as meal_type, collect(DISTINCT {
            name: f.name,
            calories: f.calories,
            carbohydrates: f.carbohydrates,
            proteins: f.proteins,
            fats: f.fats,
            category: f.category
        }) as foods
        
        // Restituisce i cibi raggruppati per tipo di pasto
        RETURN {
            meal_type: meal_type,
            foods: foods
        } as meal_data
        """
        
        try:
            print("\nExecuting Cypher query...")
            results = self.graph.query(cypher_query, {
                "goal": user_params.goal,
                "weight": user_params.weight,
                "age": user_params.age,
                "gender": user_params.gender,
                "activity_level": user_params.activity_level
            })
            print(f"Query results: {json.dumps(results, indent=2)}")
            
            # Organizza i risultati per tipo di pasto
            meals_by_type = {
                "breakfast": [],
                "lunch": [],
                "dinner": [],
                "snack": []
            }
            
            for result in results:
                meal_type = result["meal_data"]["meal_type"].lower()
                if meal_type in meals_by_type:
                    meals_by_type[meal_type].extend(result["meal_data"]["foods"])
            
            print(f"\nOrganized meals by type: {json.dumps(meals_by_type, indent=2)}")
            return meals_by_type
            
        except Exception as e:
            print(f"\nERROR in finding similar users' meals: {str(e)}")
            logger.error(f"Error finding similar users' meals: {e}")
            return {}

    def _expand_foods_with_vector_search(self, meals_by_type: Dict[str, List[Dict]], 
                                   k: int = 5) -> Dict[str, List[Dict]]:
        """Espande le opzioni di cibo usando la ricerca vettoriale"""
        print("\n=== EXPANDING FOODS WITH VECTOR SEARCH ===")
        print(f"Input meals by type: {json.dumps(meals_by_type, indent=2)}")
        
        expanded_meals = {}
        
        for meal_type, foods in meals_by_type.items():
            print(f"\nProcessing meal type: {meal_type}")
            expanded_foods = set()  # Usa set per evitare duplicati
            
            for food in foods:
                food_desc = (f"Food similar to {food['name']} with "
                        f"approximately {food['calories']} calories, "
                        f"{food['carbohydrates']}g carbs, "
                        f"{food['proteins']}g protein, "
                        f"{food['fats']}g fat")
                
                print(f"\nSearching for foods similar to: {food['name']}")
                print(f"Search query: {food_desc}")
                
                similar_foods = self._perform_vector_search(food_desc, "Food", k=k)
                print(f"Found {len(similar_foods)} similar foods")
                
                expanded_foods.add(json.dumps(food))
                
                for similar in similar_foods:
                    if similar["metadata"].get("name") != food["name"]:
                        expanded_foods.add(json.dumps(similar["metadata"]))
            
            expanded_meals[meal_type] = [json.loads(f) for f in expanded_foods]
            print(f"\nExpanded {meal_type} from {len(foods)} to {len(expanded_meals[meal_type])} foods")
        
        print(f"\nFinal expanded meals: {json.dumps(expanded_meals, indent=2)}")
        return expanded_meals

    
  