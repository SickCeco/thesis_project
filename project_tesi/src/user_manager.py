import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from logging import logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self, neo4j_connector):
        self.neo4j = neo4j_connector

    def create_user_node(self, user_data: Dict) -> Tuple[bool, str]:
        """Create a new user node in the knowledge graph."""
        try:
            # Validate required fields
            if 'username' not in user_data or 'password' not in user_data:
                return False, "Username and password are required"
                
            # Check for existing user
            check_query = """
            MATCH (u:User)
            WHERE u.Username = $username
            RETURN count(u) as count
            """
            
            check_result = self.neo4j.query(check_query, 
                {"username": user_data['username']})
            
            if check_result and check_result[0]['count'] > 0:
                return False, "Username already exists"

            # Standardize field names
            standardized_data = {
                'Username': user_data['username'],
                'Password': user_data['password'],
                'UserID': f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'Name': user_data.get('Name', ''),
                'Age': user_data.get('Age', 0),
                'Sex': user_data.get('Sex', ''),
                'Weight': user_data.get('Weight', 0.0),
                'Height': user_data.get('Height', 0.0),
                'ActivityLevel': user_data.get('Activity_level', 'Sedentary'),
                'Goal': user_data.get('Goal', ''),
                'Preference': list(user_data.get('Preference', [])),
                'DailyCalories': self.calculate_daily_calories(
                    user_data.get('Weight', 0.0),
                    user_data.get('Height', 0.0),
                    user_data.get('Age', 0),
                    user_data.get('Sex', ''),
                    user_data.get('Activity_level', 'Sedentary')
                ),
                'CreatedAt': datetime.now().isoformat()
            }
            
            create_query = """
            CREATE (u:User)
            SET u = $user_data
            RETURN u
            """
            
            result = self.neo4j.query(create_query, {"user_data": standardized_data})
            
            if result:
                return True, "User created successfully"
            else:
                return False, "Failed to create user"
                
        except Exception as e:
            print(f"Error creating user: {e}")
            return False, str(e)

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate a user with username and password."""
        query = """
        MATCH (u:User)
        WHERE u.Username = $username AND u.Password = $password
        RETURN properties(u) as user
        """
        try:
            result = self.neo4j.query(query, {
                "username": username,
                "password": password
            })
            
            if result and len(result) > 0 and 'user' in result[0]:
                user_data = result[0]['user']
                # Standardize the data structure
                standardized_data = {
                    'UserID': user_data.get('UserID'),
                    'Username': user_data.get('Username'),
                    'Name': user_data.get('Name'),
                    'Age': user_data.get('Age'),
                    'Sex': user_data.get('Sex'),
                    'Weight': user_data.get('Weight'),
                    'Height': user_data.get('Height'),
                    'ActivityLevel': user_data.get('ActivityLevel', user_data.get('Activity_level')),
                    'Goal': user_data.get('Goal'),
                    'Preference': user_data.get('Preference', []),
                    'DailyCalories': user_data.get('DailyCalories', user_data.get('Daily_calories')),
                    'CreatedAt': user_data.get('CreatedAt')
                }
                return standardized_data
            return None
            
        except Exception as e:
            print(f"Authentication error: {e}")
            return None

    def get_user_data(self, user_id: str) -> Dict:
        """Get user details and preferences."""
        query = """
        MATCH (u:User {UserID: $user_id})
        RETURN properties(u) as user
        """
        
        try:
            result = self.neo4j.query(query, {"user_id": user_id})
            if not result:
                return {"error": "User not found"}
                
            user_data = result[0]['user']
            # Standardize the data structure
            standardized_data = {
                'UserID': user_data.get('UserID'),
                'Username': user_data.get('Username'),
                'Name': user_data.get('Name'),
                'Age': user_data.get('Age'),
                'Sex': user_data.get('Sex'),
                'Weight': user_data.get('Weight'),
                'Height': user_data.get('Height'),
                'ActivityLevel': user_data.get('ActivityLevel', user_data.get('Activity_level')),
                'Goal': user_data.get('Goal'),
                'Preference': user_data.get('Preference', []),
                'DailyCalories': user_data.get('DailyCalories', user_data.get('Daily_calories')),
                'CreatedAt': user_data.get('CreatedAt')
            }
            return standardized_data
            
        except Exception as e:
            print(f"Error retrieving user data: {e}")
            return {"error": str(e)}

    def calculate_daily_calories(self, weight: float, height: float, 
                               age: int, sex: str, activity_level: str) -> int:
        """Calculate daily caloric needs using Harris-Benedict equation."""
        # BMR calculation
        if sex.lower() == "male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        # Activity multipliers
        multipliers = {
            "Sedentary": 1.2,
            "Light": 1.375,
            "Moderate": 1.55,
            "Active": 1.725,
            "Very Active": 1.9
        }
        
        return int(bmr * multipliers[activity_level])
    
    def _update_user_id(self, username: str, user_id: str):
        """Update UserID for existing user."""
        query = """
        MATCH (u:User {Username: $username})
        SET u.UserID = $user_id
        """
        try:
            self.neo4j.query(query, {
                "username": username,
                "user_id": user_id
            })
        except Exception as e:
            print(f"Error updating UserID: {e}")

    def get_user_meal_plan(self, user_id: str) -> Optional[Dict]:
        """Retrieve the user's meal plan from the knowledge graph."""
        query = """
        MATCH (u:User {UserID: $user_id})-[:FOLLOWS]->(mp:MealPlan)
        WITH mp ORDER BY mp.CreatedAt DESC LIMIT 1
        OPTIONAL MATCH (mp)-[:HAS_MEAL_DAY]->(md:MealDay)
        OPTIONAL MATCH (md)-[:HAS_MEAL]->(m:Meal)
        OPTIONAL MATCH (m)-[:CONTAINS]->(f:Food)
        RETURN mp.DailyCalorieTarget as daily_calorie_target,
            mp.CreatedAt as generated_at,
            md.DayNumber as day,
            m.Type as meal_type,
            collect({
                name: f.Name,
                portion: f.Portion,
                calories: f.Calories,
                proteins: f.Proteins,
                carbohydrates: f.Carbohydrates,
                fats: f.Fats
            }) as foods,
            sum(f.Calories) as total_meal_calories,
            sum(f.Proteins) as total_meal_proteins,
            sum(f.Carbohydrates) as total_meal_carbs,
            sum(f.Fats) as total_meal_fats
        ORDER BY md.DayNumber, m.Type
        """
        
        try:
            result = self.neo4j.query(query, {"user_id": user_id})
            if not result:
                return None

            # Organize the results into a structured meal plan
            meal_plan = {
                "summary": {
                    "daily_calorie_target": result[0]['daily_calorie_target'],
                    "generated_at": result[0]['generated_at']
                },
                "days": []
            }

            current_day = None
            current_day_meals = {}

            for row in result:
                day_number = row['day']
                
                if day_number is None:
                    continue

                if current_day != day_number:
                    if current_day is not None:
                        meal_plan["days"].append({
                            "day": current_day,
                            "meals": current_day_meals,
                            "total_calories": sum(m['total_calories'] for m in current_day_meals.values())
                        })
                    current_day = day_number
                    current_day_meals = {}

                meal_type = row['meal_type'].lower()
                if row['foods']:
                    current_day_meals[meal_type] = {
                        "foods": row['foods'],
                        "total_calories": row['total_meal_calories'],
                        "total_proteins": row['total_meal_proteins'],
                        "total_carbohydrates": row['total_meal_carbs'],
                        "total_fats": row['total_meal_fats']
                    }

            # Add the last day
            if current_day is not None:
                meal_plan["days"].append({
                    "day": current_day,
                    "meals": current_day_meals,
                    "total_calories": sum(m['total_calories'] for m in current_day_meals.values())
                })

            return meal_plan

        except Exception as e:
            print(f"Error retrieving meal plan: {e}")
            return None
        
    def save_workout_plan(self, user_id: str, workout_plan: Dict) -> bool:
        """Save the workout plan to the knowledge graph for a specific user."""
        try:
            plan_id = f"workout{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Create workout plan node
            create_plan_query = """
            MATCH (u:User {UserID: $user_id})
            CREATE (wp:WorkoutPlan { 
                WorkoutPlanID: $plan_id, 
                Goal: $goal, 
                GeneratedAt: $generated_at, 
                Duration: $duration 
            })
            CREATE (u)-[:FOLLOWS]->(wp)
            """

            # Create WorkoutDay nodes
            create_day_query = """
            MATCH (wp:WorkoutPlan {WorkoutPlanID: $plan_id})
            CREATE (wd:WorkoutDay { 
                DayID: $day_id, 
                Day: $day_number, 
                MuscleGroups: $muscle_groups 
            })
            CREATE (wp)-[:HAS_WORKOUT_DAY]->(wd)
            """

            # Create Exercise relationships
            create_exercise_query = """
            MATCH (wd:WorkoutDay {DayID: $day_id})
            CREATE (e:Exercise { 
                ExerciseID: $exercise_id, 
                Name: $name, 
                Sets: $sets, 
                Reps: $reps, 
                RestSeconds: $rest_seconds, 
                TargetMuscles: $target_muscles, 
                Notes: $notes 
            })
            CREATE (wd)-[:INCLUDES]->(e)
            """

            # Save the workout plan
            self.neo4j.query(create_plan_query, {
                "user_id": user_id,
                "plan_id": plan_id,
                "goal": workout_plan['summary']['goal'],
                "generated_at": workout_plan['summary']['generated_at'],
                "duration": len(workout_plan['days'])
            })

            # Create days
            for day_data in workout_plan['days']:
                day_id = f"{plan_id}_day{day_data['day']}"
                self.neo4j.query(create_day_query, {
                    "plan_id": plan_id,
                    "day_id": day_id,
                    "day_number": day_data['day'],
                    "muscle_groups": ','.join(day_data['muscle_groups'])
                })

                # Create exercises
                for exercise in day_data['exercises']:
                    exercise_id = f"{day_id}{exercise['name'].replace(' ', '')}"
                    self.neo4j.query(create_exercise_query, {
                        "day_id": day_id,
                        "exercise_id": exercise_id,
                        "name": exercise['name'],
                        "sets": exercise['sets'],
                        "reps": exercise['reps'],
                        "rest_seconds": exercise['rest_seconds'],
                        "target_muscles": ','.join(exercise['target_muscles']),
                        "notes": exercise['notes']
                    })

            return True

        except Exception as e:
            self.logger.error(f"Error saving workout plan: {e}", exc_info=True)
            return False
            
    def get_latest_workout_plan(self, user_id: str) -> Dict:
        """Retrieve the most recent workout plan for a specific user from the knowledge graph.
        
        Args:
            user_id: The ID of the user whose workout plan to retrieve
            
        Returns:
            Dict: The workout plan in the same format as used for display, or None if no plan exists
        """
        try:
            # Query to retrieve the most recent workout plan for the user
            query = """
            MATCH (u:User {UserID: $user_id})-[:FOLLOWS]->(wp:WorkoutPlan)
            MATCH (wp)-[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)
            MATCH (wd)-[:INCLUDES]->(e:Exercise)
            RETURN wp.WorkoutPlanID, wp.Goal, wp.GeneratedAt, wp.Duration,
                wd.DayID, wd.Day, wd.MuscleGroups,
                e.Name, e.Sets, e.Reps, e.RestSeconds, e.TargetMuscles, e.Notes
            ORDER BY wp.GeneratedAt DESC, wd.Day ASC
            """
            
            result = self.neo4j.query(query, {"user_id": user_id})
            
            if not result:
                self.logger.info(f"No workout plan found for user {user_id}")
                return None
                
            # Process the results to recreate the workout plan structure
            workout_plans = {}
            for record in result:
                plan_id = record["wp.WorkoutPlanID"]
                
                # Initialize the plan if we haven't seen it before
                if plan_id not in workout_plans:
                    workout_plans[plan_id] = {
                        "summary": {
                            "goal": record["wp.Goal"],
                            "generated_at": record["wp.GeneratedAt"],
                            "duration": record["wp.Duration"]
                        },
                        "days": []
                    }
                    
                # Get the day data
                day_id = record["wd.DayID"]
                day_number = record["wd.Day"]
                muscle_groups = record["wd.MuscleGroups"].split(',')
                
                # Find the day in our structure, or add it if not there
                day_data = None
                for day in workout_plans[plan_id]["days"]:
                    if day["day"] == day_number:
                        day_data = day
                        break
                        
                if not day_data:
                    day_data = {
                        "day": day_number,
                        "muscle_groups": muscle_groups,
                        "exercises": []
                    }
                    workout_plans[plan_id]["days"].append(day_data)
                    
                # Add the exercise to the day
                exercise = {
                    "name": record["e.Name"],
                    "sets": record["e.Sets"],
                    "reps": record["e.Reps"],
                    "rest_seconds": record["e.RestSeconds"],
                    "target_muscles": record["e.TargetMuscles"].split(','),
                    "notes": record["e.Notes"]
                }
                
                day_data["exercises"].append(exercise)
                
            # Sort days by day number
            for plan_id in workout_plans:
                workout_plans[plan_id]["days"].sort(key=lambda x: x["day"])
                
            # Return only the most recent plan (first in the dictionary)
            return list(workout_plans.values())[0] if workout_plans else None
            
        except Exception as e:
            self.logger.error(f"Error retrieving workout plan: {e}", exc_info=True)
            return None
    
    def save_meal_plan(self, user_id: str, meal_plan: Dict) -> bool:
        """Salva il meal plan nel knowledge graph per un utente specifico.
        
        Args:
            user_id: L'ID dell'utente a cui associare il meal plan
            meal_plan: Il dizionario contenente il meal plan completo
            
        Returns:
            bool: True se l'operazione è andata a buon fine, False altrimenti
        """
        try:
            plan_id = f"mealplan{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Crea il nodo MealPlan
            create_plan_query = """
            MATCH (u:User {UserID: $user_id})
            CREATE (mp:MealPlan { 
                MealPlanID: $plan_id, 
                DailyCalorieTarget: $daily_calorie_target, 
                GeneratedAt: $generated_at,
                Duration: $duration
            })
            CREATE (u)-[:FOLLOWS]->(mp)
            """
            
            # Crea i nodi MealDay
            create_day_query = """
            MATCH (mp:MealPlan {MealPlanID: $plan_id})
            CREATE (md:MealDay { 
                DayID: $day_id, 
                Day: $day_number,
                TotalCalories: $total_calories
            })
            CREATE (mp)-[:HAS_MEAL_DAY]->(md)
            """
            
            # Crea i nodi Meal
            create_meal_query = """
            MATCH (md:MealDay {DayID: $day_id})
            CREATE (m:Meal { 
                MealID: $meal_id,
                Type: $meal_type,
                TotalCalories: $total_calories,
                TotalProteins: $total_proteins,
                TotalCarbohydrates: $total_carbs,
                TotalFats: $total_fats
            })
            CREATE (md)-[:HAS_MEAL]->(m)
            """
            
            # Crea le relazioni con i Food
            create_food_relation_query = """
            MATCH (m:Meal {MealID: $meal_id})
            MERGE (f:Food { 
                Name: $food_name,
                Calories: $calories,
                Proteins: $proteins,
                Carbohydrates: $carbs,
                Fats: $fats
            })
            CREATE (m)-[:CONTAINS {Portion: $portion}]->(f)
            """
            
            # Salva il meal plan
            self.neo4j.query(create_plan_query, {
                "user_id": user_id,
                "plan_id": plan_id,
                "daily_calorie_target": meal_plan['summary']['daily_calorie_target'],
                "generated_at": meal_plan['summary']['generated_at'],
                "duration": len(meal_plan['days'])
            })
            
            # Crea i giorni
            for day_data in meal_plan['days']:
                day_id = f"{plan_id}_day{day_data['day']}"
                self.neo4j.query(create_day_query, {
                    "plan_id": plan_id,
                    "day_id": day_id,
                    "day_number": day_data['day'],
                    "total_calories": day_data['total_calories']
                })
                
                # Crea i pasti per ogni giorno
                for meal_type, meal_info in day_data['meals'].items():
                    meal_id = f"{day_id}_{meal_type}"
                    self.neo4j.query(create_meal_query, {
                        "day_id": day_id,
                        "meal_id": meal_id,
                        "meal_type": meal_type.upper(),
                        "total_calories": meal_info['total_calories'],
                        "total_proteins": meal_info['total_proteins'],
                        "total_carbs": meal_info['total_carbohydrates'],
                        "total_fats": meal_info['total_fats']
                    })
                    
                    # Crea le relazioni con i cibi per ogni pasto
                    for food in meal_info['foods']:
                        self.neo4j.query(create_food_relation_query, {
                            "meal_id": meal_id,
                            "food_name": food['name'],
                            "calories": food['calories'],
                            "proteins": food['proteins'],
                            "carbs": food['carbohydrates'],
                            "fats": food['fats'],
                            "portion": food['portion']
                        })
            
            logger.info(f"Successfully saved meal plan {plan_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving meal plan: {e}", exc_info=True)
            return False
        
    def get_latest_meal_plan(self, user_id: str) -> Dict:
        """Recupera il meal plan più recente per un utente specifico dal knowledge graph.
        
        Args:
            user_id: L'ID dell'utente di cui recuperare il meal plan
            
        Returns:
            Dict: Il meal plan nel formato usato per la visualizzazione, o None se non esiste
        """
        try:
            # Query per recuperare il meal plan più recente dell'utente
            query = """
            MATCH (u:User {UserID: $user_id})-[:FOLLOWS]->(mp:MealPlan)
            MATCH (mp)-[:HAS_MEAL_DAY]->(md:MealDay)
            MATCH (md)-[:HAS_MEAL]->(m:Meal)
            MATCH (m)-[r:CONTAINS]->(f:Food)
            RETURN mp.MealPlanID, mp.DailyCalorieTarget, mp.GeneratedAt, mp.Duration,
                md.DayID, md.Day, md.TotalCalories,
                m.MealID, m.Type, m.TotalCalories, m.TotalProteins, m.TotalCarbohydrates, m.TotalFats,
                f.Name, f.Calories, f.Proteins, f.Carbohydrates, f.Fats, r.Portion
            ORDER BY mp.GeneratedAt DESC, md.Day ASC, m.Type
            """
            
            result = self.neo4j.query(query, {"user_id": user_id})
            
            if not result:
                logger.info(f"No meal plan found for user {user_id}")
                return None
                
            # Processa i risultati per ricreare la struttura del meal plan
            meal_plans = {}
            for record in result:
                plan_id = record["mp.MealPlanID"]
                
                # Inizializza il piano se non l'abbiamo ancora visto
                if plan_id not in meal_plans:
                    meal_plans[plan_id] = {
                        "summary": {
                            "daily_calorie_target": record["mp.DailyCalorieTarget"],
                            "generated_at": record["mp.GeneratedAt"],
                            "meal_calorie_targets": {
                                "breakfast": record["mp.DailyCalorieTarget"] * 0.25,
                                "lunch": record["mp.DailyCalorieTarget"] * 0.35,
                                "dinner": record["mp.DailyCalorieTarget"] * 0.30,
                                "snack": record["mp.DailyCalorieTarget"] * 0.10
                            }
                        },
                        "days": []
                    }
                    
                # Ottieni i dati del giorno
                day_id = record["md.DayID"]
                day_number = record["md.Day"]
                day_calories = record["md.TotalCalories"]
                
                # Trova il giorno nella nostra struttura, o aggiungilo se non c'è
                day_data = None
                for day in meal_plans[plan_id]["days"]:
                    if day["day"] == day_number:
                        day_data = day
                        break
                        
                if not day_data:
                    day_data = {
                        "day": day_number,
                        "total_calories": day_calories,
                        "meals": {}
                    }
                    meal_plans[plan_id]["days"].append(day_data)
                    
                # Ottieni i dati del pasto
                meal_id = record["m.MealID"]
                meal_type = record["m.Type"].lower()
                
                # Aggiungi il pasto se non esiste già
                if meal_type not in day_data["meals"]:
                    day_data["meals"][meal_type] = {
                        "total_calories": record["m.TotalCalories"],
                        "total_proteins": record["m.TotalProteins"],
                        "total_carbohydrates": record["m.TotalCarbohydrates"],
                        "total_fats": record["m.TotalFats"],
                        "foods": []
                    }
                    
                # Aggiungi il cibo al pasto
                food = {
                    "name": record["f.Name"],
                    "calories": record["f.Calories"],
                    "proteins": record["f.Proteins"],
                    "carbohydrates": record["f.Carbohydrates"],
                    "fats": record["f.Fats"],
                    "portion": record["r.Portion"]
                }
                
                day_data["meals"][meal_type]["foods"].append(food)
                
            # Ordina i giorni per numero
            for plan_id in meal_plans:
                meal_plans[plan_id]["days"].sort(key=lambda x: x["day"])
                
            # Restituisci solo il piano più recente (il primo nel dizionario)
            return list(meal_plans.values())[0] if meal_plans else None
            
        except Exception as e:
            logger.error(f"Error retrieving meal plan: {e}", exc_info=True)
            return None