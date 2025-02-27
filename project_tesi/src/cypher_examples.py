class CypherExamples:

    @staticmethod
    def get_schema_info():
        """Returns the graph schema including nodes and relationships with their attributes"""
        return {
            "nodes": {
                "Meal": {
                    "properties": ["embedding", "type", "mealId"],
                    "relationships": {
                        "CONTAINS": {
                            "to": "Food",
                            "properties": ["quantity"]
                        },
                        "HAS_MEAL": {"from": "MealDay", "type": "incoming"}
                    }
                },
                "WorkoutPlan": {
                    "properties": ["embedding", "duration", "focus", "workoutPlanId"],
                    "relationships": {
                        "HAS_WORKOUT_DAY": "WorkoutDay",
                        "FOLLOWS": {"from": "User", "type": "incoming"}
                    }
                },
                "WorkoutDay": {
                    "properties": ["embedding", "name", "workoutDayId"],
                    "relationships": {
                        "HAS_EXERCISE": {
                            "to": "Exercise",
                            "properties": ["repetitions", "sets"]
                        },
                        "HAS_WORKOUT_DAY": {"from": "WorkoutPlan", "type": "incoming"}
                    }
                },
                "Food": {
                    "properties": ["embedding", "name", "category", "calories", "proteins", 
                                "carbohydrates", "fats", "foodId", "knowledgeBase"],
                    "relationships": {
                        "CONTAINS": {"from": "Meal", "type": "incoming"}
                    }
                },
                "Exercise": {
                    "properties": ["embedding", "name", "type", "description", 
                                "exerciseId", "primaryMuscle"],
                    "relationships": {
                        "HAS_EXERCISE": {"from": "WorkoutDay", "type": "incoming"}
                    }
                },
                "User": {
                    "properties": ["embedding", "name", "age", "goal", "goalAchieved", 
                                "height", "preferences", "weight", "activityLevel", "userId"],
                    "relationships": {
                        "FOLLOWS": ["MealPlan", "WorkoutPlan"]
                    }
                },
                "MealPlan": {
                    "properties": ["embedding", "duration", "focus", "mealPlanId"],
                    "relationships": {
                        "HAS_MEAL_DAY": "MealDay",
                        "FOLLOWS": {"from": "User", "type": "incoming"}
                    }
                },
                "MealDay": {
                    "properties": ["embedding", "name", "mealDayId"],
                    "relationships": {
                        "HAS_MEAL": "Meal",
                        "HAS_MEAL_DAY": {"from": "MealPlan", "type": "incoming"}
                    }
                }
            }
        }

    @staticmethod
    def get_all_examples():
        """Returns all example Cypher queries that match the updated graph structure"""
        all_queries = {}
        
        # Add meal plan queries
        all_queries.update(CypherExamples.get_meal_plan_queries())
        
        # Add workout plan queries
        all_queries.update(CypherExamples.get_workout_plan_queries())
        
        # Add food queries
        all_queries.update(CypherExamples.get_food_queries())
        
        # Add analysis queries
        all_queries.update(CypherExamples.get_analysis_queries())
        
        # Add user queries
        all_queries.update(CypherExamples.get_user_queries())
        
        return all_queries

    @staticmethod
    def get_example_conversations():
        """Returns examples of natural language questions and their corresponding Cypher queries"""
        return [
            {
                "input": "What meals are in the meal plan focused on weight loss?",
                "tool_calls": [{
                    "query": """
                    MATCH (mp:MealPlan {focus: 'weight_loss'})
                    MATCH (mp)-[:HAS_MEAL_DAY]->(md:MealDay)
                    MATCH (md)-[:HAS_MEAL]->(m:Meal)
                    RETURN md.name as day, m.type as mealType, m.mealId
                    ORDER BY md.name, m.type;
                    """,
                    "sub_queries": [
                        "Find meal plan focused on weight loss",
                        "Get all meal days in the plan",
                        "Get all meals for each day"
                    ]
                }]
            },
            {
                "input": "Show me exercises targeting chest muscles for beginners",
                "tool_calls": [{
                    "query": """
                    MATCH (e:Exercise)
                    WHERE e.primaryMuscle = 'chest'
                    AND EXISTS((e)<-[:HAS_EXERCISE]-(:WorkoutDay)<-[:HAS_WORKOUT_DAY]-(:WorkoutPlan {focus: 'beginner'}))
                    RETURN e.name, e.type, e.description
                    ORDER BY e.name;
                    """,
                    "sub_queries": [
                        "Find exercises targeting chest",
                        "Filter for exercises used in beginner workout plans",
                        "Return exercise details"
                    ]
                }]
            },
            {
                "input": "Show me all breakfast meals under 500 calories",
                "tool_calls": [{
                    "query": """
                    MATCH (m:Meal {type: 'Breakfast'})-[:CONTAINS]->(f:Food)
                    WHERE f.calories < 500
                    RETURN f.name SUM(f.calories) AS TotalCalories ;
                    """,
                    "sub_queries": [
                        "Find meals of type Breakfast",
                        "Filter for foods under 500 calories",
                        "Collect food names and calculate total calories"
                    ]
                }]
            },
            {
                "input": "What's the nutrition breakdown for a user's daily meal plan?",
                "tool_calls": [{
                    "query": """
                    MATCH (u:User {userId: $userId})-[:FOLLOWS]->(mp:MealPlan)
                    MATCH (mp)-[:HAS_MEAL_DAY]->(md:MealDay)
                    MATCH (md)-[:HAS_MEAL]->(m:Meal)
                    MATCH (m)-[r:CONTAINS]->(f:Food)
                    WITH md.name as day, 
                         SUM(f.calories * r.servingSize) as totalCalories,
                         SUM(f.proteins * r.servingSize) as totalProteins,
                         SUM(f.carbohydrates * r.servingSize) as totalCarbs,
                         SUM(f.fats * r.servingSize) as totalFats
                    RETURN day, totalCalories, totalProteins, totalCarbs, totalFats
                    ORDER BY day;
                    """,
                    "sub_queries": [
                        "Find user's meal plan",
                        "Get all meals and their foods",
                        "Calculate daily nutrition totals"
                    ]
                }]
            },
            {
                "input": "Find users who have achieved their weight loss goals",
                "tool_calls": [{
                    "query": """
                    MATCH (u:User)
                    WHERE u.goal = 'weight_loss' AND u.goalAchieved = true
                    RETURN u.name, u.age, 
                           u.initialWeight - u.weight as weightLost,
                           u.activityLevel
                    ORDER BY weightLost DESC;
                    """,
                    "sub_queries": [
                        "Find users with weight loss goal",
                        "Filter for those who achieved it",
                        "Calculate weight lost"
                    ]
                }]
            }
        ]

    @staticmethod
    def get_meal_plan_queries():
        """Returns Cypher queries related to meal plans"""
        return {
            "get_meal_plan_by_focus": """
                MATCH (mp:MealPlan {focus: $focus})
                RETURN mp
                ORDER BY mp.duration;
                """,
            "get_meal_plan_days": """
                MATCH (mp:MealPlan {mealPlanId: $planId})-[:HAS_MEAL_DAY]->(md:MealDay)
                RETURN md
                ORDER BY md.name;
                """,
            "get_meal_plan_nutrition": """
                MATCH (mp:MealPlan {mealPlanId: $planId})
                MATCH (mp)-[:HAS_MEAL_DAY]->(md:MealDay)
                MATCH (md)-[:HAS_MEAL]->(m:Meal)
                MATCH (m)-[r:CONTAINS]->(f:Food)
                WITH md.name as day,
                     SUM(f.calories * r.servingSize) as calories,
                     SUM(f.proteins * r.servingSize) as proteins
                RETURN day, calories, proteins
                ORDER BY day;
                """
        }

    @staticmethod
    def get_workout_plan_queries():
        """Returns Cypher queries related to workout plans"""
        return {
            "get_workout_plan_by_focus": """
                MATCH (wp:WorkoutPlan {focus: $focus})
                RETURN wp
                ORDER BY wp.duration;
                """,
            "get_workout_days": """
                MATCH (wp:WorkoutPlan {workoutPlanId: $planId})-[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)
                RETURN wd
                ORDER BY wd.name;
                """,
            "get_exercises_by_muscle": """
                MATCH (e:Exercise)
                WHERE e.primaryMuscle = $muscle
                RETURN e
                ORDER BY e.name;
                """,
            "get_complete_workout": """
                MATCH (wp:WorkoutPlan {workoutPlanId: $planId})
                MATCH (wp)-[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)
                MATCH (wd)-[r:HAS_EXERCISE]->(e:Exercise)
                RETURN wp.focus, wd.name, e.name, e.type, r.sets, r.reps
                ORDER BY wd.name, e.name;
                """
        }

    @staticmethod
    def get_food_queries():
        """Returns Cypher queries related to foods"""
        return {
            "get_foods_by_category": """
                MATCH (f:Food)
                WHERE f.category = $category
                RETURN f
                ORDER BY f.calories;
                """,
            "get_foods_by_nutrition": """
                MATCH (f:Food)
                WHERE f.proteins >= $minProtein
                AND f.carbohydrates <= $maxCarbs
                AND f.fats <= $maxFats
                RETURN f
                ORDER BY f.proteins DESC;
                """,
            "get_food_sources": """
                MATCH (f:Food)
                WHERE f.knowledgeBase IS NOT NULL
                RETURN f.name, f.knowledgeBase
                ORDER BY f.name;
                """
        }

    @staticmethod
    def get_analysis_queries():
        """Returns Cypher queries for analysis purposes"""
        return {
            "analyze_user_progress": """
                MATCH (u:User)
                WHERE u.goalAchieved = true
                WITH u.goal as goal, COUNT(u) as achievedCount
                MATCH (total:User {goal: goal})
                WITH goal, achievedCount, COUNT(total) as totalCount
                RETURN goal, 
                       achievedCount,
                       totalCount,
                       (toFloat(achievedCount) / totalCount * 100) as successRate
                ORDER BY successRate DESC;
                """,
            "analyze_popular_exercises": """
                MATCH (:WorkoutPlan)-[:HAS_WORKOUT_DAY]->(wd:WorkoutDay)
                MATCH (wd)-[:HAS_EXERCISE]->(e:Exercise)
                WITH e.name as exercise, e.primaryMuscle as muscle, COUNT(*) as usage
                RETURN exercise, muscle, usage
                ORDER BY usage DESC
                LIMIT 10;
                """
        }

    @staticmethod
    def get_user_queries():
        """Returns Cypher queries related to users"""
        return {
            "get_users_by_goal": """
                MATCH (u:User {goal: $goal})
                RETURN u
                ORDER BY u.age;
                """,
            "get_user_preferences": """
                MATCH (u:User {userId: $userId})
                RETURN u.preferences
                """,
            "find_similar_users": """
                MATCH (u1:User {userId: $userId})
                MATCH (u2:User)
                WHERE u1.goal = u2.goal
                AND ABS(u1.age - u2.age) <= 5
                AND u1.activityLevel = u2.activityLevel
                AND u1 <> u2
                RETURN u2.name, u2.age, u2.goal, u2.activityLevel
                ORDER BY u2.goalAchieved DESC;
                """
        }