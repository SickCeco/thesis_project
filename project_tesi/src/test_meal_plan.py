import json
from meal_plans_manager import MealPlanGraphRAG, UserParameters, Goal, ActivityLevel
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)

# Create user parameters
user_params = UserParameters(
    name="John",
    age=30,
    weight=113.0,  # in kg
    height=196.0,  # in cm
    goal=Goal.WEIGHT_LOSS,  # Using the Enum value
    activity_level=ActivityLevel.MODERATE,  # Using the Enum value
    dietary_restrictions=["Omnivore"],
    gender="Male"
)

# Initialize system
planner = MealPlanGraphRAG()

# Create plan
try:
    meal_plan = planner.create_personalized_meal_plan(user_params)
    
    if meal_plan:
        print("Plan created successfully!")
        print(json.dumps(meal_plan, indent=2))
    else:
        print("Failed to create plan")
except Exception as e:
    print(f"Error creating meal plan: {str(e)}")