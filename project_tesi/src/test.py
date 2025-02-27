import streamlit as st
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from datetime import datetime

from meal_plans_manager import MealPlanManager

# Dummy classes for demo purposes if you don't have actual implementations
class DummyUserManager:
    def get_user_data(self, user_id):
        return {
            "Goal": "weight_loss",
            "Weight": 75,
            "ActivityLevel": "moderate",
            "Preference": "balanced",
            "DailyCalories": 2000
        }

class Neo4jConnector:
    def __init__(self, url, username, password):
        self.url = url
        self.username = username
        self.password = password
        self._driver = GraphDatabase.driver(url, auth=(username, password))

    def query(self, query, parameters=None):
        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

# Initialize Streamlit app
st.title("🥗 Meal Plan Manager")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Load environment variables
load_dotenv()

# Get Neo4j credentials from environment variables or user input
neo4j_url = st.sidebar.text_input("Neo4j URL", os.getenv("NEO4J_URL", "bolt://localhost:7687"))
neo4j_user = st.sidebar.text_input("Neo4j Username", os.getenv("NEO4J_USER", "neo4j"))
neo4j_password = st.sidebar.text_input("Neo4j Password", os.getenv("NEO4J_PASSWORD", ""), type="password")

# Initialize managers when credentials are provided
if neo4j_url and neo4j_user and neo4j_password:
    try:
        neo4j_connector = Neo4jConnector(neo4j_url, neo4j_user, neo4j_password)
        user_manager = DummyUserManager()
        meal_plan_manager = MealPlanManager(neo4j_connector, user_manager)
        st.sidebar.success("✅ Connected to Neo4j!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to connect: {str(e)}")

# Main content
st.header("Generate Meal Plan")

# User input form
with st.form("user_input"):
    user_id = st.text_input("User ID", "user123")
    st.markdown("### User Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", 40, 200, 75)
        activity_level = st.selectbox(
            "Activity Level",
            ["sedentary", "light", "moderate", "very_active", "extra_active"]
        )
    
    with col2:
        goal = st.selectbox(
            "Goal",
            ["weight_loss", "maintenance", "muscle_gain"]
        )
        dietary_pref = st.selectbox(
            "Dietary Preference",
            ["balanced", "low_carb", "high_protein", "vegetarian", "vegan"]
        )

    submit_button = st.form_submit_button("Generate Meal Plan")

# Handle form submission
if submit_button:
    with st.spinner("Generating meal plan..."):
        try:
            # Update user data in the dummy manager
            user_manager.get_user_data = lambda x: {
                "Goal": goal,
                "Weight": weight,
                "ActivityLevel": activity_level,
                "Preference": dietary_pref,
                "DailyCalories": 2000  # This could be calculated based on user data
            }
            
            # Generate meal plan
            result = meal_plan_manager.suggest_meal_plan(user_id)
            
            if "error" in result:
                st.error(f"Failed to generate meal plan: {result['error']}")
            else:
                st.success("Meal plan generated successfully!")
                
                # Display the meal plan
                plan = result["plan"]["structured_plan"]
                for day, meals in plan.items():
                    st.subheader(day)
                    for meal_type, foods in meals.items():
                        with st.expander(f"{meal_type} ({len(foods)} items)"):
                            for food in foods:
                                st.write(f"• {food['name']}")
                                cols = st.columns(4)
                                cols[0].write(f"Quantity: {food['quantity']}g")
                                cols[1].write(f"Calories: {food['nutrition']['calories']}")
                                cols[2].write(f"Protein: {food['nutrition']['protein']}g")
                                cols[3].write(f"Carbs: {food['nutrition']['carbs']}g")
                
                # Display metadata
                with st.expander("Plan Metadata"):
                    st.json(result["metadata"])
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Additional features in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
show_debug = st.sidebar.checkbox("Show Debug Info")

if show_debug:
    st.sidebar.markdown("### Debug Information")
    st.sidebar.write("Last Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.sidebar.write("Active User:", user_id)