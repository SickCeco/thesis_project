
from meal_plans_manager import MealPlanManager
import streamlit as st
from config_loader import load_config
from neo4j_connector import Neo4jConnector
from user_manager import UserManager
from streamlit_interface import StreamlitInterface
from datetime import datetime

def main():
    config = load_config("config/settings.yaml")
    
    neo4j = Neo4jConnector(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="Test1-50"
    )
    
    user_manager = UserManager(neo4j)
    meal_plans_manager = MealPlanManager(neo4j, user_manager)
    
    interface = StreamlitInterface(
    user_manager=user_manager,
    meal_plans_manager=meal_plans_manager,
)
    interface.run()
    
if __name__ == "__main__":
    main()