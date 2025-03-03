from datetime import datetime
import json
import logging
import io
import pandas as pd
from chatbot import GraphRAGBot
from meal_plans_manager import MealPlanGraphRAG, UserParameters
from neo4j_connector import Neo4jConnector
from user_manager import UserManager
from workout_plans_manager import Goal, WorkoutPlanGraphRAG
import streamlit as st

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"

if st.session_state.get('current_page') in ["Login", "Register"]:
    st.set_page_config(
        page_title="Fitness and Nutrition Assistant",
        page_icon="🤖",
        layout="centered"  # Layout normale per login e registrazione
    )
else:
    st.set_page_config(
        page_title="Fitness and Nutrition Assistant",
        page_icon="🤖",
        layout="wide"  # Layout wide per tutte le altre pagine
    )

class StreamlitInterface:
    def __init__(self):
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create handlers if they don't exist
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            try:
                file_handler = logging.FileHandler('streamlit_interface.log')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Could not create log file: {e}")

        # Initialize Neo4j connection
        self.neo4j = Neo4jConnector(
            uri="neo4j+s://72f45d03.databases.neo4j.io",
            username="neo4j",
            password="f9bogjM_hyfkYrTUmspKEk6FCX_hoLKtQxPTEOf98TM"
        )
        
        # Initialize managers
        self.user_manager = UserManager(self.neo4j)
       
        self.logger.info("Initializing StreamlitInterface")

        # Initialize session states
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Login"
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'registration_success' not in st.session_state:
            st.session_state.registration_success = False
        if 'previous_page' not in st.session_state:
            st.session_state.previous_page = None
        if 'bot_initialized' not in st.session_state:
            st.session_state.bot_initialized = False

        self.logger.info("StreamlitInterface initialization completed")

    def logout_user(self):
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.session_state.current_page = "Login"
        st.session_state.messages = []
        if 'bot' in st.session_state:
            del st.session_state.bot
        st.session_state.previous_page = None
        st.session_state.bot_initialized = False

    def run(self):

        if st.session_state.authenticated:
            if st.session_state.user_data and 'Username' in st.session_state.user_data:
                # Navigation bar
                self.show_improved_sidebar()
            else:
                self.logger.error("Invalid user_data in session state")
                self.logout_user()
                st.rerun()
                return

            # Page handling
            if st.session_state.current_page == "Chatbot":
                self.show_chatbot_page()
            elif st.session_state.current_page == "Meal Plan":
                self.show_meal_plan_page()
            elif st.session_state.current_page == "Workout Plan":
                self.show_workout_plan_page()
            elif st.session_state.current_page == "Profile":  
                self.show_profile_page()
        else:
            if st.session_state.current_page == "Login":
                self.show_login_page()
            elif st.session_state.current_page == "Register":
                self.show_registration_page()
                
    def show_improved_sidebar(self):
        """Shows an improved sidebar with user info and navigation"""
        with st.sidebar:
            # User profile section with styling
            st.markdown("## User Profile")
            
            # Display username and BMI on the same line
            if 'Weight' in st.session_state.user_data and 'Height' in st.session_state.user_data:
                bmi = st.session_state.user_data['Weight'] / ((st.session_state.user_data['Height']/100) ** 2)
                st.markdown(f"**{st.session_state.user_data['Username']}** | BMI: **{bmi:.1f}**")
            else:
                st.markdown(f"**{st.session_state.user_data['Username']}**")
            
            # Add profile navigation and logout in the user section
            if st.button("👤 Edit Profile", key="nav_Profile", 
                        type="primary" if st.session_state.current_page == "Profile" else "secondary", 
                        use_container_width=True):
                st.session_state.previous_page = st.session_state.current_page
                st.session_state.current_page = "Profile"
                st.rerun()
                
            if st.button("Logout", key="logout_button", type="secondary", use_container_width=True):
                self.logout_user()
                st.rerun()
            
            st.divider()
            
            # Navigation section with improved styling
            st.markdown("## Navigation")
            
            # Navigation buttons with icons - without profile as it's now in the user section
            pages = {
                "Chatbot": "💬",
                "Meal Plan": "🍽️",
                "Workout Plan": "🏋️‍♂️"
            }
            
            for page, icon in pages.items():
                # Highlight current page
                button_type = "primary" if st.session_state.current_page == page else "secondary"
                if st.button(f"{icon} {page}", key=f"nav_{page}", type=button_type, use_container_width=True):
                    st.session_state.previous_page = st.session_state.current_page
                    st.session_state.current_page = page
                    st.rerun()
            
            # Add some health stats or tips
            st.divider()
            st.markdown("## Health Tip")
            tips = [
                "Drink at least 8 glasses of water daily.",
                "Aim for 150 minutes of moderate exercise weekly.",
                "Include protein with every meal for muscle recovery.",
                "Get 7-9 hours of sleep for optimal recovery.",
                "Take short breaks from sitting every 30 minutes."
            ]
            import random
            st.info(random.choice(tips))
            
    def show_profile_page(self):
        """Display the user profile edit page."""
        if not st.session_state.authenticated:
            st.warning("Please login first!")
            return
            
        st.title("Edit Your Profile")
        
        if not st.session_state.user_data or 'UserID' not in st.session_state.user_data:
            st.error("User data not found. Please try logging in again.")
            return
        
        # Create a form with current user values as defaults
        with st.form("edit_profile_form"):
            # Display current values
            user_data = st.session_state.user_data
            
            # Personal information
            st.subheader("Personal Information")
            name = st.text_input("Name", value=user_data.get('Name', ''))
            age = st.number_input("Age", min_value=18, max_value=100, value=int(user_data.get('Age', 30)))
            sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(user_data.get('Sex', 'Other')))
            
            # Body measurements
            st.subheader("Body Measurements")
            weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=float(user_data.get('Weight', 70.0)))
            height = st.number_input("Height (cm)", min_value=130.0, max_value=250.0, value=float(user_data.get('Height', 170.0)))
            
            # Fitness preferences
            st.subheader("Fitness Preferences")
            activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                value=user_data.get('ActivityLevel', 'Moderate')
            )
            
            goal = st.selectbox(
                "Goal",
                ["Weight Loss", "Weight Gain", "Maintenance"],
                index=["Weight Loss", "Weight Gain", "Maintenance"].index(user_data.get('Goal', 'Maintenance'))
            )
            
            preferences = st.multiselect(
                "Dietary Preferences",
                ["Omnivore", "Vegetarian", "Gluten-Free"],
                default=user_data.get('Preference', [])
            )
            
            # Submit button
            submitted = st.form_submit_button("Update Profile", type="primary")
            
            if submitted:
                # Calculate new daily calories based on updated parameters
                daily_calories = self.user_manager.calculate_daily_calories(weight, height, age, sex, activity_level)
                
                # Prepare updated user data
                updated_data = {
                    "UserID": user_data['UserID'],
                    "Name": name,
                    "Age": age,
                    "Sex": sex,
                    "Weight": weight,
                    "Height": height,
                    "ActivityLevel": activity_level,
                    "Goal": goal,
                    "Preference": preferences,
                    "Daily_calories": daily_calories
                }
                
                # Update user data in database
                with st.spinner("Updating your profile..."):
                    success = self.user_manager.update_user_node(updated_data)
                    if success:
                        # Update session state with new data
                        st.session_state.user_data.update(updated_data)
                        st.success("✅ Profile updated successfully!")
                        
                        # Force refresh the sidebar to show updated BMI
                        st.rerun()
                    else:
                        st.error("❌ Failed to update profile. Please try again.")
                        
    def show_chatbot_page(self):
        if not st.session_state.authenticated:
            st.warning("Please login first!")
            return

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        if 'bot' not in st.session_state:
            with st.spinner('Initializing the chatbot...'):
                st.session_state.bot = GraphRAGBot()
            st.success('Chatbot ready!')
        
        st.title("GraphRAG Fitness & Nutrition Assistant")

        # Invia un messaggio di benvenuto solo una volta quando la pagina viene aperta per la prima volta
        if not st.session_state.bot_initialized:
            welcome_message = self.generate_welcome_message()
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            st.session_state.bot_initialized = True

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_query = st.chat_input("Type your question here...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.bot.get_response(user_query)
                    # Remove .content since response is already a string
                    st.write(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    def generate_welcome_message(self):
        """Generates a welcome message with examples based on the actual knowledge graph structure"""
        welcome_message = """
👋 **Welcome to the GraphRAG Fitness & Nutrition Assistant!**

I'm here to help you with questions about fitness and nutrition based on my knowledge graph. Here are some examples of questions I can answer:

📊 **Food & Nutrition Analysis**:
- "Which foods have the highest protein-to-calorie ratio?"
- "What foods in the database are high in protein but low in fat?"
- "Show me all foods in the 'Vegetable' category"

🍽️ **Meal & Plan Information**:
- "What meals are included in the high-protein meal plan?"
- "What types of meals are available in the database?"
- "What's the average calorie content of breakfast meals?"

💪 **Exercise & Workout Data**:
- "What exercises target the chest as primary muscle?"
- "List all exercises of type 'strength'"

Feel free to ask me questions about the specific nutrition and fitness data in my knowledge graph!
        """
        return welcome_message

    def show_meal_plan_page(self):
        """Display the meal plan page with daily breakdown of meals and generation capability."""
        if not st.session_state.authenticated:
            st.warning("Please login first!")
            return

        # Add generate plan button first
        st.header("Generate New Plan")
        col1, col2 = st.columns([2, 1])
        with col1:
            generate_clicked = st.button(
                "Generate Personalized Meal Plan",
                help="Click to generate a new meal plan based on your profile",
                type="primary"
            )
            
        # Now add the title below
        st.title("Your Meal Plan")

        if not st.session_state.user_data or 'UserID' not in st.session_state.user_data:
            st.error("User data not found. Please try logging in again.")
            return

        if 'current_meal_plan' not in st.session_state:
            with st.spinner("Loading your saved meal plan..."):
                saved_plan = self.user_manager.get_latest_meal_plan(st.session_state.user_data['UserID'])
                if saved_plan:
                    st.session_state.current_meal_plan = saved_plan
                    st.success("✅ Loaded your most recent meal plan!")

        # Add generate plan button and section
        
        
        # Initialize meal plan manager if needed
        if 'meal_plan_manager' not in st.session_state:
            st.session_state.meal_plan_manager = MealPlanGraphRAG()

        if generate_clicked:
            with st.spinner("🔄 Generating your personalized meal plan... This may take a moment."):
                try:
                    # Convert user data to UserParameters
                    user_params = UserParameters(
                        name=st.session_state.user_data.get('Name', ''),
                        age=st.session_state.user_data.get('Age', 0),
                        weight=float(st.session_state.user_data.get('Weight', 0)),
                        height=float(st.session_state.user_data.get('Height', 0)),
                        goal=st.session_state.user_data.get('Goal', 'maintain'),
                        activity_level=st.session_state.user_data.get('ActivityLevel', 'Moderate'),
                        dietary_restrictions=st.session_state.user_data.get('Preference', []),
                        gender=st.session_state.user_data.get('Sex', 'Other')
                    )
                    
                    # Generate new plan
                    meal_plan = st.session_state.meal_plan_manager.create_personalized_meal_plan(user_params)
                    
                    if meal_plan:
                        st.success("✨ New meal plan generated successfully!")
                        # Store the meal plan in session state
                        st.session_state.current_meal_plan = meal_plan
                        # Force a rerun to show the new plan
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate meal plan. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating meal plan: {str(e)}")
                    return

        # Get meal plan from session state
        meal_plan = getattr(st.session_state, 'current_meal_plan', None)
        
        if not meal_plan:
            st.info("👋 No meal plan found. Click the button above to generate your personalized plan!")
            return

        # Display summary information
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Daily Calorie Target", 
                    value=f"{meal_plan['summary']['daily_calorie_target']:.0f} kcal"
                )
            with col2:
                st.metric(
                    label="Generated On", 
                    value=meal_plan['summary']['generated_at'].split('T')[0]
                )
            with col3:
                st.metric(
                    label="Days in Plan",
                    value=len(meal_plan['days'])
                )

        # Create tabs for each day
        day_tabs = st.tabs([f"Day {day['day']}" for day in meal_plan['days']])
        
        # Populate each day's tab
        for idx, (day_tab, day_data) in enumerate(zip(day_tabs, meal_plan['days'])):
            with day_tab:
                # Daily summary
                st.subheader(f"Day {day_data['day']} Summary")
                st.progress(
                    min(day_data['total_calories'] / meal_plan['summary']['daily_calorie_target'], 1.0),
                    text=f"Daily calories: {day_data['total_calories']:.0f} / {meal_plan['summary']['daily_calorie_target']:.0f} kcal"
                )

                # Create expandable sections for each meal type
                meal_types = {
                    'breakfast': '🌅 Breakfast',
                    'lunch': '🌞 Lunch',
                    'dinner': '🌙 Dinner',
                    'snack': '🍎 Snack'
                }

                for meal_type, meal_icon in meal_types.items():
                    if meal_type in day_data['meals']:
                        meal_data = day_data['meals'][meal_type]
                        
                        with st.expander(f"{meal_icon} - {meal_data['total_calories']:.0f} kcal", expanded=True):
                            # Macronutrient distribution
                            cols = st.columns(4)
                            cols[0].metric("Calories", f"{meal_data['total_calories']:.0f} kcal")
                            cols[1].metric("Protein", f"{meal_data['total_proteins']:.1f}g")
                            cols[2].metric("Carbs", f"{meal_data['total_carbohydrates']:.1f}g")
                            cols[3].metric("Fats", f"{meal_data['total_fats']:.1f}g")

                            # Food items table
                            st.markdown("#### Foods")
                            food_data = []
                            for food in meal_data['foods']:
                                food_data.append({
                                    "Food": food['name'],
                                    "Portion (g)": f"{food['portion']:.0f}g",
                                    "Calories": f"{food['calories']:.0f} kcal",
                                    "Protein": f"{food['proteins']:.1f}g",
                                    "Carbs": f"{food['carbohydrates']:.1f}g",
                                    "Fats": f"{food['fats']:.1f}g"
                                })
                            
                            st.table(food_data)

                # Add a visual separator between meal sections
                st.divider()

        # Add export functionality
        st.divider()
        st.subheader("Export Meal Plan")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as CSV"):
                st.download_button(
                    label="Download CSV",
                    data=self._convert_meal_plan_to_csv(meal_plan),
                    file_name=f"meal_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export as Excel"):
                st.download_button(
                    label="Download Excel",
                    data=self._convert_meal_plan_to_excel(meal_plan),
                    file_name=f"meal_plan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        st.divider()
        st.subheader("Save Plan to Your Profile")
        save_plan_col1, save_plan_col2 = st.columns([3, 1])
        with save_plan_col1:
            st.write("Save this meal plan to your profile for future reference.")
        with save_plan_col2:
            if st.button("Save Plan to Profile", type="primary"):
                with st.spinner("Saving your plan..."):
                    success = self.user_manager.save_meal_plan(
                        st.session_state.user_data['UserID'], 
                        st.session_state.current_meal_plan
                    )
                    if success:
                        st.success("✅ Meal plan saved successfully to your profile!")
                    else:
                        st.error("❌ Failed to save meal plan. Please try again.")

    def show_login_page(self):
        st.markdown("<h1 style='text-align: center;'>Fitness and Nutrition Assistant</h1>", unsafe_allow_html=True)
        
        left_col, center_col, right_col = st.columns([0.5, 3, 0.5])
        
        with center_col:
            st.header("Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if not username or not password:
                        st.error("Please fill in all fields!")
                        return
                        
                    user_data = self.user_manager.authenticate_user(username, password)
                    if user_data:
                        st.session_state.authenticated = True
                        st.session_state.user_data = user_data
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")

            st.markdown("---") 
            col1, col2 = st.columns([4, 2])
            with col1:
                st.write("Don't have an account yet?", unsafe_allow_html=True)
            with col2:
                if st.button("Sign up now!"):
                    st.session_state.current_page = "Register"
                    st.rerun()

            if st.session_state.registration_success:
                st.success("Registration successful! Please login with your credentials.")
                st.session_state.registration_success = False

    def show_registration_page(self):
        st.markdown("<h1 style='text-align: center;'>Fitness and Nutrition Assistant</h1>", unsafe_allow_html=True)
        
        st.header("User Registration")

        # Aggiungiamo un pulsante "Torna indietro" in alto
        if st.button("← Back to Login"):
            st.session_state.current_page = "Login"
            st.rerun()
            
        with st.form("user_registration"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Campi esistenti
            Name = st.text_input("Name")
            Age = st.number_input("Age", min_value=18, max_value=100)
            Sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            Weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0)
            Height = st.number_input("Height (cm)", min_value=130.0, max_value=250.0)
            
            Activity_level = st.select_slider(
                "Activity Level",
                options=["Sedentary", "Light", "Moderate", "Active", "Very Active"]
            )
            
            Goal = st.selectbox(
                "Goal",
                ["Weight Loss", "Weight Gain", "Maintenance"]
            )
            Preferences = st.multiselect(
                "Dietary Preferences",
                ["Omnivore", "Vegetarian", "Gluten-Free"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Register")
                
            if submitted:
                if not all([username, password, confirm_password, Name, Preferences]):
                    st.error("Please fill in all required fields!")
                    return
                    
                if password != confirm_password:
                    st.error("Passwords do not match!")
                    return

                user_data = {
                    "User_id": f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "username": username,
                    "password": password, 
                    "Name": Name,
                    "Age": Age,
                    "Sex": Sex,
                    "Weight": Weight,
                    "Height": Height,
                    "Activity_level": Activity_level,
                    "Goal": Goal,
                    "Preference": Preferences,
                    "Daily_calories": self.user_manager.calculate_daily_calories(Weight, Height, Age, Sex, Activity_level)
                }
                
                success, message = self.user_manager.create_user_node(user_data)
                if success:
                    st.success("Registration successful! Please login to continue.")
                    st.session_state.current_page = "Login"
                    st.rerun()
                else:
                    st.error(f"Registration failed: {message}")

    def show_workout_plan_page(self):
        """Display the workout plan page with daily breakdown of exercises and generation capability."""
        if not st.session_state.authenticated:
            st.warning("Please login first!")
            return
        
        # Add generate plan button first
        st.header("Generate New Plan")
        col1, col2 = st.columns([2, 1])
        with col1:
            generate_clicked = st.button(
                "Generate Personalized Workout Plan",
                help="Click to generate a new workout plan based on your profile",
                type="primary"
            )
            
        # Now add the title below
        st.title("Your Workout Plan")
        
        if not st.session_state.user_data or 'UserID' not in st.session_state.user_data:
            st.error("User data not found. Please try logging in again.")
            return

        # Initialize workout plan manager if needed
        if 'workout_plan_manager' not in st.session_state:
            st.session_state.workout_plan_manager = WorkoutPlanGraphRAG()
            
        # Load saved workout plan if not already in session
        if 'current_workout_plan' not in st.session_state:
            with st.spinner("Loading your saved workout plan..."):
                saved_plan = self.user_manager.get_latest_workout_plan(st.session_state.user_data['UserID'])
                if saved_plan:
                    st.session_state.current_workout_plan = saved_plan
                    st.success("✅ Loaded your most recent workout plan!")

        
        if generate_clicked:
            with st.spinner("🔄 Generating your personalized workout plan... This may take a moment."):
                try:
                    # Use existing user parameters directly from session state
                    user_data = st.session_state.user_data
                    
                    # Create compatible parameters structure for the API
                    # Rename fields to match what's expected in the find_similar_users_workouts method
                    user_params = {
                        "goal": user_data.get("Goal"),
                        "activity_level": user_data.get("ActivityLevel"),
                        "weight": user_data.get("Weight"),
                        "height": user_data.get("Height"),
                        "age": user_data.get("Age"),
                        "gender": user_data.get("Gender"),
                        "preferences": user_data.get("Preferences", []),
                        "name": user_data.get("Name", ""),
                        "username": user_data.get("Username", "")
                    }
                    
                    # Generate new plan
                    workout_plan = st.session_state.workout_plan_manager.create_personalized_workout_plan(user_params)
                    
                    if workout_plan:
                        st.success("✨ New workout plan generated successfully!")
                        # Store the workout plan in session state
                        st.session_state.current_workout_plan = workout_plan
                        # Force a rerun to show the new plan
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate workout plan. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error generating workout plan: {str(e)}")
                    return
                    
        # Get workout plan from session state
        workout_plan = getattr(st.session_state, 'current_workout_plan', None)
        
        if not workout_plan:
            st.info("👋 No workout plan found. Click the button above to generate your personalized plan!")
            return

        # Display summary information
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Goal", 
                    value=workout_plan['summary']['goal']
                )
            with col3:
                st.metric(
                    label="Generated On",
                    value=workout_plan['summary']['generated_at'].split('T')[0]
                )

        # Create tabs for each day
        day_tabs = st.tabs([f"Day {day['day']}" for day in workout_plan['days']])
        
        # Populate each day's tab
        for idx, (day_tab, day_data) in enumerate(zip(day_tabs, workout_plan['days'])):
            with day_tab:
                # Daily summary
                st.subheader(f"Day {day_data['day']} - {', '.join(day_data['muscle_groups'])}")
                
                # Create expandable sections for each exercise
                for exercise in day_data['exercises']:
                    with st.expander(f"🏋️‍♂️ {exercise['name']}", expanded=True):
                        # Exercise details in columns
                        cols = st.columns(3)
                        cols[0].metric("Sets", exercise['sets'])
                        cols[1].metric("Reps", exercise['reps'])
                        cols[2].metric("Rest", f"{exercise['rest_seconds']}s")
                        
                        # Target muscles
                        st.markdown("**Target Muscles:**")
                        st.write(", ".join(exercise['target_muscles']))
                        
                        # Form notes
                        st.markdown("**Form Notes:**")
                        st.info(exercise['notes'])

                # Add a visual separator between days
                st.divider()

        # Add export functionality
        st.divider()
        st.subheader("Export Workout Plan")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as CSV"):
                # Convert workout plan to CSV format
                csv_data = self._convert_workout_plan_to_csv(workout_plan)
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"workout_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export as Excel"):
                # Convert workout plan to Excel format
                excel_data = self._convert_workout_plan_to_excel(workout_plan)
                
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"workout_plan_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        st.divider()
        st.subheader("Save Plan to Your Profile")
        save_plan_col1, save_plan_col2 = st.columns([3, 1])
        with save_plan_col1:
            st.write("Save this workout plan to your profile for future reference.")
        with save_plan_col2:
            if st.button("Save Plan to Profile", type="primary"):
                with st.spinner("Saving your plan..."):
                    success = self.user_manager.save_workout_plan(
                        st.session_state.user_data['UserID'], 
                        st.session_state.current_workout_plan
                    )
                    if success:
                        st.success("✅ Workout plan saved successfully to your profile!")
                    else:
                        st.error("❌ Failed to save workout plan. Please try again.")
    
    def _convert_workout_plan_to_csv(self, workout_plan):
        """Convert workout plan to CSV format for export"""
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Day', 'Muscle Groups', 'Exercise', 'Sets', 'Reps', 'Rest (s)', 'Target Muscles', 'Notes'])
        
        # Write data
        for day in workout_plan['days']:
            day_num = day['day']
            muscle_groups = ', '.join(day['muscle_groups'])
            
            for exercise in day['exercises']:
                writer.writerow([
                    day_num,
                    muscle_groups,
                    exercise['name'],
                    exercise['sets'],
                    exercise['reps'],
                    exercise['rest_seconds'],
                    ', '.join(exercise['target_muscles']),
                    exercise['notes']
                ])
        
        return output.getvalue()

    def _convert_workout_plan_to_excel(self, workout_plan):
        """Convert workout plan to Excel format for export"""
        # Create a DataFrame with workout plan data
        rows = []
        for day in workout_plan['days']:
            day_num = day['day']
            muscle_groups = ', '.join(day['muscle_groups'])
            
            for exercise in day['exercises']:
                rows.append({
                    'Day': day_num,
                    'Muscle Groups': muscle_groups,
                    'Exercise': exercise['name'],
                    'Sets': exercise['sets'],
                    'Reps': exercise['reps'],
                    'Rest (s)': exercise['rest_seconds'],
                    'Target Muscles': ', '.join(exercise['target_muscles']),
                    'Notes': exercise['notes']
                })
        
        df = pd.DataFrame(rows)
        
        # Get summary info for a second sheet
        summary_data = {
            'Goal': [workout_plan['summary']['goal']],
            'Generated At': [workout_plan['summary']['generated_at']],
            'Days in Plan': [len(workout_plan['days'])]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create an Excel file with two sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Workout Plan', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output.getvalue()

    def _convert_meal_plan_to_csv(self, meal_plan):
        """Convert meal plan to CSV format for export"""
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Day', 'Meal Type', 'Food', 'Portion (g)', 'Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)'])
        
        # Write data
        for day in meal_plan['days']:
            day_num = day['day']
            
            for meal_type, meal_icon in {
                'breakfast': '🌅 Breakfast',
                'lunch': '🌞 Lunch',
                'dinner': '🌙 Dinner',
                'snack': '🍎 Snack'
            }.items():
                if meal_type in day['meals']:
                    meal_data = day['meals'][meal_type]
                    
                    for food in meal_data['foods']:
                        writer.writerow([
                            day_num,
                            meal_icon,
                            food['name'],
                            food['portion'],
                            food['calories'],
                            food['proteins'],
                            food['carbohydrates'],
                            food['fats']
                        ])
        
        return output.getvalue()

    def _convert_meal_plan_to_excel(self, meal_plan):
        """Convert meal plan to Excel format for export"""
        # Create a DataFrame for each day's meals
        days_dfs = []
        
        for day in meal_plan['days']:
            day_rows = []
            
            for meal_type, meal_icon in {
                'breakfast': '🌅 Breakfast',
                'lunch': '🌞 Lunch',
                'dinner': '🌙 Dinner',
                'snack': '🍎 Snack'
            }.items():
                if meal_type in day['meals']:
                    meal_data = day['meals'][meal_type]
                    
                    for food in meal_data['foods']:
                        day_rows.append({
                            'Day': day['day'],
                            'Meal Type': meal_icon,
                            'Food': food['name'],
                            'Portion (g)': food['portion'],
                            'Calories': food['calories'],
                            'Protein (g)': food['proteins'],
                            'Carbs (g)': food['carbohydrates'],
                            'Fats (g)': food['fats']
                        })
            
            days_dfs.append(pd.DataFrame(day_rows))
        
        # Combine all days
        all_meals_df = pd.concat(days_dfs, ignore_index=True)
        
        # Create a summary dataframe
        summary_data = {
            'Daily Calorie Target': [meal_plan['summary']['daily_calorie_target']],
            'Generated At': [meal_plan['summary']['generated_at']],
            'Days in Plan': [len(meal_plan['days'])]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Create an Excel file with two sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_meals_df.to_excel(writer, sheet_name='Meal Plan', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output.getvalue()

    
if __name__ == "__main__":
    app = StreamlitInterface()
    app.run()