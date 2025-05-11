import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.Recommendation import process_user_assessment

# validation styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        font-weight: 600;
        color: #333333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
            
    /* Section headers styling */
    .section-header {
        color: #333333;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }
            
    .section-banner {
        background-color: #F6F6F8;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Form styling */
    .stForm {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
            
    /* Input field styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #E0E0E0;
        padding: 10px;
    }
    
    /* Slider styling */
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
            
    /* Radio button styling */
    .stRadio > div {
        padding: 10px 0;
    }
            
    /* Select box styling */
    .stSelectbox > div > div > div {
        border-radius: 5px;
        border: 1px solid #E0E0E0;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4B7BF5;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #3A6AD4;
        color: white;
    }
            
    /* Section divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #e0e0e0;
    }
    
    /* Results cards styling */
    .results-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Gauge styling */
    .gauge-container {
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Recommendations styling */
    .recommendation-item {
        padding: 0.5rem 0;
    }
    
.error-border {
    border: 2px solid red !important;
    border-radius: 4px;
}
.error-message {
    color: red;
    font-size: 0.8em;
    margin-top: -1em;
    margin-bottom: 1em;
}
</style>
""", unsafe_allow_html=True)

# initialize session state for validation
if 'validation_errors' not in st.session_state:
    st.session_state.validation_errors = {}
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False

st.title("Mental Health Prediction & Support")
st.write(
    "This assessment collects information about your lifestyle and well-being. "
    "Your answers will be used to provide insights in your mental health and offer personalized support and recommendations. "
    "Your answers will remain confidential and anonymous. The results of this assessment will not be shared with anyone. "
    "**Note:** This is not a diagnosis."
)


with st.form(key='survey_form'):
    # Section 1: Personal Information
    st.markdown("<h2 class='section-header'>Personal Information</h2>", unsafe_allow_html=True)
    
    gender = st.selectbox(
        "What is your assigned sex at birth?",
        options=["", "Male", "Female"],
        index=0,
        key="gender"
    )
    
    age = st.number_input(
        "What is your age?",
        min_value=0,
        max_value=120,
        value=0,
        key="age"
    )
    
    # Section 2: Academics and Work
    st.markdown("<h2 class='section-header'>Academics & Work</h2>", unsafe_allow_html=True)
    
    work_study_hours = st.slider(
        "How many hours per day do you work/study?",
        min_value=0,
        max_value=24,
        value=0,
        key="work_study_hours"
    )
    
    academic_pressure = st.slider(
        "Rate your academic pressure:",
        min_value=1,
        max_value=5,
        value=1,
        key="academic_pressure_slider"
    )
    
    work_pressure = st.slider(
        "Rate your work pressure (1 = Very Low, 5 = Very High)",
        min_value=1,
        max_value=5,
        value=1,
        key="work_pressure_slider"
    )
    
    study_satisfaction = st.slider(
        "Rate your study satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):",
        min_value=1,
        max_value=5,
        value=1,
        key="study_satisfaction_slider"
    )
    
    job_satisfaction = st.slider(
        "Rate your job satisfaction:",
        min_value=1,
        max_value=5,
        value=1,
        key="job_satisfaction_slider"
    )
    
    # Section 3: Lifestyle & Health
    st.markdown("<h2 class='section-header'>Lifestyle & Health</h2>", unsafe_allow_html=True)
    
    sleep_duration = st.selectbox(
        "How much sleep do you typically get?",
        options=["", "Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"],
        index=0,
        key="sleep_duration"
    )
    
    dietary_habits = st.selectbox(
        "How would you describe your dietary habits?",
        options=["", "Healthy", "Moderate", "Unhealthy"],
        index=0,
        key="dietary_habits"
    )
    
    # Section 4: Mental Health History
    st.markdown("<h2 class='section-header'>Mental Health History</h2>", unsafe_allow_html=True)
    
    family_history = st.selectbox(
        "Is there a history of mental illness in your family?",
        options=["", "Yes", "No"],
        index=0,
        key="family_history"
    )
    
    suicidal_thoughts = st.selectbox(
        "Have you ever had suicidal thoughts?",
        options=["", "Yes", "No"],
        index=0,
        key="suicidal_thoughts"
    )

    # Section 5: Financial Stress
    st.markdown("<h2 class='section-header'>Financial Stress</h2>", unsafe_allow_html=True)
    
    financial_stress = st.slider(
        "Rate your financial stress level:",
        min_value=1,
        max_value=5,
        value=1,
        key="financial_stress_slider"
    )
    
    # if there are validation errors
    if st.session_state.validation_errors:
        st.error("Please fill in all required fields")
        for field, error in st.session_state.validation_errors.items():
            st.markdown(f'<p class="error-message">{error}</p>', unsafe_allow_html=True)
    
    submitted = st.form_submit_button("Submit")

# handle form submission and validation
if submitted:
    # reset validation errors
    st.session_state.validation_errors = {}
    st.session_state.form_submitted = True
    
    # validate non-slider fields
    required_fields = {
        'gender': "Gender is required",
        'age': "Age is required",
        'sleep_duration': "Sleep duration information is required",
        'dietary_habits': "Dietary habits information is required",
        'suicidal_thoughts': "Suicidal thoughts information is required",
        'family_history': "Family history information is required"
    }
    
    for field, error_msg in required_fields.items():
        if field == 'age':
            # check number fields
            if locals()[field] is None or locals()[field] == 0:
                st.session_state.validation_errors[field] = error_msg
        else:
            # check text fields
            if not locals()[field]:
                st.session_state.validation_errors[field] = error_msg
    
    # if no validation errors, process the form
    if not st.session_state.validation_errors:
        # collect all form data
        assessment_data = {
            'gender': gender,
            'age': age,
            'family_history': family_history,
            'work_study_hours': work_study_hours,
            'academic_pressure': academic_pressure,
            'work_pressure': work_pressure,
            'study_satisfaction': study_satisfaction,
            'job_satisfaction': job_satisfaction,
            'sleep_duration': sleep_duration,
            'dietary_habits': dietary_habits,
            'suicidal_thoughts': suicidal_thoughts,
            'financial_stress': financial_stress
        }
        
        results = process_user_assessment(assessment_data)
        st.success("Assessment submitted successfully!")
        
        logreg_pred = results['logreg_prediction']
        logreg_confidence = results['individual_confidence']['logistic']
        
        # determine risk level based on logistic regression model's prediction and confidence
        if logreg_pred: # if it predicts depression
            if logreg_confidence > 75:
                risk_level = "High"
            else:
                risk_level = "Moderate"
        else:
            if logreg_confidence > 75:
                risk_level = "Low"
            else:
                risk_level = "Moderate"
        
        # Display risk level
        st.markdown("<h1>Your Results</h1>", unsafe_allow_html=True)
        
        if risk_level == "Low":
            st.markdown("### Risk Level: <span style='color:green'>Low</span>", unsafe_allow_html=True)
        elif risk_level == "Moderate":
            st.markdown("### Risk Level: <span style='color:orange'>Moderate</span>", unsafe_allow_html=True)
        else:
            st.markdown("### Risk Level: <span style='color:red'>High</span>", unsafe_allow_html=True)

        st.write(f"Our machine learning model predicted this result with: {logreg_confidence:.1f}% confidence. **Note:** This is not a diagnosis.")
        

        # Recommendations section
        st.markdown("<div class='section-banner'><h1>Personalized Recommendations</h1></div>", unsafe_allow_html=True)
        st.markdown("<p>Based on your results, we suggest you:</p>", unsafe_allow_html=True)
        
        # base recommendations that apply to everyone
        base_recommendations = [
            "Practice regular physical exercise for at least 30 minutes daily",
            "Maintain a consistent sleep schedule",
            "Stay connected with friends and family"
        ]
        
        if risk_level == "Low":
            recommendations = [
                "Continue your current positive lifestyle habits",
                "Consider starting a mindfulness or meditation practice",
                "Set regular check-ins with yourself to monitor your mental health"
            ] + base_recommendations
        elif risk_level == "Moderate":
            recommendations = [
                "Consider speaking with a mental health professional",
                "Practice stress-reduction techniques daily",
                "Establish a support network of trusted friends or family",
                "Monitor your mood using a journal or mood tracking app"
            ] + base_recommendations
        else:
            recommendations = [
                "**Strongly recommended:** Schedule an appointment with a mental health professional",
                "Reach out to a trusted friend or family member about your feelings",
                "Contact a mental health crisis hotline if you need immediate support",
                "Create a daily routine that includes regular meals and exercise",
                "Avoid making major life decisions until you've consulted with a professional"
            ] + base_recommendations
        
        for rec in recommendations:
            st.markdown(f"<li style='margin-bottom: 0.5rem;'>{rec}</li>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            button_col1, gap, button_col2 = st.columns([1, 0.2, 1])
            with button_col1:
                st.button("Start Relaxation Exercise", use_container_width=True)
            with button_col2:
                st.button("Find Local Help", use_container_width=True)

        if risk_level == "High":
            st.markdown("---")
            st.markdown("### Crisis Resources")
            st.markdown("""
            - **National Suicide Prevention Lifeline (US):** 988 or 1-800-273-8255
            - **Crisis Text Line:** Text HOME to 741741
            - **The Trevor Project (for LGBTQ youth):** 1-866-488-7386 or text START to 678-678
            - **Veterans Crisis Line:** Dial 988 then Press 1, or text 838255
            - **Emergency Services:** 911 (US) or your local emergency number
            
            Remember, reaching out is a sign of strength. Help is available.
            """)

        st.markdown("---")
        st.info("Disclaimer: This assessment is based on a machine learning model and is not a substitute for " \
        " professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other" \
        " qualified health provider with any questions you may have regarding a medical condition. Never disregard" \
        " professional medical advice or delay in seeking it because of something you have read from this assessment.")

    else:
        # re-display the form with validation errors
        st.error("Please correct the errors and submit again")
