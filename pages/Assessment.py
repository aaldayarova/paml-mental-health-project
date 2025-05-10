import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.Recommendation import process_user_assessment

# validation styling
st.markdown("""
<style>
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
    st.header("Personal Information")
    
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
    st.header("Academics and Work")
    
    work_study_hours = st.slider(
        "How many hours do you work/study per day?",
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
        "Rate your work pressure:",
        min_value=1,
        max_value=5,
        value=1,
        key="work_pressure_slider"
    )
    
    study_satisfaction = st.slider(
        "Rate your satisfaction with your studies:",
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
    
    # Section 3: Lifestyle
    st.header("Lifestyle")
    
    sleep_duration = st.selectbox(
        "How much sleep do you typically get?",
        options=["", "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
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
    st.header("Mental Health History")
    
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
        st.header("Assessment Results")
        
        if risk_level == "Low":
            st.markdown("### Risk Level: <span style='color:green'>Low</span>", unsafe_allow_html=True)
        elif risk_level == "Moderate":
            st.markdown("### Risk Level: <span style='color:orange'>Moderate</span>", unsafe_allow_html=True)
        else:
            st.markdown("### Risk Level: <span style='color:red'>High</span>", unsafe_allow_html=True)

        st.write(f"Our machine learning model predicted this result with: {logreg_confidence:.1f}% confidence. **Note:** This is not a diagnosis.")
        

        # Recommendations section
        st.header("Recommendations")
        st.markdown("Based on your results, we suggest you:", unsafe_allow_html=True)
        
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
            st.markdown(f"- {rec}")
    else:
        # re-display the form with validation errors
        st.error("Please correct the errors and submit again")
