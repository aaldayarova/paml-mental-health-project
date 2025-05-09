import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from utils.Recommendation import process_user_assessment

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
    
    /* Action buttons styling */
    .action-button {
        background-color: #4B7BF5;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 5px;
        text-align: center;
        margin-top: 1rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .action-button.secondary {
        background-color: #8BA3E0;
    }
    
    .action-button:hover {
        background-color: #3A6AD4;
    }
</style>
""", unsafe_allow_html=True)

st.title("Mental Health Prediction & Support")

st.write(
    "This assessment collects information about your lifestyle and well-being."
    "Your answers will be used to provide insights in your mental health and offer personalized support and recommendations."
    "Your answers will remain confidential and anonymous. The results of this assessment will not be shared with anyone."
    "**Note:** This is not a diagnosis."
)

# Create the form
with st.form(key='survey_form'):
    # Section 1
    st.markdown("<h2 class='section-header'>Background Information</h2>", unsafe_allow_html=True)
    gender = st.selectbox("What is your gender?", ["Male", "Female"], key='gender')
    age = st.number_input("What is your age?", min_value=1, max_value=100, value=20, key='age')

    st.markdown("---")

    # Section 2
    st.markdown("<h2 class='section-header'>Academics & Work</h2>", unsafe_allow_html=True)
    academic_pressure = st.slider("Rate your academic pressure (1 = Very Low, 5 = Very High):", 1, 5, key='academic_pressure')
    work_pressure = st.slider("Rate your work pressure (1 = Very Low, 5 = Very High):", 1, 5, key='work_pressure')
    study_satisfaction = st.slider("Rate your study satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='study_satisfaction')
    job_satisfaction = st.slider("Rate your job satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='job_satisfaction')
    work_study_hours = st.number_input("How many hours per week do you work/study?", min_value=0.0, max_value=168.0, value=40.0, step=1.0, key='work_study_hours')

    st.markdown("---")

    # Section 3
    st.markdown("<h2 class='section-header'>Lifestyle & Health</h2>", unsafe_allow_html=True)
    sleep_duration = st.selectbox("How many hours do you sleep on average?", 
                                ["Less than 6", "6-7", "7-8", "8-9", "More than 9"], 
                                key='sleep_duration')
    dietary_habits = st.selectbox("How would you rate your dietary habits?", 
                                ["Poor", "Average", "Good"], 
                                key='dietary_habits')

    st.markdown("---")

    # Section 4
    st.markdown("<h2 class='section-header'>Mental Health History</h2>", unsafe_allow_html=True)
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"], key='suicidal_thoughts')
    family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"], key='family_history')

    st.markdown("---")

    # Section 5
    st.markdown("<h2 class='section-header'>Financial Stress</h2>", unsafe_allow_html=True)
    financial_stress = st.slider("Rate your financial stress (1 = Very Low, 5 = Very High):", 1, 5, key='financial_stress')

    st.markdown("---")

    # Submit button
    submitted = st.form_submit_button("Submit")

# Show results after submission
if submitted:
    st.success("Thank you for completing the assessment!")
    
    # Collect all form data
    assessment_data = {
        'gender': gender,
        'age': age,
        'academic_pressure': academic_pressure,
        'work_pressure': work_pressure,
        'study_satisfaction': study_satisfaction,
        'job_satisfaction': job_satisfaction,
        'sleep_duration': sleep_duration,
        'dietary_habits': dietary_habits,
        'work_study_hours': work_study_hours,
        'suicidal_thoughts': suicidal_thoughts,
        'financial_stress': financial_stress,
        'family_history': family_history
    }

    # Get predictions
    results = process_user_assessment(assessment_data)
    
    st.markdown("<h1>Your Results</h1>", unsafe_allow_html=True)

    # Display overall risk assessment
    st.subheader("Depression Risk Assessment")
    risk_score = results['confidence']
    st.metric(label="Risk Level", value=f"{risk_score:.1f}%")

    # Create three columns for individual model predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Linear Model", f"{results['individual_scores']['linear']:.1f}%")
    with col2:
        st.metric("SVM Model", f"{results['individual_scores']['svm']:.1f}%")
    with col3:
        st.metric("Logistic Model", f"{results['individual_scores']['logistic']:.1f}%")

    # Risk level interpretation
    if risk_score < 30:
        risk_level = "Low"
        st.write("Your responses indicate a **low risk** of depression.")
    elif risk_score < 70:
        risk_level = "Moderate"
        st.write("Your responses indicate a **moderate risk** of depression.")
    else:
        risk_level = "High"
        st.write("Your responses indicate a **high risk** of depression.")

    st.markdown("---")

    # Personalized recommendations based on risk level
    st.markdown("<div class='section-banner'><h1>Personalized Recommendations</h1></div>", unsafe_allow_html=True)
    st.markdown("<p>Based on your results, we suggest you:</p>", unsafe_allow_html=True)
    
    # Base recommendations that apply to everyone
    base_recommendations = [
        "Practice regular physical exercise for at least 30 minutes daily",
        "Maintain a consistent sleep schedule",
        "Stay connected with friends and family"
    ]
    
    # Risk-specific recommendations
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
    else:  # High risk
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


    # Add crisis resources if risk is high
    if risk_level == "High":
        st.markdown("---")
        st.markdown("### Crisis Resources")
        st.markdown("""
        - **National Suicide Prevention Lifeline (US):** 988 or 1-800-273-8255
        - **Crisis Text Line:** Text HOME to 741741
        - **Emergency Services:** 911 (US) or your local emergency number
        """)

    st.markdown("---")
    st.info("Remember: This assessment is not a diagnosis. If you're concerned about your mental health, please consult with a qualified mental health professional.")