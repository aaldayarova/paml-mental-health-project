import streamlit as st
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports from sibling modules
sys.path.append(str(Path(__file__).parent))
from Recommendation import process_user_assessment

st.title("Mental Health Prediction & Support")

st.write(
    "This assessment collects information about your lifestyle and well-being. "
    "Your answers will be used to provide personalized insights and support. "
    "Your answers will remain confidential and anynonmous. The results of this assessment will not be shared with anyone. "
    "**Note:** This is not a diagnosis."
)

# Create the form
with st.form(key='survey_form'):
    # Section 1: Basic Information
    st.markdown("## Basic Information")
    gender = st.selectbox("What is your gender?", ["Male", "Female"], key='gender')
    age = st.number_input("What is your age?", min_value=1, max_value=100, value=20, key='age')

    st.markdown("---")

    # Section 2: Academic & Work
    st.markdown("## Academic & Work")
    academic_pressure = st.slider("Rate your academic pressure (1 = Very Low, 5 = Very High):", 1, 5, key='academic_pressure')
    work_pressure = st.slider("Rate your work pressure (1 = Very Low, 5 = Very High):", 1, 5, key='work_pressure')
    study_satisfaction = st.slider("Rate your study satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='study_satisfaction')
    job_satisfaction = st.slider("Rate your job satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='job_satisfaction')
    work_study_hours = st.number_input("How many hours per week do you work/study?", min_value=0.0, max_value=168.0, value=40.0, step=1.0, key='work_study_hours')

    st.markdown("---")

    # Section 3: Lifestyle & Health
    st.markdown("## Lifestyle & Health")
    sleep_duration = st.selectbox("How many hours do you sleep on average?", 
                                ["Less than 6", "6-7", "7-8", "8-9", "More than 9"], 
                                key='sleep_duration')
    dietary_habits = st.selectbox("How would you rate your dietary habits?", 
                                ["Poor", "Average", "Good"], 
                                key='dietary_habits')

    st.markdown("---")

    # Section 4: Mental Health History
    st.markdown("## Mental Health History")
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"], key='suicidal_thoughts')
    family_history = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"], key='family_history')

    st.markdown("---")

    # Section 5: Financial Stress
    st.markdown("## Financial Stress")
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
    
    st.header("Your Results")

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
    st.header("Personalized Recommendations")
    
    # Base recommendations that apply to everyone
    base_recommendations = [
        "Practice regular physical exercise for at least 30 minutes daily",
        "Maintain a consistent sleep schedule",
        "Stay connected with friends and family"
    ]
    
    # Risk-specific recommendations
    if risk_level == "Low":
        recommendations = base_recommendations + [
            "Continue your current positive lifestyle habits",
            "Consider starting a mindfulness or meditation practice",
            "Set regular check-ins with yourself to monitor your mental health"
        ]
    elif risk_level == "Moderate":
        recommendations = base_recommendations + [
            "Consider speaking with a mental health professional",
            "Practice stress-reduction techniques daily",
            "Establish a support network of trusted friends or family",
            "Monitor your mood using a journal or mood tracking app"
        ]
    else:  # High risk
        recommendations = base_recommendations + [
            "**Strongly recommended:** Schedule an appointment with a mental health professional",
            "Reach out to a trusted friend or family member about your feelings",
            "Contact a mental health crisis hotline if you need immediate support",
            "Create a daily routine that includes regular meals and exercise",
            "Avoid making major life decisions until you've consulted with a professional"
        ]
    
    for rec in recommendations:
        st.write(f"• {rec}")

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
