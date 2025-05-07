import streamlit as st

st.title("Mental Health Prediction & Support")

st.write(
    "This assessment collects information about your lifestyle and well-being. "
    "Your answers will be used to provide personalized insights and support. "
    "**Note:** This is not a diagnosis."
)

# Create the form
with st.form(key='survey_form'):
    # Section 1: Demographics
    st.markdown("## Demographics")
    st.selectbox("What is your gender?", ["Male", "Female", "Other", "Prefer not to say"], key='gender')
    st.text_input("What is your age?", key='age')

    st.markdown("---")

    # Section 2: Academic & Work Pressure
    st.markdown("## Academic & Work Pressure")
    st.slider("Rate your academic pressure (1 = Very Low, 5 = Very High):", 1, 5, key='academic_pressure')
    st.slider("Rate your work pressure (1 = Very Low, 5 = Very High):", 1, 5, key='work_pressure')

    st.markdown("---")

    # Section 3: Satisfaction
    st.markdown("## Satisfaction")
    st.slider("Rate your study satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='study_satisfaction')
    st.slider("Rate your job satisfaction (1 = Very Dissatisfied, 5 = Very Satisfied):", 1, 5, key='job_satisfaction')

    st.markdown("---")

    # Section 4: Sleep Patterns
    st.markdown("## Sleep Patterns")
    st.number_input("On average, how many hours do you sleep per night?", min_value=0.0, step=0.5, format="%.1f", key='sleep_duration')
    st.radio("Rate your sleep quality:", ["Very Poor", "Poor", "Average", "Good", "Excellent"], key='sleep_quality')

    st.markdown("---")

    # Section 5: Lifestyle
    st.markdown("## Lifestyle")
    st.radio("How would you describe your dietary habits?", ["Poor", "Average", "Good"], key='dietary_habits')
    st.number_input("How many hours per week do you work/study?", min_value=0.0, step=1.0, key='work_study_hours')

    st.markdown("---")

    # Section 6: Mental Health History
    st.markdown("## Mental Health History")
    st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"], key='suicidal_thoughts')
    st.radio("Do you have a family history of mental illness?", ["Yes", "No"], key='family_history')

    st.markdown("---")

    # Section 7: Financial & Emotional Stress
    st.markdown("## Financial & Emotional Stress")
    st.slider("Rate your financial stress (1 = Very Low, 5 = Very High):", 1, 5, key='financial_stress')

    st.markdown("---")

    # Submit button
    submitted = st.form_submit_button("Submit")

# Show results after submission
if submitted:
    st.success("Thank you for completing the assessment!")
    st.header("Your Results")

    
    depression_risk = 27  # placeholder
    

    st.subheader("Depression Risk")
    st.metric(label="Risk Level", value=f"{depression_risk}%")
    if depression_risk < 30:
        st.write("Low risk of depression.")
    elif depression_risk < 70:
        st.write("Moderate risk of depression.")
    else:
        st.write("High risk of depression. Consider seeking professional advice.")


    st.markdown("---")

    # Recommendations section
    st.header("Personalized Recommendations")
    st.write("""
    - Practice 20 minutes of relaxation or mindfulness daily.
    - Maintain a regular sleep schedule and reduce screen time before bed.
    - Reach out to a counselor or mental health professional if symptoms persist.
    - Stay connected with supportive friends and family.
    """)
