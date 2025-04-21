import streamlit as st


if 'page' not in st.session_state:
    st.session_state['page'] = 'welcome'

if 'sleep_hours' not in st.session_state:
    st.session_state['sleep_hours'] = 0.0
if 'sleep_quality' not in st.session_state:
    st.session_state['sleep_quality'] = None
if 'sleep_consistency' not in st.session_state:
    st.session_state['sleep_consistency'] = None

if 'physical_activity_level' not in st.session_state:
    st.session_state['physical_activity_level'] = None

if 'social_interaction' not in st.session_state:
    st.session_state['social_interaction'] = None

if 'stress_level' not in st.session_state:
    st.session_state['stress_level'] = None

if 'mood_level' not in st.session_state:
    st.session_state['mood_level'] = None

if 'thought_patterns' not in st.session_state:
    st.session_state['thought_patterns'] = None

if 'screen_time_hours' not in st.session_state:
    st.session_state['screen_time_hours'] = None


def go_to_sleep_assessment():
    st.session_state['page'] = 'sleep_assessment'

def go_to_physical_activity():
    st.session_state['page'] = 'physical_activity'

def go_to_social_assessment():
    st.session_state['page'] = 'social_assessment'

def go_to_stress_indicators():
    st.session_state['page'] = 'stress_indicators'

def go_to_mood_assessment():
    st.session_state['page'] = 'mood_assessment'

def go_to_thought_patterns():
    st.session_state['page'] = 'thought_patterns'

def go_to_screen_time():
    st.session_state['page'] = 'screen_time'

def go_to_results():
    st.session_state['page'] = 'results'

def go_to_recommendations():
    st.session_state['page'] = 'recommendations'

def go_to_welcome():
    st.session_state['page'] = 'welcome'
    for key in st.session_state.keys():
        if key != 'page':
            st.session_state[key] = None
    st.session_state['sleep_hours'] = 0.0

if st.session_state['page'] == 'welcome':
    st.title("Mental Health Check-in")
    st.write("This questionnaire collects data on various aspects of your well-being.")
    st.write("This tool provides iinformation about your mental health, offering personalized support and recommendations.")
    st.write("Note: Your answers are confidential and will not be shared with anyone")
    st.write("Also note that this is not a diagnostic tool and should not be used as a substitute for professional help.")
    st.button("Begin Questionnaire", on_click=go_to_sleep_assessment)

elif st.session_state['page'] == 'sleep_assessment':
    st.header("Section 1 of 7: Sleep Patterns")
    st.number_input("How many hours do you sleep on average?", min_value=0.0, step=0.5, format="%.1f", key="sleep_hours")
    st.subheader("Rate your sleep quality (1=very poor, 5=excellent):")
    quality_options = [1, 2, 3, 4, 5]
    st.session_state['sleep_quality'] = st.segmented_control("", quality_options, key="sleep_quality_control")
    st.subheader("Over the last two weeks, has your sleep been consistent?")
    st.session_state['sleep_consistency'] = st.radio("Consistency", ["Yes", "No"], key="sleep_consistency_radio", index=None)
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_welcome):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_physical_activity):
        pass

elif st.session_state['page'] == 'physical_activity':
    st.header("Section 2 of 7: Physical Activity")
    st.subheader("How many days per week do you engage in moderate to vigorous physical activity (at least 30 minutes)?")
    st.session_state['physical_activity_level'] = st.slider("", 0, 7, 0, key="physical_activity_slider")
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_sleep_assessment):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_social_assessment):
        pass

elif st.session_state['page'] == 'social_assessment':
    st.header("Section 3 of 7: Social Interaction")
    st.subheader("How often do you interact socially with friends, family, or colleagues?")
    social_options = ["Rarely", "Sometimes", "Often"]
    st.session_state['social_interaction'] = st.radio("Social Interaction", social_options, key="social_interaction_radio", index=None)
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_physical_activity):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_stress_indicators):
        pass

elif st.session_state['page'] == 'stress_indicators':
    st.header("Section 4 of 7: Stress Indicators")
    st.subheader("How often have you felt overwhelmed or stressed in the past week?")
    stress_options = ["Rarely", "Sometimes", "Often", "Very Often"]
    st.session_state['stress_level'] = st.radio("Stress Level", stress_options, key="stress_level_radio", index=None)
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_social_assessment):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_mood_assessment):
        pass

elif st.session_state['page'] == 'mood_assessment':
    st.header("Section 5 of 7: Mood Tracking")
    st.subheader("How would you rate your overall mood today?")
    mood_options = ["Very Low", "Low", "Neutral", "Good", "Very Good"]
    st.session_state['mood_level'] = st.radio("Mood Level", mood_options, key="mood_level_radio", index=None)
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_stress_indicators):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_thought_patterns):
        pass

elif st.session_state['page'] == 'thought_patterns':
    st.header("Section 6 of 7: Thought Patterns")
    st.subheader("In the past week, how often have you experienced negative or worrying thoughts?")
    thought_options = ["Rarely", "Sometimes", "Often", "Very Often"]
    st.session_state['thought_patterns'] = st.radio("Thought Patterns", thought_options, key="thought_patterns_radio", index=None)
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_mood_assessment):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_screen_time):
        pass

elif st.session_state['page'] == 'screen_time':
    st.header("Section 7 of 7: Screen Time Usage")
    st.subheader("On average, how many hours per day do you spend on screens (phone, computer, TV)?")
    st.session_state['screen_time_hours'] = st.number_input("", min_value=0.0, step=0.5, format="%.1f", key="screen_time_input")
    cols_buttons = st.columns([1, 1])
    if cols_buttons[0].button("Back", on_click=go_to_thought_patterns):
        pass
    if cols_buttons[1].button("Next", on_click=go_to_results): 
        pass

elif st.session_state['page'] == 'results':
    st.header("Questionnaire Completed")
    st.write("Thank you for completing the questionnaire.")
    st.write("Results will be displayed here.")
    st.button("View Recommendations ", on_click=go_to_recommendations)

elif st.session_state['page'] == 'recommendations':
    st.header("Recommendations ")
    st.write("Recommendations based on your responses will be displayed here.")
    st.button("Back to Start", on_click=go_to_welcome)