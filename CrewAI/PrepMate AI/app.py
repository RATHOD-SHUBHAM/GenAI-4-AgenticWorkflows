import streamlit as st
from src.crew import run_crew

# streamlit code for viewing document
st.set_page_config(layout="wide",
                   page_title="PrepMateAI",
                   page_icon="🦦",
                   initial_sidebar_state="collapsed"
                   )

# hide hamburger and customize footer
hide_menu = """
    <style>

    #MainMenu {
        visibility:hidden;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: grey;
        text-align: center;
    }

    </style>

    <div class="footer">
        <p>'With 🫶️ from Shubham Shankar.'</p>
    </div>

"""

# Styling ----------------------------------------------------------------------
st.image("icon.jpg", width=85)
st.title("PrepMateAI")
st.subheader(" Prep Smarter, Meet Better..🦦")
st.write("Precision-Powered by Agents.")
st.markdown(hide_menu, unsafe_allow_html=True)

# Intro ----------------------------------------------------------------------

st.write(
    """
    Hi 👋, I'm **:red[Shubham Shankar]**, and Welcome to **:green[PrepMateAI]**! :rocket: **:blue[PrepMateAI]**, streamlines your meeting preparation by conducting in-depth research, gathering actionable insights, and crafting strategic plans tailored to your agenda. Equipped with an integrated **:orange[email automation]**, PrepMate AI ensures you're always a step ahead.
    **:green[PrepMateAI]** combines intelligence and efficiency to make every meeting impactful and effortless.✨
    """
)

st.markdown('---')

st.write(
    """
    ### App Interface!!

    :dog: The web app has an easy-to-use interface. 

    1] **:red[Meeting Participants]**: Input the email addresses of meeting participants, excluding yourself.
    
    2] **:blue[Meeting Context]**: Provide the context of the meeting to guide the agent's research.
    
    3] **:orange[Meeting Objective]**: Specify your objective for the meeting to tailor the strategic plan.
    """
)

st.markdown('---')

st.error(
    """
    Connect with me on [**Github**](https://github.com/RATHOD-SHUBHAM) and [**LinkedIn**](https://www.linkedin.com/in/shubhamshankar/). ✨
    """,
    icon="🧟‍♂️",
)

st.markdown('---')

# Title of the app
st.title("PrepMate AI: Your Meeting Preparation Assistant")

st.subheader("Input Meeting Essentials")
meeting_participants = st.text_input("What are the emails for the participants (other than you) in the meeting?", "sundarpic@google.com, tim@apple.com, satya@microsoft.com")
meeting_context = st.text_input("What is the context of the meeting?", "The participant of the meeting has decided to that they want to find a new industry to invest in AI.")
meeting_objective = st.text_input("What is your objective for this meeting?", "The participant of the meeting has decided to that they want to find a new industry to invest in AI.")

if st.button('RUN'):
    if meeting_participants is not None and meeting_context is not None and meeting_objective is not None:
        with st.status("Agents are working, Do Not Disturb ...!"):
            info = run_crew(meeting_participants, meeting_context, meeting_objective)

            st.write(info)