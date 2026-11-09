import streamlit as st

st.set_page_config(page_title="MPI Heat Lab", page_icon=":material/thermostat:", layout="wide")

navigation = st.navigation(
    [
        st.Page("app_pages/solver.py", title="Live solver", icon=":material/play_circle:"),
        st.Page("app_pages/verification.py", title="Verification", icon=":material/verified:"),
        st.Page("app_pages/scaling.py", title="Parallel scaling", icon=":material/speed:"),
        st.Page("app_pages/surrogate.py", title="Neural surrogate", icon=":material/neurology:"),
        st.Page("app_pages/planner.py", title="Runtime planner", icon=":material/timer:"),
    ]
)
navigation.run()
