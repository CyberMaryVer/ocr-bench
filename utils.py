import streamlit as st

def init_session_state():
    if 'dataset_df' not in st.session_state:
        st.session_state['dataset_df'] = None
    if 'openrouter_api_key' not in st.session_state:
        st.session_state['openrouter_api_key'] = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'selected_result' not in st.session_state:
        st.session_state.selected_result = None