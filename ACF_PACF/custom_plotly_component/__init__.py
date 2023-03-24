import streamlit as st
from streamlit.components.v1 import declare_component

_component_func = declare_component("custom_plotly_component")

def custom_plotly_chart(**kwargs):
    return _component_func(**kwargs)
