# Save this code in a file named "app.py"
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

# Generate data
n_points = 500
x = np.linspace(0, 10, n_points)

linear = x
sinusoidal = np.sin(x)
quadratic = x**2
random_noise = np.random.randn(n_points)
gaussian = np.exp(-(x - 5)**2 / 2)
logistic = 1 / (1 + np.exp(-x))
exponential = np.exp(x)
linear_sinusoidal = x + np.sin(x)

graphs = {
    "Random Noise": np.random.randn(n_points),
    "Linear": x,
    "Quadratic": x**2,
    "Sinusoidal": np.sin(x),
    "Linear + Sinusoidal": x + np.sin(x),
    "Gaussian": np.exp(-(x - 5)**2 / 2),
    "Logistic": 1 / (1 + np.exp(-x)),
    "Exponential": np.exp(x),
}

# Add more graphs to illustrate important concepts
random_walk = np.random.randn(n_points).cumsum()
graphs["Random Walk"] = random_walk

def add_noise(data, n_points):
    data_with_noise = data.copy()
    error_indices = np.random.choice(n_points, 5, replace=False)
    data_with_noise[error_indices] += np.random.uniform(-10, 10, 5)
    return data_with_noise

def plot_graph_acf_pacf(data, title, nlags):
    data_with_noise = add_noise(data, n_points)
    acf_values = acf(data, nlags=nlags)
    acf_values_noise = acf(data_with_noise, nlags=nlags)
    pacf_values = pacf(data, nlags=nlags)
    pacf_values_noise = pacf(data_with_noise, nlags=nlags)

    fig = make_subplots(rows=1, cols=5, subplot_titles=(title, f"{title} with Noise", "ACF", "ACF with Noise", "Combined PACF"))

    fig.add_trace(go.Scatter(x=x, y=data, name="Without Noise"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=data_with_noise, name="With Noise"), row=1, col=2)
    fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values, name="Without Noise"), row=1, col=3)
    fig.add_trace(go.Bar(x=list(range(len(acf_values_noise))), y=acf_values_noise, name="With Noise"), row=1, col=4)
    fig.add_trace(go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name="Without Noise"), row=1, col=5)
    fig.add_trace(go.Bar(x=list(range(len(pacf_values_noise))), y=pacf_values_noise, name="With Noise"), row=1, col=5)

    fig.update_layout(height=400, width=1500, showlegend=True)
    return fig

st.set_page_config(layout="wide", page_title="My Streamlit App", page_icon="💠")
st.title('ACF & PACF Visualiser')
nlags = st.slider('Number of lags', min_value=1, max_value=(n_points -1 )// 2, value=20)


for title, data in graphs.items():
    fig = plot_graph_acf_pacf(data, title, nlags)
    st.plotly_chart(fig)
