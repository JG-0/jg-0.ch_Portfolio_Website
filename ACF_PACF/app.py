import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from statsmodels.tsa.stattools import acf, pacf
import colorsys
# from custom_plotly_component import custom_plotly_chart   # I wanted to make it to listen and turn off autoresize if a user disabled a line, but that'll have to wait


# Define constants
PLOTLY_COLORS = [
    (31, 119, 180), #   "blue",
    (44, 160, 44), #   "green",
    (214, 39, 40), #   "red",
    (23, 190, 207), #   "cyan",
    (148, 103, 189), #   "magenta",
    (255, 127, 14), #   "yellow",
]

MIN_POINTS = 100
MAX_POINTS = 1000
DEFAULT_POINTS = 500

MIN_LAGS = 1
# MAX_LAGS = (n_points - 1) // 2
DEFAULT_LAGS = 220

MIN_NOISE_INTENSITY = 0.0
MAX_NOISE_INTENSITY = 20.0
DEFAULT_NOISE_INTENSITY = 10.0

MIN_NOISE_COUNT = 1
MAX_NOISE_COUNT = 100
DEFAULT_NOISE_COUNT = 5

MIN_NOISE_DURATION = 1
MAX_NOISE_DURATION = 20
DEFAULT_NOISE_DURATION = 1

MIN_NOISE_VARIABILITY = 0.0
MAX_NOISE_VARIABILITY = 1.0
DEFAULT_NOISE_VARIABILITY = 0.2


# Sets up Streamlit

st.set_page_config(
    layout="wide", page_title="Time-Series Analysis ACF & PACF Visualiser   by JG-0", page_icon="📈")
st.title("ACF & PACF Visualiser &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _by_ [JG-0](https://www.linkedin.com/in/Jerome--G)")

st.sidebar.markdown("""
## About

- This app was created by [JG-0](https://www.linkedin.com/in/Jerome--G) using Python and Streamlit, with source code available [here](https://github.com/JG-0/streamlit-apps).
- The ACF and PACF plots show the correlation structure of time series data.
- Different graph types can be selected from the dropdown menu.
- Noise can be added to the selected graph type using the sliders.

### Settings

- **Number of Points:** Turn this down for faster performance.
- **Number of Lags:** Control how wide the ACF and PACF graphs get.
- **Noise Intensity:** Tweak the size of each noise section.
- **Noise Count:** Control the total number of noise sections.
- **Noise Duration:** Set to 1 for individual noise sections, or crank it up for blocky noise sections.
- **Noise Variability:** Zero means flat noise sections, otherwise they get jumpy.

### Available Graph Types:

- **White Noise**
- **Random Walk**
- **Linear**
- **Quadratic**
- **Sinusoidal**
- **Linear + Sinusoidal**
- **Exponential**
- **Logistic**
- **Gaussian**

""")



# Defines the functions

def desaturate_color(color, desaturation_factor):
    """Converts an RGB color to an HLS color and desaturates it by the specified factor.
    Returns the desaturated color as an RGB tuple.
    """
    r, g, b = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0, s - desaturation_factor)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return int(r), int(g), int(b)


@st.cache_data
def add_noise(data, n_points, noise_intensity, noise_count, noise_duration, noise_variability):
    """Adds noise to a given time series data.
    data: time series data to add noise to
    n_points: number of data points in the time series
    noise_intensity: maximum intensity of each noise chunk
    noise_count: total number of noise pieces
    noise_duration: duration of each noise piece
    noise_variability: variability in the duration of each noise piece
    Returns the noisy time series data.
    """
    data_with_noise = data.copy()
    error_indices = np.random.choice(n_points - (noise_duration - 1), noise_count, replace=False)
    for idx in error_indices:
        initial_noise = np.random.uniform(-noise_intensity, noise_intensity)
        data_with_noise[idx] += initial_noise
        for i in range(1, noise_duration):
            extra_noise = initial_noise * (1 + np.random.uniform(-noise_variability, noise_variability))
            data_with_noise[idx + i] += extra_noise
    return data_with_noise

def create_figure(title, trace1, trace2):
    """Creates a plotly figure with two traces.
    title: title of the plot
    trace1: first trace to add to the figure
    trace2: second trace to add to the figure
    Returns the plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(trace2)
    fig.add_trace(trace1)
    fig.update_layout(
        title=f"{title}",
        height=400,
        showlegend=True,
        hovermode='x',
        legend=dict(orientation='h', y=-0.2),
        bargroupgap=0
    )
    return fig

def plot_default(graphs, title, nlags, color):
    """Creates a plot with the time default OR ACF OR PACF time series data and the noisy time series data.
    title: title of the plot
    nlags: number of lags to show in the ACF/PACF plot, empty for plot_default()
    color: color to use for the plot
    Returns the plotly figure.
    """
    trace1 = go.Scatter(x=graphs.index, y=graphs[title], name=f"{title}", line=dict(color=f"rgb{desaturate_color(color, 0.5)}"))
    trace2 = go.Scatter(x=graphs_noised.index, y=graphs_noised[title], name=f"{title} with Noise", line=dict(color=f"rgb{color}"))
    return create_figure(title, trace1, trace2)

def plot_ACF(graphs, title, nlags, color):
    acf_values = acf(graphs[title], nlags=nlags)
    acf_values_noise = acf(graphs_noised[title], nlags=nlags)
    trace2 = go.Bar(x=list(range(len(acf_values_noise))), y=acf_values_noise, name=f"{title} ACF with Noise", marker=dict(color=f"rgb{color}"))
    trace1 = go.Bar(x=list(range(len(acf_values))), y=acf_values, name=f"{title} ACF", marker=dict(color=f"rgb{desaturate_color(color, 0.5)}"))
    return create_figure(f"{title} ACF", trace1, trace2)

def plot_PACF(graphs, title, nlags, color):
    pacf_values = pacf(graphs[title], nlags=nlags)
    pacf_values_noise = pacf(graphs_noised[title], nlags=nlags)
    trace2 = go.Bar(x=list(range(len(pacf_values_noise))), y=pacf_values_noise, name=f"{title} PACF with Noise", marker=dict(color=f"rgb{color}"))
    trace1 = go.Bar(x=list(range(len(pacf_values))), y=pacf_values, name=f"{title} PACF", marker=dict(color=f"rgb{desaturate_color(color, 0.5)}"))
    return create_figure(f"{title} PACF", trace1, trace2)


# Sets up the Sliders and Dropdown Menu

if 'n_points' not in st.session_state:
    st.session_state['n_points'] = DEFAULT_POINTS
if 'nlags' not in st.session_state:
    st.session_state['nlags'] = DEFAULT_LAGS

def update_nlags():
    if st.session_state.nlags > (st.session_state.n_points - 1) // 2:
        st.session_state.nlags = (st.session_state.n_points - 1) // 2


n_points = st.slider('Number of points', min_value=MIN_POINTS, max_value=MAX_POINTS,
                     value=DEFAULT_POINTS, key="n_points", on_change=update_nlags)
nlags = st.slider('Number of lags', min_value=MIN_LAGS, max_value=(n_points - 1) // 2,
                  value=st.session_state.get('nlags', DEFAULT_LAGS), key="nlags")

noise_intensity = st.slider('Noise Intensity', min_value=MIN_NOISE_INTENSITY,
                            max_value=MAX_NOISE_INTENSITY, value=DEFAULT_NOISE_INTENSITY, key="noise_intensity", step=0.01)
noise_count = st.slider('Noise Count', min_value=MIN_NOISE_COUNT,
                        max_value=MAX_NOISE_COUNT, value=DEFAULT_NOISE_COUNT, key="noise_count")
noise_duration = st.slider(
    'Noise Duration', min_value=MIN_NOISE_DURATION, max_value=MAX_NOISE_DURATION, value=DEFAULT_NOISE_DURATION, key="noise_duration")
noise_variability = st.slider('Noise Variability', min_value=MIN_NOISE_VARIABILITY,
                              max_value=MAX_NOISE_VARIABILITY, value=DEFAULT_NOISE_VARIABILITY, key="noise_variability", step=0.01)

x = np.linspace(0, 10, n_points)
graphs_dict = {
    "White Noise": np.random.randn(n_points),
    "Random Walk": (1/3) * np.random.randn(n_points).cumsum(),
    "Linear": x,
    "Quadratic": (1/40) * x**2,
    "Sinusoidal": np.sin(x),
    "Linear + Sinusoidal": (2/3) * (x + np.sin(x)),
    "Exponential": (1/10000) * np.exp(x),
    "Logistic": 1 / (1 + np.exp(-x)),
    "Gaussian": np.exp(-(x - 5)**2 / 2),
}

graphs_dict_sel = st.multiselect('Select data pattern to plot (click):', list(
    graphs_dict.keys()), default=['White Noise', 'Random Walk'])
graphs_dict_selected = {key: value for key, value in graphs_dict.items() if key in graphs_dict_sel}


# Applies noise only to the selected series

graphs = pd.DataFrame(graphs_dict_selected)
graphs_noised = graphs.apply(add_noise, n_points=n_points, noise_intensity=noise_intensity, noise_count=noise_count, noise_duration=noise_duration ,noise_variability=noise_variability)
# df = pd.concat([graphs, graphs_noised], keys=["graphs", "graphs_noised"], axis=1)    # Just for fun you could create a 3D dataframe



# Iterates through the selected series and graphs the plots:

for index, key in enumerate(graphs.keys()):
    color = PLOTLY_COLORS[index % len(PLOTLY_COLORS)]
    col1, col2, col3 = st.columns(3)
    with col1:
        # st.write(key)
        st.plotly_chart(plot_default(graphs, key, nlags, color), use_container_width=True)
        # custom_plotly_chart(plot_default(key, nlags, color))   # I wanted to make it to listen and turn off autoresize if a user disabled a line, but it didn't work
    with col2:
        # st.write(key)
        st.plotly_chart(plot_ACF(graphs, key, nlags, color), use_container_width=True)
    with col3:
        # st.write(key)
        st.plotly_chart(plot_PACF(graphs, key, nlags, color), use_container_width=True)
