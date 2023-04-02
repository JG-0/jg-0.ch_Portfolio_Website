import pandas as pd
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression, Lasso
import plotly.graph_objects as go


sushi_colors = {
    'nigiri_salmon': 'salmon',
    'nigiri_ika': 'lightblue',
    'nigiri_maguro': 'red',
    'nigiri_ebi': 'pink',
    'maki_salmon': 'darkorange',
    'maki_pickles': 'forestgreen',
    'maki_maguro': 'firebrick',
    'uramaki_salmon': 'peachpuff',
    'uramaki_tuna': 'lightcoral',
    'uramaki_onion': 'gold',
    'uramaki_pickles': 'yellowgreen',
}

models = ['linear', 'lstsq', 'lasso']
colors = ['red', 'blue', 'purple']

model_colors = dict(zip(models, colors))

data = [
{'price': 7.95, 'uramaki_tuna': 6, 'uramaki_onion': 6},
{'price': 8.50, 'maki_salmon': 3, 'maki_maguro': 3, 'nigiri_salmon': 2, 'nigiri_maguro': 2},
{'price': 8.51, 'uramaki_salmon': 10},
{'price': 9.80, 'nigiri_salmon': 8},
{'price': 9.95, 'nigiri_salmon': 1, 'nigiri_ika': 1, 'nigiri_maguro': 1, 'nigiri_ebi': 1, 'uramaki_salmon': 3, 'uramaki_tuna': 3, 'uramaki_onion': 2},
{'price': 12.00, 'nigiri_salmon': 2, 'nigiri_ika': 1, 'nigiri_maguro': 1, 'nigiri_ebi': 2, 'uramaki_salmon': 3, 'uramaki_pickles': 2, 'uramaki_onion': 2},
{'price': 12.90, 'nigiri_salmon': 6, 'maki_salmon': 6},
{'price': 13.95, 'uramaki_salmon': 6, 'uramaki_tuna': 6, 'uramaki_onion': 6},
{'price': 13.951, 'maki_salmon': 6, 'maki_pickles': 6, 'maki_maguro': 6},
{'price': 14.95, 'nigiri_salmon': 8, 'uramaki_salmon': 6},
{'price': 14.951, 'nigiri_salmon': 4, 'nigiri_maguro': 4, 'maki_salmon': 3, 'maki_maguro': 3},
{'price': 18.95, 'nigiri_salmon': 2, 'nigiri_ika': 2, 'nigiri_maguro': 2, 'nigiri_ebi': 2, 'uramaki_salmon': 3, 'uramaki_tuna': 3, 'uramaki_onion': 3, 'uramaki_pickles': 3},
]

data1 = [
{'price': 7.95, 'uramaki_tuna': 6, 'uramaki_onion': 6},
{'price': 5.94, 'maki_salmon': 3, 'maki_maguro': 3, 'nigiri_salmon': 2, 'nigiri_maguro': 2},
{'price': 8.51, 'uramaki_salmon': 10},
{'price': 9.80, 'nigiri_salmon': 8},
{'price': 9.97, 'nigiri_salmon': 1, 'nigiri_ika': 1, 'nigiri_maguro': 1, 'nigiri_ebi': 1, 'uramaki_salmon': 3, 'uramaki_tuna': 3, 'uramaki_onion': 2},
{'price': 12.00, 'nigiri_salmon': 2, 'nigiri_ika': 1, 'nigiri_maguro': 1, 'nigiri_ebi': 2, 'uramaki_salmon': 3, 'uramaki_pickles': 2, 'uramaki_onion': 2},
{'price': 9.93, 'nigiri_salmon': 6, 'maki_salmon': 6},
{'price': 13.95, 'uramaki_salmon': 6, 'uramaki_tuna': 6, 'uramaki_onion': 6},
{'price': 9.94, 'maki_salmon': 6, 'maki_pickles': 6, 'maki_maguro': 6},
{'price': 14.95, 'nigiri_salmon': 8, 'uramaki_salmon': 6},
{'price': 11.89, 'nigiri_salmon': 4, 'nigiri_maguro': 4, 'maki_salmon': 3, 'maki_maguro': 3},
{'price': 18.95, 'nigiri_salmon': 2, 'nigiri_ika': 2, 'nigiri_maguro': 2, 'nigiri_ebi': 2, 'uramaki_salmon': 3, 'uramaki_tuna': 3, 'uramaki_onion': 3, 'uramaki_pickles': 3},
]


def preprocess_dataframe(data):
    df = pd.DataFrame(data).fillna(0)
    df = df.reindex(sorted(df.columns), axis=1)
    price = df.pop('price')
    df.insert(0, 'price', price)
    return df

def compute_lstsq_coefficients(df):
    A = df['price'].values
    B = df.drop(columns=['price']).values
    x, _, _, _ = lstsq(B, A)
    return dict(zip(df.columns[1:], x))

def compute_coefficients(model_class, X, y, alpha=None):
    if alpha is not None:
        model = model_class(alpha=alpha)
    else:
        model = model_class()
    model.fit(X, y)
    return dict(zip(X.columns, model.coef_))

def compute_linear_regression_coefficients(X, y):
    return compute_coefficients(LinearRegression, X, y)

def compute_lasso_coefficients(X, y, alpha=1.0):
    return compute_coefficients(Lasso, X, y, alpha=alpha)

def plot_combined_results(combined_results):
    # Sort the dataframe by the values in the "lstsq" column
    combined_results = combined_results.sort_values(by=("lstsq"))

    # Extract the index values (sushi types) and convert them to a list
    x = combined_results.index.tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(x=x, y=combined_results['linear'], name='Linear', marker_color='red'))
    fig.add_trace(go.Bar(x=x, y=combined_results['lstsq'], name='Lstsq', marker_color='blue'))
    fig.add_trace(go.Bar(x=x, y=combined_results['lasso'], name='Lasso', marker_color='purple'))

    fig.update_layout(
        title='Sushi Prices by Estimation Method',
        xaxis_title='Sushi Type',
        yaxis_title='Price',
        yaxis=dict(tickprefix='€'),
        barmode='group',
    )

    fig.show()

def plot_sushi_type_frequency(df_sorted, sushi_colors):
    df_sorted['box_id'] = range(1, len(df_sorted) + 1)
    x_labels = [f"Box {box_id} (€{price:.2f})" for box_id, price in zip(df_sorted['box_id'], df_sorted['price'])]

    fig1 = go.Figure()

    for sushi_type, color in sushi_colors.items():
        counts = df_sorted[sushi_type]
        fig1.add_trace(go.Bar(x=x_labels, y=counts, name=sushi_type, marker_color=color, text=counts, textposition='auto'))

    fig1.update_layout(
        title='Sushi Type Frequency in Boxes',
        xaxis=dict(title='Box Price (€)'),
        yaxis=dict(title='Sushi Count'),
        barmode='stack',
    )

    fig1.show()

def plot_actual_vs_estimated_price(df_sorted, combined_results, sushi_colors, model_colors):
    df_sorted = df_sorted.sort_values('price')
    sorted_sushi_types = df_sorted.drop(columns=['price', 'box_id']).columns
    # Remove Lasso from combined_results
    combined_results = combined_results.loc[sorted_sushi_types, ['linear', 'lstsq']]

    x_labels = [f"Box {box_id} (€{price:.2f})" for box_id, price in zip(df_sorted['box_id'], df_sorted['price'])]
    fig = go.Figure()

    # Add traces for each method
    for method, method_results in combined_results.items():
        estimated_prices = df_sorted[sorted_sushi_types].values * method_results.values.reshape(1, -1)
        method_results = estimated_prices.sum(axis=1)
        fig.add_trace(go.Scatter(x=method_results, y=df_sorted['price'], name=method.capitalize(), mode='markers', marker=dict(color=model_colors[method], size=5)))

    # Add diagonal y=x line
    max_price = df_sorted['price'].max()
    fig.add_trace(go.Scatter(x=[0, max_price], y=[0, max_price], name='y=x', mode='lines', line=dict(color='black', dash='dash')))

    # Layout settings
    fig.update_layout(title='Actual vs Estimated Price per Box', xaxis_title='Estimated Price (€)', yaxis_title='Actual Price (€)', yaxis_tickprefix='€', xaxis=dict(tickangle=-45, tickfont=dict(size=10)), yaxis=dict(scaleanchor="x", scaleratio=1), width=600, height=600)

    fig.show()

def plot_actual_vs_estimated_price_per_component(df_sorted, combined_results, sushi_colors):
    x_labels = [f"Box {box_id} (€{price:.2f})" for box_id, price in zip(df_sorted['box_id'], df_sorted['price'])]
    fig = go.Figure()

    for model in ['linear']:
        sushi_components = []
        for sushi_type, color in sushi_colors.items():
            estimated_prices = df_sorted[sushi_type] * combined_results.loc[sushi_type, model]
            hover_text = [f'{sushi_type}: {count}<br>Estimated Price: {price:.2f}' for count, price in zip(df_sorted[sushi_type], estimated_prices)]
            sushi_components.append(go.Bar(x=x_labels, y=estimated_prices, name=sushi_type + f' ({model.capitalize()})', marker_color=color, hovertemplate='<br>'.join(['%{text}', '']), text=hover_text))

        offset = -0.25 if model == 'linear' else 0
        for trace in sushi_components:
            trace.update(width=0.2, offset=offset)
            fig.add_trace(trace)

    fig.add_trace(go.Bar(x=x_labels, y=df_sorted['price'], name='Actual Price', marker_color='black', width=0.3, offset=-0.18, base=0, hovertemplate='Actual Price: %{y}'))

    fig.update_layout(
        title='Actual Price vs. Estimated Price (Per-Component)',
        xaxis=dict(title='Box Price (€)'),
        yaxis=dict(title='Price (€)'),
        barmode='stack',
        legend=dict(x=1, y=0.5, title_text='Sushi Type'),
    )

    fig.update_traces(marker_line_color='rgba(0, 0, 0, 0.3)', marker_line_width=1, selector=dict(type='bar'))
    fig.show()

    fig = go.Figure()
    for model in ['lstsq']:
        sushi_components = []
        for sushi_type, color in sushi_colors.items():
            estimated_prices = df_sorted[sushi_type] * combined_results.loc[sushi_type, model]
            hover_text = [f'{sushi_type}: {count}<br>Estimated Price: {price:.2f}' for count, price in zip(df_sorted[sushi_type], estimated_prices)]
            sushi_components.append(go.Bar(x=x_labels, y=estimated_prices, name=sushi_type + f' ({model.capitalize()})', marker_color=color, hovertemplate='<br>'.join(['%{text}', '']), text=hover_text))

        offset = -0.25 if model == 'linear' else 0
        for trace in sushi_components:
            trace.update(width=0.2, offset=offset)
            fig.add_trace(trace)

    fig.add_trace(go.Bar(x=x_labels, y=df_sorted['price'], name='Actual Price', marker_color='black', width=0.3, offset=-0.18, base=0, hovertemplate='Actual Price: %{y}'))

    fig.update_layout(
        title='Actual Price vs. Estimated Price (Per-Component)',
        xaxis=dict(title='Box Price (€)'),
        yaxis=dict(title='Price (€)'),
        barmode='stack',
        legend=dict(x=1, y=0.5, title_text='Sushi Type'),
    )

    fig.update_traces(marker_line_color='rgba(0, 0, 0, 0.3)', marker_line_width=1, selector=dict(type='bar'))
    fig.show()


def runmain(data, sushi_colors, model_colors):
    df = preprocess_dataframe(data)
    X = df.drop(columns=['price'])
    y = df['price']

    lstsq_results = compute_lstsq_coefficients(df)
    linear_results = compute_linear_regression_coefficients(X, y)
    lasso_results = compute_lasso_coefficients(X, y)

    combined_results = pd.DataFrame({
        'linear': linear_results,
        'lstsq': lstsq_results,
        'lasso': lasso_results
    })

    df_sorted = df.sort_values('price')
    df_sorted['box_id'] = range(1, len(df_sorted) + 1)

    plot_sushi_type_frequency(df_sorted, sushi_colors)
    plot_combined_results(combined_results)
    plot_actual_vs_estimated_price(df_sorted, combined_results, sushi_colors, model_colors)
    plot_actual_vs_estimated_price_per_component(df_sorted, combined_results, sushi_colors)
    print(combined_results.sort_values(by=("lstsq")))


runmain(data, sushi_colors, model_colors)
runmain(data1, sushi_colors, model_colors)



# save the results from combined_results to a global variable from data and data1, then calculate and visualise their difference

# plot the discount price breakdown but with the black bars of the non-discount prices

# plot a single barchart with both the data and data1 results

# plot the scatter graph with the data and data1 results

# find some sort of conclusion about which is objectively the best value for money, and also memorize some rule of thumb that if the prices change you can still take a good estimate at what signs to look out for to determine if its overpriced or not
