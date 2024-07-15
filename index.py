import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('RumahYogya.csv')

# Remove rows with NaN values
df = df.dropna()

# Define models
def train_model(features):
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    return regr

model1 = train_model(['surface_area', 'building_area', 'bed', 'bath'])
model2 = train_model(['surface_area', 'building_area', 'bath'])
model3 = train_model(['building_area', 'bath'])

# Initialize the Dash app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

# Add the external CSS file
app.css.append_css({'external_url': '/assets/styles.css'})

# Define the layout of the dashboard 
app.layout = html.Div(children=[
    html.H1(children='Real Estate Dashboard'),

    dcc.Tabs([
        dcc.Tab(label='Visualizations', children=[
            dcc.Graph(id='bed-price-scatter'),
            dcc.Graph(id='bath-price-scatter'),
            dcc.Graph(id='surface-area-price-scatter'),
            dcc.Graph(id='building-area-price-scatter'),
            dcc.Graph(id='correlation-matrix')
        ]),
        dcc.Tab(label='Predict House Price', children=[
            html.Label('Select Model:'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'Model 1 (4 Variables)', 'value': 'model1'},
                    {'label': 'Model 2 (3 Variables)', 'value': 'model2'},
                    {'label': 'Model 3 (2 Variables)', 'value': 'model3'}
                ],
                value='model1'
            ),
            html.Div(id='input-features'),
            html.Button('Predict', id='predict-button'),
            html.Div(id='prediction-output')
        ])
    ])
])

@app.callback(
    Output('bed-price-scatter', 'figure'),
    Input('model-dropdown', 'value')
)
def update_bed_price_scatter(selected_model):
    fig = px.scatter(df, x='bed', y='price', title='Bedrooms vs Price', template='plotly')
    return fig

@app.callback(
    Output('bath-price-scatter', 'figure'),
    Input('model-dropdown', 'value')
)
def update_bath_price_scatter(selected_model):
    fig = px.scatter(df, x='bath', y='price', title='Bathrooms vs Price', template='plotly')
    return fig

@app.callback(
    Output('surface-area-price-scatter', 'figure'),
    Input('model-dropdown', 'value')
)
def update_surface_area_price_scatter(selected_model):
    fig = px.scatter(df, x='surface_area', y='price', title='Surface Area vs Price', template='plotly')
    return fig

@app.callback(
    Output('building-area-price-scatter', 'figure'),
    Input('model-dropdown', 'value')
)
def update_building_area_price_scatter(selected_model):
    fig = px.scatter(df, x='building_area', y='price', title='Building Area vs Price', template='plotly')
    return fig

@app.callback(
    Output('correlation-matrix', 'figure'),
    Input('model-dropdown', 'value')
)
def update_correlation_matrix(selected_model):
    corr = df.corr()
    fig = px.imshow(corr, title='Correlation Matrix', template='plotly')
    return fig

@app.callback(
    Output('input-features', 'children'),
    Input('model-dropdown', 'value')
)
def update_input_features(selected_model):
    if selected_model == 'model1':
        return html.Div([
            html.Label('Surface Area (m²)'),
            dcc.Input(id={'type': 'input-feature', 'index': 'surface_area'}, type='number', value=0),
            html.Label('Building Area (m²)'),
            dcc.Input(id={'type': 'input-feature', 'index': 'building_area'}, type='number', value=0),
            html.Label('Bedrooms'),
            dcc.Input(id={'type': 'input-feature', 'index': 'bed'}, type='number', value=0),
            html.Label('Bathrooms'),
            dcc.Input(id={'type': 'input-feature', 'index': 'bath'}, type='number', value=0),
        ])
    elif selected_model == 'model2':
        return html.Div([
            html.Label('Surface Area (m²)'),
            dcc.Input(id={'type': 'input-feature', 'index': 'surface_area'}, type='number', value=0),
            html.Label('Building Area (m²)'),
            dcc.Input(id={'type': 'input-feature', 'index': 'building_area'}, type='number', value=0),
            html.Label('Bathrooms'),
            dcc.Input(id={'type': 'input-feature', 'index': 'bath'}, type='number', value=0),
        ])
    elif selected_model == 'model3':
        return html.Div([
            html.Label('Building Area (m²)'),
            dcc.Input(id={'type': 'input-feature', 'index': 'building_area'}, type='number', value=0),
            html.Label('Bathrooms'),
            dcc.Input(id={'type': 'input-feature', 'index': 'bath'}, type='number', value=0),
        ])

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State('model-dropdown', 'value'),
     State({'type': 'input-feature', 'index': ALL}, 'value')]
)
def predict_price(n_clicks, selected_model, input_values):
    if n_clicks is None:
        return ''

    input_data = {}
    if selected_model == 'model1':
        input_data = {
            'surface_area': input_values[0],
            'building_area': input_values[1],
            'bed': input_values[2],
            'bath': input_values[3]
        }
        model = model1
    elif selected_model == 'model2':
        input_data = {
            'surface_area': input_values[0],
            'building_area': input_values[1],
            'bath': input_values[2]
        }
        model = model2
    elif selected_model == 'model3':
        input_data = {
            'building_area': input_values[0],
            'bath': input_values[1]
        }
        model = model3

    input_df = pd.DataFrame([input_data])
    predicted_price = model.predict(input_df)[0]
    return f'Predicted Price: {predicted_price}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
