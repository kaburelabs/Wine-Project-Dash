
# Data Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
#import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from plotly.offline import iplot

import dash
from dash import Input, Output, html, dcc, dash_table
import dash_bootstrap_components as dbc
import os 
import time


## CSS EXTERNAL FILE
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 
                        'https://codepen.io/chriddyp/pen/brPBPO.css']


## Defining the instance of dash
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets)

# server instance to run map when deploying
server = app.server

# Since I am adding callbacks to elements that don’t ~
# exist in the app.layout as they are spread throughout files
#app.config.suppress_callback_exceptions = True

sunb_point = pd.read_csv("data/sunburst/sunburst_point.csv", index_col=0)
sunb_price = pd.read_csv("data/sunburst/sunburst_price.csv", index_col=0)
sunb_count = pd.read_csv("data/sunburst/sunburst_count.csv", index_col=0)


def sunburst_view(data, value):

    fig= px.sunburst(data, 
                       path=['country', 'province', 'region_1','variety'], 
                       values=value, #height=700,
                       maxdepth=2)
    return fig


app.layout = html.Div(
    children=[
        html.Div(
            [   dcc.Dropdown(
                            id='dropdown',
                            options=[{'label': i, 'value': i} for i in ['count', 'price', 'points']],
                            value='points'
    ),
 
            ], className='Row'),
               html.Div([dcc.Graph(id='graph')])
                   
            ]
        )

@app.callback(Output("graph", "graph"),
             [Input("dropdown", "value")])
def _update_graph(value):
    if value == 'count':
            print('count active')
            return sunburst_view(sunb_count, 'price')
    elif value == 'price':
            print('price active')
            return sunburst_view(sunb_price, value)
    elif value == 'points':
            print('points active')
            return sunburst_view(sunb_point, value)

# @app.callback(Output("output-2", "children"), [Input("input-2", "value")])
# def input_triggers_nested(value):
#     time.sleep(1)
#     return value


if __name__ == "__main__":
    app.run_server(debug=True)