

# Data Libraries
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Visualization Libraries
#import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from plotly.offline import iplot

import dash
import dash_table
from dash import Input, Output, html, dcc
import dash_bootstrap_components as dbc
import os 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#import sklearn 


data_path = 'data/winemag-data-130k-v2.csv'
## Importing the dataset
# df_train = pd.read_csv(data_path, index_col=0)

# pandarallel.iniatilize()

target_path = "assets/profiling1.html"


if os.path.exists(target_path): 
    pass
else: 
    profile = ProfileReport(df_train, title='Pandas Profiling Report')
    #profile = df_train.parallel_apply(profiling) #html={'style':{'full_width':True}}
    profile.to_file(output_file="assets/profiling1.html")

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
# app.config.suppress_callback_exceptions = True

sunb_point = pd.read_csv("data/sunburst/sunburst_point.csv", index_col=0)
sunb_price = pd.read_csv("data/sunburst/sunburst_price.csv", index_col=0)
sunb_count = pd.read_csv("data/sunburst/sunburst_count.csv", index_col=0)
scatt = pd.read_csv("data/matrix_wines.csv", index_col=0)


def sunburst_view(data, value):

    fig= px.sunburst(data, 
                    path=['country', 'province', 'region_1','variety'], 
                    values=value, height=450, width=450,
                    maxdepth=2, 
                    #color_continuous_scale='RdBu',
                    branchvalues='total')
                    
    return fig

def _scatter_view(data, color='country'):

    fig = px.scatter(data, x='price_log', y='points',
                     color=color, hover_name='variety', opacity=.3,
                     hover_data=['country', 'price'], 
                     height=450, width=450)

    fig.update_layout(showlegend=False)

    return fig
from textwrap import dedent as d
app.layout = html.Div(
    children=[
        html.Div(
            [   dcc.Dropdown(
                            id='dropdown',
                            options=[{'label': i, 'value': i} for i in ['count', 'price', 'points']],
                            value='count'
    ),          dcc.Markdown(d("""
                                **Hover Data**

                                Mouse over values in the graph.
                            """)),
                html.Pre(id='hover-data'),
            ], className='Row'),
        html.Div([dcc.Graph(id='graph', className='six columns', styele={'border':'dotted'}),
                  dcc.Graph(id='graph2', figure=_scatter_view(scatt), className='six columns')
                  ], className='row', style={'margin':'0 auto', 'width':'80%', 
                                             'margin':'24x', 'display':'inline-block'})      
                ], className='ten columns offset-by-one' 
        )


@app.callback(Output("graph", "figure"),
              #Output("hover-data", 'hoverData')
             [Input("dropdown", "value")])
def _update_graph(value):
    if value == 'count':
            fig =  sunburst_view(sunb_count, 'price')

    elif value == 'price':
            fig =  sunburst_view(sunb_price, value)

    elif value == 'points':
            fig = sunburst_view(sunb_point, value)

    # return the figure 
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)