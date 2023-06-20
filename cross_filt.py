from textwrap import dedent as d
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import dash
from dash import Input, Output, html, dcc, dash_table
import pandas as pd
import plotly.express as px

import dash_bootstrap_components as dbc

import re

from operator import itemgetter


def remove_year(year):
    # years= year.str.extract(r'([0-9]{4})')
    expr = re.compile('[0-9]{4}')
    line = re.sub(expr, '', year)
    return line


def split_it(year):
    return year.str.extract(r'([0-9]{4})')

# external_stylesheets = ['https://codepen.io/kaburelabs/pen/xxGRXWa.css']

# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# def dataset():
#     data_path = 'data/wine_data_new.csv'
#     ## Importing the dataset
#     data = pd.read_csv(data_path, index_col=0)
#     print(data)

#     return data

# df_train = dataset()
# # print(df_train.columns)

# scatt = pd.read_csv("data/matrix_wines.csv", index_col=0)

# scatt['harvest'] = scatt['harvest'].astype(str)
# scatt['rating_cat'] = scatt['rating_cat'].astype(str)
# #scatt['clusters'] = scatt['clusters'].astype(str)
# df_train['rating_cat'] = df_train['rating_cat'].astype(str)

# print(scatt.columns)
# available_indicators = df['Indicator Name'].unique()


def _scatter_view(data, x_val, y_val, log):

    fig = px.scatter(data, x='price', y='points',
                     title=f"Distribution of Wine Price and Ratings",
                     color=x_val, hover_name='variety', opacity=.4,
                     hover_data=['country'], height=600,
                     log_x=log)

    fig.update_layout(showlegend=False, title_x=.5,
                      # title=f"Distribution of {x_val} <br>by {color}",
                      xaxis_title=f"Mean Price per Bottle (in USD)",
                      yaxis_title=f"Mean Points Range",
                      # xaxis={'type':'category'},
                      # margin=dict(t=100, l=50)
                      )
    return fig


def tf_idf_words(df, country_var, variety, ngram, province):

    if ngram == 1:
        ngran_range = (1, 2)
        title = (f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")
    elif ngram == 2:
        ngran_range = (2, 2)
        title = (f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")
    elif ngram == 3:
        ngran_range = (3, 3)
        title = (f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")

    vectorizer = TfidfVectorizer(ngram_range=ngran_range, min_df=1,
                                 stop_words='english', max_features=10,
                                 max_df=500)

    # X2 = vectorizer.fit_transform(df_train.loc[(df_train.country == country_var)]['description'])
    X2 = vectorizer.fit_transform(df['description_desc'])

    features = (vectorizer.get_feature_names())
    scores = (X2.toarray())

    # Getting top ranking features
    sums = X2.sum(axis=0)
    data1 = []

    for col, term in enumerate(features):
        data1.append((term, sums[0, col]))

    ranking = pd.DataFrame(data1, columns=['term', 'rank'])
    words = (ranking.sort_values('rank', ascending=False))[:15]

    fig = px.bar(words, x='term', y='rank', title=title, height=500)
    fig.update_layout(autosize=False, title_x=.5)
    return fig


columns = ['province', 'winery', 'title', 'harvest', 'points', 'price']

# print(scatt.columns)
layout2 = html.Div([
    dbc.Row([
            dbc.Col(
                html.Div([
                    html.Div("Colored by: ", className="text-color font-md"),
                    dcc.Dropdown(
                        id='crossfilter-xaxis-column',
                        options=[  # {'label': 'Review Cluster', 'value': 'clusters'},
                            {'label': 'Country', 'value': 'country'},
                            {'label': 'Harvest', 'value': 'harvest'},
                            {'label': 'Rating', 'value': 'rating_cat'}],
                        value='rating_cat', placeholder='Select a feature')
                ], className="text-center"
                ), width={"size": 8, "offset": 2}, md={"size": 5, "offset": 1}, lg={"size": 2, "offset": 0},
                className="bottom16"),
            dbc.Col(
                html.Div([
                    html.Div("Price Distribution",
                             className="text-color font-md"),
                    dcc.RadioItems(
                        id='crossfilter-yaxis-type',
                        options=[{'label': 'Linear', 'value': 0},
                                 {'label': 'Log', 'value': 1}],
                        value=1,
                        labelStyle={'display': 'inline-block',
                                    "marginRight": "16px"},
                    )], className="text-center"),
                width={"size": 8, "offset": 2}, md={"size": 5, "offset": 0}, lg={"size": 2, "offset": 0},
                className="bottom16"),
            dbc.Col(html.Div([html.Div("Selection Data", className="bold"),
                              html.Div(["To get informations about an specific pair of ", html.Span("Country/Variety", className="bold"), " click on the points in scatter chart below."
                                        # dcc.Markdown(d("""
                                        #     **Selection Data**

                                        #     To get informations about an specific pair of **Country/Variety**, click on the points in scatter chart below.
                                        # """)),
                                        ])]),
                    width={"size": 12, "offset": 0}, md={"size": 10, "offset": 1}, lg={"size": 8, "offset": 0},
                    className="bottom16"
                    ),
            ],  style={'borderLeft': 'thin lightgrey solid',
                       'backgroundColor': '#50d890',
                       'padding': '32px 16px', 'margin': '0 auto'}),

    # Gráficos
    dbc.Row([
        # Scatter plot
        dbc.Col(html.Div([
            # Gráfico principal que tem o hover
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                clickData={'points': [
                    {'customdata': 'Portugal', 'hovertext': 'Portuguese White'}]},
                # hoverData={'points': [{'customdata': 'General', 'hovertext':'Ramisco'}]

                # clear_on_unhover=True,
            )
        ],), width={"size": 12, "offset": 0}, md={"size": 10, "offset": 1}, lg={"size": 7, "offset": 0}, className="bottom32"),

        dbc.Col(
            html.Div([
                html.Div([

                    html.Div(
                        [
                            html.Div(
                                children=[
                                    html.Div(className='title text-center padding16 font-md',
                                             children="Information for Country/Variety"),
                                    html.Div(
                                        id='display-1', className="text-center bold font-lg")

                                ]),
                            dbc.Row([
                                dbc.Col(html.Div(
                                    children=[
                                        html.Div(
                                            id='display-2', className="text-center text-color bold font-lg"),
                                        html.Div(
                                            children="Reviews", className="text-center text-color font-md")
                                    ]), width={"size": 6, "offset": 3}, md={"size": 6, "offset": 3}),
                            ]),
                            dbc.Row([
                                    dbc.Col(html.Div(
                                        children=[
                                            html.Div(
                                                id='display-10', className="text-center text-color bold font-lg"),
                                            html.Div(
                                                children="Provinces", className="text-center text-color font-md")
                                        ]), width={"size": 4, "offset": 0}, md={"size": 4, "offset": 0}),
                                    dbc.Col(html.Div(
                                        children=[
                                            html.Div(
                                                id='display-3', className="text-center text-color bold font-lg"),
                                            html.Div(
                                                children="Wineries", className="text-center text-color font-md")
                                        ]), width={"size": 4, "offset": 0}, md={"size": 4, "offset": 0}),
                                    dbc.Col(html.Div(
                                        children=[
                                            html.Div(
                                                id='display-4', className="text-center text-color bold font-lg"),
                                            html.Div(
                                                children="Titles", className="text-center text-color font-md")
                                        ]), width={"size": 4, "offset": 0}, md={"size": 4, "offset": 0})], style={"padding": "0 12px"})

                        ], id="number-plate", className="BgDisplay",
                        style={'padding': '24px 0', 'height': 'auto'}),

                    html.Div([
                        html.Div(id='title-table'),
                        dash_table.DataTable(id='table',
                                             columns=[{"name": i, "id": i}
                                                      for i in columns],
                                             page_current=0,
                                             page_size=5,
                                             page_action='custom',
                                             filter_action='custom',
                                             sort_by=[],
                                             sort_mode='single',
                                             sort_action='custom',
                                             # sort_by='points', sort_mode='single',
                                             # sortDirection='ASC',
                                             filter_query='',
                                             # style_data={'whiteSpace':'normal'},
                                             # style_table={'overflowX': 'scroll'},
                                             fixed_columns={
                                                         'headers': True, 'data': 0},
                                             fixed_rows={
                                                 'headers': True, 'data': 5},
                                             # style_table={'height': '240px',
                                             #      #'minHeight': '150px',  'maxHeight': '150px',
                                             #      #'minWidth': '525px', 'width': '525px', 'maxWidth': '525px',
                                             # #     'overflowX': 'scroll'
                                             #  },
                                             style_cell={
                                                 #     'minWidth': '15px', 'maxWidth': '150px',
                                                 #     , 'maxHeight':'50px',
                                                 # #    'textOverflow': 'ellipsis',
                                             },
                                             style_cell_conditional=[
                                                 {
                                                     'if': {'column_id': c},
                                                     'maxWidth': '90px',
                                                     'textAlign': 'left',
                                                     #    'overflowY': 'hidden',
                                                     'textOverflow': 'hidden'
                                                 } for c in ['province', 'winery', 'title']
                                             ],
                                             )], style={'margin': '0 auto'})
                ], style={"height": "600px"}),
            ]), width={"size": 12, "offset": 0}, md={"size": 10, "offset": 1}, lg={"size": 5, "offset": 0})
    ], style={'padding': '24px 0'}),

    dbc.Row(
        html.Div([
            html.Div("Reviews Specification",
                     className="font-xl text-center bottom16"),
            html.Div(["You can select in the Scatter plot on the right the Titles and see some aspects of the reviews of sommeliers."],
                     className="font-md text-center bottom32"),

            html.Div("This section is composed by three parts: ",
                     className="font-lg text-center bottom16 top16"),
            html.Div(dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Div("TFIDF", className="font-lg"),
                        html.Div(
                            "TFIDF - You can select the N-gram that you desire for each Country/Variety pair.", className="font-md bottom16")
                    ], style={"height": "100%"}, className="BgDisplay2 radius12 padding16"), width=4),
                dbc.Col(html.Div([
                    html.Div("Scatter Plot", className="font-lg"),
                    html.Div(
                        "Scatter Plot Chart where you can select an specific variety to see some informations and main aspects of sommelier reviews.", className="font-md bottom16")
                ], style={"height": "100%"}, className="BgDisplay2 radius12 padding16"), width=4),
                dbc.Col(html.Div([
                    html.Div("Aspects - SpaCy", className="font-lg"),
                    html.Div(
                        "The Aspects of reviews was extracted by SpaCy using some POS tagging techniques", className="font-md bottom16")
                ], style={"height": "100%"}, className="BgDisplay2 radius12 padding16"), width=4),
            ], className="boxed margin-auto")),
        ], className="BgDisplay padding32 bottom32"),
    ),
    dbc.Row([
        dbc.Col(html.Div([
            html.Div("TfIdf N-grams: ", className="font-md text-color"),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': 'Unigram', 'value': 1},
                         {'label': 'Bigram', 'value': 2},
                         {'label': 'Trigram', 'value': 3}],
                value=1,
                labelStyle={'display': 'inline-block', "marginLeft": "16px"})
        ], className="text-center"), width=3, className="bottom32"),
        dbc.Col(html.Div([
            html.Div("Select the Province(s): ",
                     className="font-md text-color"),
            dcc.Dropdown(
                id='province-selector1',
                multi=False,  # labelStyle={'display': 'inline-block'}
            )
        ], className="text-center"), width=3, className="bottom32")
    ], style={'borderBottom': 'thin lightgrey solid',
              'backgroundColor': '#50d890',
              'padding': '30px 0 0'}, className="bottom32"),

    # Table and tfidf plot
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='fig-1'), width=7),    # style={'width':'75%', 'margin':'0 auto'},
        dbc.Col(
            dcc.Graph(id='fig-2',
                      clickData={'points': [
                          {'customdata': 'Portugal', 'hovertext': 'Portuguese White'}]},
                      hoverData={'points': [{'customdata': ['Seacampo  Reserva Red (Dão)', 'Portugal']}]},),
            width=5)

    ]),

    html.Div([
        html.Div(id='inf-1')
        # style={'width':'75%', 'margin':'0 auto'},
        # html.Div([
        #     dcc.Graph(id='fig-2')
        #     ], className='four columns')

    ], style={'padding': '24px 0'}, className="BgDisplay")

])
