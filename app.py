
# Data Libraries
import pickle
import random
import sklearn
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Visualization Libraries
# import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from plotly.offline import iplot

import dash
import dash_table
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD

import cross_filt
from cross_filt import *


def dataset():
    data_path = 'data/wine_data_new.csv'
    # Importing the dataset
    data = pd.read_csv(data_path, index_col=0)

    return data


df_train = dataset()

scatt = pd.read_csv("data/matrix_wines.csv", index_col=0)

scatt['harvest'] = scatt['harvest'].astype(str)
scatt['rating_cat'] = scatt['rating_cat'].astype(str)
# scatt['clusters'] = scatt['clusters'].astype(str)
df_train['rating_cat'] = df_train['rating_cat'].astype(str)

target_path = "assets/profiling1.html"


if os.path.exists(target_path):
    pass
else:
    profile = ProfileReport(df_train, title='Pandas Profiling Report')
    # profile = df_train.parallel_apply(profiling) #html={'style':{'full_width':True}}
    profile.to_file(output_file="assets/profiling1.html")

# CSS EXTERNAL FILE
external_stylesheets = [dbc.themes.GRID,
                        'https://codepen.io/kaburelabs/pen/xxGRXWa.css',
                        "https://codepen.io/chriddyp/pen/brPBPO.css",
                        'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
                        'https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css',
                        "https://fonts.googleapis.com/css2?family=Squada+One&display=swap"]

meta_tags_arg = {"name": "viewport",
                 "content": "width=device-width, initial-scale=1, maximum-scale=1"}

# App Name
app_name = 'trich.ai Dashboard'

# Defining the instance of dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                meta_tags=[meta_tags_arg])

app.title = app_name

# server instance to run map when deploying
server = app.server

# Since I am adding callbacks to elements that don’t ~
# exist in the app.layout as they are spread throughout files
app.config.suppress_callback_exceptions = True


stopwords = set(STOPWORDS)

newStopWords = ['fruit', "Drink", "black", 'wine', 'drink']


def navbar(logo="/assets/logo-placeholder.png", height="35px",  appname="PlaceHolder Name"):

    navbar = html.Div(
        [dbc.Container(
            [dbc.Row([
                dbc.Col(html.A(
                    # Use row and col to control vertical alignment of logo / brand
                        html.Div(
                            "trich.ai", className="trich-navbar white font-xl", ),
                        href="https://trich.ai", target="_blank"
                        ), width=4),
                dbc.Col(dbc.NavbarBrand(
                    appname, className="font-lg text-white"), className="text-right", width=8)],
                style={"align-items": "center", "min-height": "75px"}
            )
            ],
            style={"maxWidth": "1140px"})
         ],
        className="bottom16 navbarColor",
        # style={'height': '100px', "borderBottom":".5px solid lightgrey", "padding":"18px 0px"}

        # dark=True,
    )

    return navbar


def create_header(some_string):
    header_style = {
        'background-color': 'rgb(79, 152, 202)',
        'padding': '1.5rem',
        'display': 'inline-block',
        'width': '100%'

        # 'border-style': 'dotted'
    }
    logo_trich = html.Img(
        src='/assets/fundo_transp-b.png',
        className='three columns',
        style={
            'height': 'auto',
            'width': '140px',  # 'padding': 1
            'float': 'right',  # 'position': 'relative'
            'margin-right': '66px',  # 'border-style': 'dotted'
            'display': 'inline-block'})

    title = html.H1(children=some_string, className='eight columns',
                    style={'margin': '0 0 0 36px',
                           'color': '#ffffff', 'font-size': '35px'})

    header = html.Header(html.Div([title, logo_trich]), style=header_style)

    return header


def create_footer():
    p = html.Div(
        children=[
            html.Span('Developed By: '),
            html.A('trich.ai | Data Intelligence Solutions',
                   style={'text-decoration': 'none', 'color': '#ffffff'},
                   href='https://trich.ai', target='_blank')
        ], style={'float': 'right', 'margin-top': '8px',
                  'font-size': '18px', 'color': '#ffffff'}
    )

    span_style = {'horizontal-align': 'right',
                  'padding-left': '1rem',
                  'font-size': '15px',
                  'vertical-align': 'middle'}

    kaggle = html.A(
        children=[
            html.I([], className='fab fa-kaggle'),
            html.Span('Kaggle', style=span_style)
        ], style={'text-decoration': 'none', 'color': '#ffffff', 'margin-right': '20px'},
        href="https://www.kaggle.com/kabure/kernels",
        target='_blank')

    mapbox = html.A(
        children=[
            html.I([], className='fab fa-python'),
            html.Span('Dash Plotly', style=span_style)
        ], style={'text-decoration': 'none', 'color': '#ffffff', 'margin-right': '20px'},
        href='https://plot.ly/dash/', target='_blank')

    font_awesome = html.A(
        children=[
            html.I([], className='fa fa-font-awesome'),
            html.Span('Font Awesome', style=span_style)
        ], style={'text-decoration': 'none', 'color': '#ffffff', 'margin-right': '20px'},
        href='http://fontawesome.io/', target='_blank')

    datatables = html.A(
        children=[
            html.I([], className='fa fa-github'),
            html.Span('trich.ai\n Github', style=span_style)
        ], style={'text-decoration': 'none', 'color': '#ffffff', 'margin-right': '20px'},
        href='https://github.com/kaburelabs/', target='_blank')

    ul1 = html.Div(
        children=[
            html.Li(mapbox, style={
                    'display': 'inline-block', 'color': '#ffffff'}),
            html.Li(font_awesome, style={
                    'display': 'inline-block', 'color': '#ffffff'}),
            html.Li(datatables, style={
                    'display': 'inline-block', 'color': '#ffffff'}),
            html.Li(kaggle, style={
                    'display': 'inline-block', 'color': '#ffffff'}),
        ],
        style={'list-style-type': 'none', 'font-size': '30px'},
    )

    hashtags = 'plotly,dash,trich.ai,wine,nlp'
    tweet = 'trich.ai Wine Reviews WebApp, a cool dashboard with Plotly Dash!'
    twitter_href = 'https://twitter.com/intent/tweet?hashtags={}&text={}'\
        .format(hashtags, tweet)
    twitter = html.A(
        children=html.I(children=[], className='fa fa-twitter',
                        style={"color": "rgb(29, 161, 242)"}),
        title='Tweet me!', href=twitter_href, target='_blank')

    github = html.A(
        children=html.I(children=[], className='fa fa-github',
                        style={'color': 'black'}),
        title='Repo on GitHub',
        href='https://github.com/kaburelabs/Wine-Project-Dash', target='_blank')

    li_right_first = {'line-style-type': 'none', 'display': 'inline-block'}
    li_right_others = {k: v for k, v in li_right_first.items()}
    li_right_others.update({'margin-left': '10px'})
    ul2 = html.Ul(
        children=[
            html.Li(twitter, style=li_right_first),
            html.Li(github, style=li_right_others),
        ],
        style={
            'position': 'fixed',
            'right': '1.5rem',
            'bottom': '75px',
            'font-size': '60px'
        }
    )
    div = html.Div([p, ul1, ul2])

    footer_style = {
        'font-size': '2.2rem',
        'background-color': 'rgb(79, 152, 202)',
        # 'padding': '2.5rem',
        'margin-top': '5rem',
        'display': 'inline-block', 'padding': '16px 32px 8px'
    }
    footer = html.Footer(div, style=footer_style, className='twelve columns')
    return footer


model = pickle.load(open("modelKnn.pkl", 'rb'))
matrix = pickle.load(open("data.h5", 'rb'))


app.layout = html.Div([
    navbar(appname='Dashboard'),
    dbc.Container([
        dcc.Tabs(id="tabs-example",
                 children=[
                    dcc.Tab(label='Data & Details', value='tab-1'),
                    dcc.Tab(label='Profilling', value='tab-2'),
                    dcc.Tab(label='Graphs & EDA', value='tab-3'),
                    dcc.Tab(label='Reviews Clustering', value='tab-4'),
                    dcc.Tab(label='Recommender System', value='tab-5')
                 ],
                 value='tab-1', className="bottom32"),
        html.Div(id='tabs-content-example')
    ], style={"maxWidth": "1140px"}),
    create_footer()
], style={'margin': '0 auto', 'overflow': 'hidden'})


@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value'),
               Input('tabs-example', 'value')])
def render_content(tab, val_clusters):
    if tab == 'tab-1':
        return html.Div([
            html.Div([
                        main_page()
                        ])
            #         html.Div([
            #             dcc.Upload([
            #                 'Drag and Drop or ',
            #                 html.A('Select a File')
            #             ], style={
            #                 'width': '100%',
            #                 'height': '60px',
            #                 'lineHeight': '60px',
            #                 'borderWidth': '1px',
            #                 'borderStyle': 'dashed',
            #                 'borderRadius': '5px',
            #                 'textAlign': 'center'
            #         })
            # ], className='row', style={'height':'70vh'})
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([
                    html.Div(
                        'Data Profiling', className="font-xl text-color paddingtop32 bottom16"),
                    html.Div(["This report was developed totally using ",
                              html.A(
                                  'Pandas Profiling', href='https://pandas-profiling.github.io/pandas-profiling/docs/', target='_blank'),
                              " it's a Open Source tool that create an insightful and well structured menu of informations about our data."], className="font-md text-color bottom32"),
                    html.Div([
                        html.Div(["For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:"],
                                 className="text-color font-md bold bottom16"),
                        html.Ul(
                            children=[
                                html.Li([html.Span('Type Inference: ', style={
                                    'font-weight': 'bold'}), "detect the types of columns in a dataframe."], className="text-color"),
                                html.Li([html.Span('Essentials: ', style={
                                    'font-weight': 'bold'}), "type, unique values, missing values."], className="text-color"),
                                html.Li([html.Span('Quantile Statistics ', style={
                                    'font-weight': 'bold'}), "like minimum value, Q1, median, Q3, maximum, range, interquartile range"], className="text-color"),
                                html.Li([html.Span('Descriptive statistics: ', style={
                                    'font-weight': 'bold'}), "like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness"], className="text-color"),
                                html.Li([html.Span("Most frequent values ", style={
                                    'font-weight': 'bold'})], className="text-color"),
                                html.Li([html.Span('Histogram ', style={
                                    'font-weight': 'bold', })], className="text-color"),
                                html.Li([html.Span('Correlations ', style={
                                    'font-weight': 'bold', }), "highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices"], className="text-color"),
                                html.Li([html.Span('Missing values ', style={
                                    'font-weight': 'bold', }), "matrix, count, heatmap and dendrogram of missing values"], className="text-color"),
                                html.Li([html.Span('Text analysis ', style={
                                    'font-weight': 'bold', }), "learn about categories (Uppercase, Space), scripts (Latin, Cyrillic) and blocks (ASCII) of text data."], className="text-color"),
                                # html.Ul([html.Li(x) for x in my_list])
                            ],
                            style={'list-style-type': 'none'}, className="font-md")
                    ])], className="bottom32"),
                html.Iframe(
                    id='graph-1-tabs',
                    src=app.get_asset_url('profiling1.html'),
                    # src=app.get_asset_url('trich-dash.jpg'),
                    # className='ten columns offset-by-one',
                    style={'margin': '0 auto',
                           'width': '100%',
                           'height': '650px'}
                )
            ], className='boxed')], className="BgDisplay")

    elif tab == 'tab-3':
        return layout2

    elif tab == 'tab-4':
        return html.Div([
            html.Div([
                html.Div([
                        html.Div("K-Means Clustering",
                                 className="font-lg paddingtop32 bottom16"),
                        html.Div("The problem of K means can be thought of as grouping the data into K clusters where assignment to the clusters is based on some similarity or distance measure to a centroid.", className="font-md bottom16"),
                        html.Div("I have used tf-idf with n_gram(1, 3) with max of 600 and the values of Elbow Method are below to 2 until 25 clusters and you can explore the counts, the words and also the countries for each of this clusterizations.", className="font-md")
                        ], className="boxed paddingbottom32")
            ], className="BgDisplay bottom32"),

            dbc.Row([
                dbc.Col([html.Div(id='tab-4-display-c', className="text-center bottom16"),
                         # button line
                         dcc.Slider(id='dropdown-tab4',
                                    min=2,
                                    max=24,
                                    value=6,
                                    marks={
                                       2: {'label': '2', 'style': {'color': '#272727'}},
                                       5: {'label': '5', 'style': {'color': '#272727'}},
                                       10: {'label': '10', 'style': {'color': '#272727'}},
                                       15: {'label': '15', 'style': {'color': '#f50'}},
                                       20: {'label': '20', 'style': {'color': '#f50'}},
                                       24: {'label': '24', 'style': {'color': '#f50'}}
                                    },
                                    included=False,
                                    )], width=4),

            ], className="bottom32", style={"background": "rgb(80, 216, 144)", 'padding': '25px 50px'}),


            html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(
                        # className='five columns'
                        id='graph-4-tabs', style={'margin': '0 auto'},
                    ), width=5),
                    dbc.Col(dcc.Graph(id='table-tab4-2',  # className='seven columns',
                                      style={'height': '450px'}), width=7)

                ], ),
                html.Div([
                    html.Div("Cluster Analysis",
                             className="font-xl text-center"),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Div("Cluster Analysis", className="font-lg"),
                            html.Div(["The model has a cluster center that returns the coordinates of the k cluster centroids. Each token in the created vectorizer has a dimension or doordinate in the centroid and represents it's relative frequency within that cluster. In the table beside we have the top ten most important words for each cluster."], className="font-md")
                        ], style={'display': 'inline-block'}), width=3),
                        dbc.Col(html.Div(
                            id='table-tab4', style={'height': '310px', 'display': 'inline-block'}), width=9)]),
                ]),
                html.Div([
                    html.Div([
                        html.Div("Crosstab Clusters x Country",
                                 className="font-lg text-center")
                    ]),
                    html.Div([
                        dcc.Graph(id='graph-4-tabs-2',
                                  style={'height': '500px', 'margin': '0 auto'})
                    ])
                ], style={'margin-top': '25px', 'padding': '25px'})

            ], style={'margin': '0 auto'})
        ])

    elif tab == 'tab-5':
        return html.Div([
            html.Div([
                html.Div(
                    [
                        html.Div("Collabrative filtering - Nearest Neighbour",
                                 className="font-lg paddingtop32 bottom16"),
                        html.Div("A simple but very useful and acurate recommender system is made using Nearest Neighbors algorithm.",
                                 className="font-md bottom32"),
                        html.Div("It's made based on two main points: ",
                                 className="font-md bottom16"),
                        html.Ul([html.Li("Similarity is the cosine of the angle between the 2 vectors of the item vectors of A and B.",
                                         className="font-md"),
                                 html.Li("Closer the vectors, smaller will be the angle and larger the cosine",
                                         className="font-md")
                                 ], className="bottom32"),

                        html.Div("To get the recommendations",
                                 className="font-lg"),
                        html.Div("Choose a variety that you want to get similar ones.",
                                 className="font-md paddingbottom32")
                    ], className="boxed")
            ], className="BgDisplay bottom32"),

            html.Div([
                dbc.Row([
                    dbc.Col(
                        [html.Div("Select the variety to get recommendation", className="font-md"),
                            dcc.Dropdown(id='tab-5-varieties',
                                         options=[
                                             {'label': i, 'value': n} for n, i in enumerate(matrix.index)],
                                         placeholder='(Cabernet, Merlot, Chardonnay, Red Blend...)',
                                         style={'zIndex': '999',
                                                'position': 'relative'})
                         ], width={"size": 5, "offset": 2}),

                    dbc.Col(html.Button('Get Recommendation', id='input-box', className="text-center", style={"width": "100%"}),
                            width={"size": 3, "offset": 0}, style={"paddingTop": "24px"}),
                ]),
                dbc.Col(
                    html.Div([
                        html.Div(id='recommender-title',
                                 className="font-lg padding16"),
                        html.Div(dash_table.DataTable(id='recommender-table',
                                                      columns=[{'name': i, 'id': i} for i in [
                                                          'K', 'variety', 'Distance']],
                                                      page_size=5, fixed_columns={'headers': True, 'data': 0},
                                                      fixed_rows={
                                                          'headers': True, 'data': 5},
                                                      style_cell_conditional=[
                                                          # {
                                                          #     'if': {'column_id': c},
                                                          #     'maxWidth': '90px',
                                                          #     'textAlign':'left',
                                                          # #    'overflowY': 'hidden',
                                                          #     'textOverflow':'hidden'
                                                          # } for c in ['province', 'winery', 'title']
                                                          {'if': {
                                                              'column_id': 'variety'}, 'width': '300px'},
                                                          {'if': {'column_id': 'K'},
                                                           'width': '20px'},
                                                          {'if': {
                                                              'column_id': 'Distance'}, 'width': '80px'},
                                                      ],

                                                      ))
                    ],  # style={'allign': 'left', 'zIndex': 15}
                    ),
                    width={"size": 8, "offset": 2})
            ], style={'height': '72vh'})
        ])


def main_page():
    return html.Div([
        html.Div([
            html.Div(html.Div("Data Specifications",
                              className="font-xl bottom16 paddingtop32 text-color")),
            html.Div(html.Div(["The dataset was firstly get in ",
                               html.A(
                                   'Wine Reviews Dataset', href='https://www.kaggle.com/zynicide/wine-reviews', target='_blank'),
                               " where I developed a ", html.A(
                                   'Kernel', href='https://www.kaggle.com/kabure/wine-review-s-eda-recommend-systems', target='_blank'),
                               " and learned a lot about wines, so I decided to create a more robust solution to both learn more about wines and also, to train my NLP and Web Development skills."
                               ], className="font-md  text-color")),
            html.Br(),
            html.Div(
                html.Div("Selections", className="font-lg bottom16 text-color")),
            html.Div(html.Div("The webapp was done using only titles from the harvests of 2004 to 2015.",
                              className="font-md  text-color")),
            html.Div(html.Div("This selection give us the total of: 117350 total rows, 43 countries, 406 provinces, 674 varieties\
                from 15595 wineries that has 73194 unique titles. (You can access all\
                     this informations in Data Profiling Section.", className="font-md  text-color")),
            html.Br()], className="boxed"),

        dbc.Row([
            html.Div("Principal techniques applied to extract informations and insights: ",
                     className="font-lg margin-auto bottom16 text-color"),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Div("Data Profiling:",
                             className="font-md bold  text-color"),
                    html.Div("Data profiling is the process of examining the data available from an \
                            existing information source and collecting statistics or informative summaries about that data.",
                             className="font-md text-color")
                ], style={"height": "100%"}, className="radius12 padding16 BgDisplay2"),
                    width={"size": 12, "offset": 0}, md={"size": 6, "offset": 3}, lg={"size": 4, "offset": 0},
                    className="bottom32"
                ),
                dbc.Col(html.Div([
                    html.Div(
                        "TF-IDF:", className="font-md bold text-color"),
                    html.Div("TF- IDF stands for Term Frequency and Inverse Document Frequency .\
                             TF-IDF helps in evaluating importance of a word in a document. In order to ascertain \
                                 how frequent the term/word appears in the document and also to represent the \
                                     document in vector form, let's break it down to following steps.",
                             className="font-md text-color")
                ], style={"height": "100%"}, className="radius12 padding16 BgDisplay2"),
                    width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 4, "offset": 0},
                    className="bottom32"
                ),
                dbc.Col(html.Div([
                    html.Div("Usupervised Techniques:",
                             className="font-md bold text-color"),
                    html.Div("We are using two different unsupervised learning algorithmn to help detect patterns in the \
                            reviews(K-means) and in the recommender system (K-NN). I'm also using (nltk.Vader) to extract and \
                                detect subjective informations and Sentiment and affective states (sentiment analysis) \
                                    and PCA (dimension reduction) techniques", className="font-md text-color")
                ], style={"height": "100%"}, className="radius12 padding16 BgDisplay2"),
                    width={"size": 12, "offset": 0}, md={"size": 6, "offset": 0}, lg={"size": 4, "offset": 0},
                    className="bottom32"
                ),
            ])
        ], className="boxed"),
    ], className="BgDisplay")

    # I have worked a lot in data cleaning, exploration, to get insights from the features to build this dashboarddatetime A combination of a date and a time. Attributes: ()
    # \

    # \
    # Also, I have done an [Kernel on Kaggle](https://www.kaggle.com/kabure/wine-review-s-eda-recommend-systems), where you can visit and get another type of insights and also, understand the behind parts of this project;


@ app.callback(
    [dash.dependencies.Output('recommender-table', 'data'),
     dash.dependencies.Output('recommender-title', 'children')],
    [dash.dependencies.Input('input-box', 'n_clicks')],
    [dash.dependencies.State('tab-5-varieties', 'value')])
def update_output(n_clicks, variety):
    # print(matrix.iloc[value,:].values)
    print('n_clicks', n_clicks, 'varieties', variety)
    if variety == None:
        variety = 5
    distance, indice = model.kneighbors(
        matrix.iloc[variety, :].values.reshape(1, -1), n_neighbors=6)
    # print(distance, indice)
    cols = ['K', 'variety', 'Distance']
    vals = []

    tmp = pd.DataFrame()
    i_n = 1
    for i in range(0, len(distance.flatten())):
        if matrix.index[indice.flatten()[i]] == matrix.iloc[variety].name:
            pass
        else:
            vals.append(
                (i_n, matrix.index[indice.flatten()[i]], distance.flatten()[i]))
            i_n = i_n+1

    df = pd.DataFrame(vals, columns=cols)

    return [df.to_dict('rows'), f'Recommender for: {matrix.index[variety]}']


operators = [['ge ', '>='], ['le ', '<='], ['lt ', '<'],
             ['gt ', '>'], ['ne ', '!='], ['eq ', '='],
             ['contains '], ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                # word operators need spaces after them in the filter string,
                        # but we don't want these later
                        return name, operator_type[0].strip(), value
    return [None] * 3


@ app.callback(Output('table-sorting-filtering', 'data'),
               [Input('table-sorting-filtering', "page_current"),
                Input('table-sorting-filtering', "page_size"),
                Input('table-sorting-filtering', 'sort_by'),
                Input('table-sorting-filtering', 'filter_query')])
def update_table(page_current, page_size, sort_by, filter):
    filtering_expressions = filter.split(' && ')
    dff = df_train.copy()
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)
        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
        if len(sort_by):
            dff = dff.sort_values(
                [col['column_id'] for col in sort_by],
                ascending=[
                    col['direction'] == 'asc'
                    for col in sort_by
                ],
                inplace=False
            )
    page = page_current

    size = page_size

    return dff.iloc[page * size: (page + 1) * size].to_dict('records')


kmeans_elbow = pd.read_csv('data/elbow_method.csv', index_col=0)


@ app.callback(Output('graph-4-tabs', 'figure'),
               [Input('dropdown-tab4', "value")])
def tf_idf_words(country_var):
    tmp = kmeans_elbow[kmeans_elbow.K_number == country_var]
    # tmp['mark'] = tmp[]
    # kmeans_elbow['selected'] = kmeans_elbow['selected'].astype(str)

    fig2 = px.scatter(tmp, x="K_number", y="Distorcions", )

    fig = px.scatter(kmeans_elbow, x="K_number",
                     y="Distorcions",
                     title="Elbow Method Runs").update_traces(mode='lines+markers',
                                                              marker=dict(size=np.where((kmeans_elbow.K_number == country_var), 15, 8),
                                                                          color=(kmeans_elbow.K_number == country_var).astype(
                                                                  int),
                                                                  colorscale=[
                                                                  [0, '#636efa'], [1, 'red']],
                                                                  symbol=np.where((kmeans_elbow.K_number == country_var), 'star', 'circle'))
                                                              # symbol=((kmeans_elbow.K_number == country_var).astype(int), 'circle', 'x-thin'))
                                                              )

    fig.update_layout(title_x=.5, )
    # fig2.update_layout(markers=dict(color='red'))
    # fig.add_trace(fig2.data[0])
    # fig['data'][0]['marker']['color']
    fig['data'][0]['marker']['color']
    print(fig['data'][0])
    return fig


@ app.callback(Output('graph-4-tabs-2', 'figure'),
               [Input('dropdown-tab4', "value")])
def tf_idf_words(country_var):

    tmp = pd.read_csv(
        f"data/clusters/crosstab_country{country_var}.csv", index_col=0)

    fig = px.bar(tmp, x="clusters",
                 y="total", color='country',
                 title="Percent of Countries in each Cluster")

    fig.update_layout(title_x=.5, height=500,
                      xaxis=dict(type='category',
                                 title='Cluster Number',
                                 categoryarray=[x for x in sorted(tmp['clusters'])]),
                      yaxis=dict(title='Percent of Total'))

    return fig


@ app.callback(Output('tab-4-display-c', 'children'),
               [Input('dropdown-tab4', "value")])
def tf_idf_words(country_var):

    # fig = px.line(kmeans_elbow, x="K_number",
    #                             y="Distorcions",
    #                             title="Elbow Method Runs")

    return html.Div([f'Number of Selected Clusters: {country_var}'], style={'font-weight': 'bold', 'fontSize': '16px'})


def generate_html_table(max_rows=10, n_clusters=4):

    df = pd.read_csv(f"data/clusters/clusters{n_clusters}.csv", index_col=0)

    return html.Div(
        html.Table(
            # Header
            [html.Tr([html.Th()])]
            +
            # Body
            [
                html.Tr(
                    [
                        html.Td(
                            dash_table.DataTable(
                                columns=[{"name": i, "id": i}
                                         for i in df.columns],
                                data=df.to_dict('rows')
                            ), style={'width': '945px'}

                        )

                    ]
                )
                # for i in range(min(len(df), max_rows))
            ]
        ),
        style={"height": "265px", 'width': '960px',
               "overflowY": "scroll", 'margin': '0 auto'},
    )


@ app.callback(Output('table-tab4', 'children'),
               [Input('dropdown-tab4', "value")])
def _update_cluster(dropdown):
    return generate_html_table(n_clusters=dropdown)


@ app.callback(Output('table-tab4-2', 'figure'),
               [Input('dropdown-tab4', "value")])
def _update_cluster(n_clusters):

    df = pd.read_csv(
        f"data/clusters/count_clusters{n_clusters}.csv", index_col=0)
    fig = px.bar(df, x="clusters",
                 # category_orders={'clusters':df.sort_values('clusters')},
                 y="total_reviews",
                 title="Distribution of Clusters")

    fig.update_layout(title_x=.5,
                      xaxis=dict(type='category',
                                 categoryarray=[x for x in sorted(df['clusters'])])
                      )

    return fig


def _scatter_view(data, x_val, log):

    fig = px.scatter(data, x='price', y='points',
                     title=f"Distribution of Wine Price and Ratings",
                     color=x_val, hover_name='variety', opacity=.4,
                     hover_data=['country'], height=600,
                     log_x=log)

    fig.update_layout(showlegend=False, title_x=.5,
                      # title=f"Distribution of {x_val} <br>by {color}",
                      xaxis_title=f"Mean Price per Bottle (in USD)",
                      yaxis_title=f"Mean Points Range",
                      margin=dict(
                          l=0,
                          r=15,
                          b=15,
                          t=65
                      )
                      # xaxis={'type':'category'},
                      # margin=dict(t=100, l=50)
                      )
    return fig


@ app.callback(
    dash.dependencies.Output(
        'crossfilter-indicator-scatter', 'figure'),  # output the figure
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),  # price
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),  # log or not
     dash.dependencies.Input('crossfilter-yaxis-type', 'value'),  # log or not
     # dash.dependencies.Input('crossfilter-year--slider', 'value') #year
     ])
def update_graph(xaxis_column_name,
                 xaxis_type, yaxis_type):

    fig = _scatter_view(data=scatt, x_val=xaxis_column_name,
                        log=yaxis_type)

    return fig


# def create_time_series(df, country):
#     return tf_idf_words(df, country)

# def create_time_series(df, country):
#     return tf_idf_words(df, country)

def tf_idf_words_t(df, country_var, variety, ngram, province):

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


@ app.callback(
    dash.dependencies.Output('province-selector1', 'options'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value')]
)
def update_tf_idf(clickData, xaxis_column_name, axis_type):

    country_name = clickData['points'][0]['customdata']
    hover = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    dff = df_train[(df_train['country'] == country_name) &
                   (df_train['variety'] == hover)].copy()

    # [{'label': i, 'value': i} for i in fnameDict[name]]
    return [{'label': province, 'value': province} for province in dff['province'].unique()]


print()


@ app.callback(
    dash.dependencies.Output('province-selector1', 'value'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')]
)
def update_tf_idf(clickData):

    country_name = clickData['points'][0]['customdata']
    hover = clickData['points'][0]['hovertext']
    print(clickData)
    # print(clickData)
    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    dff = df_train[(df_train['country'] == country_name) &
                   (df_train['variety'] == hover)].copy()

    # [{'label': i, 'value': i} for i in fnameDict[name]]
    return dff['province'].value_counts().index[0]


@ app.callback(
    dash.dependencies.Output('fig-1', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('province-selector1', 'value')]
)
def update_tf_idf(clickData, xaxis_column_name, axis_type, provinces):

    country_name = clickData['points'][0]['customdata']
    hover = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    if provinces is list:
        isin_val = provinces
    else:
        isin_val = [provinces]

    df = df_train[(df_train['country'] == country_name) & (
        df_train['variety'] == hover) & (df_train['province'].isin(isin_val))].copy()

    return tf_idf_words_t(df, country_name, hover, ngram=axis_type, province=provinces)


@ app.callback(
    dash.dependencies.Output('table', 'data'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('table',  'page_size'),
     dash.dependencies.Input('table', "page_current"),
     dash.dependencies.Input('table', "filter_query"),
     dash.dependencies.Input('table', "sort_by")])
def update_x_timeseries(clickData, axis_type, page_size, page_current, filter_query, sort_by):

    country_name = clickData['points'][0]['customdata']
    hover = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    dff = df_train[(df_train['country'] == country_name)
                   & (df_train['variety'] == hover)]
    filtering_expressions = filter_query.split(' && ')

    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=True
        )
    else:
        # No sort is applied
        pass

    return dff.iloc[
        page_current*page_size:(page_current + 1)*page_size
    ].sort_values('points').to_dict('records')


@ app.callback(
    dash.dependencies.Output('fig-2', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('province-selector1', 'value')]
)
def update_tf_idf(clickData, xaxis_column_name, axis_type, province):

    country_name = clickData['points'][0]['customdata']
    hover = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    # print(country_name == 'General')

    fig = px.scatter(df_train[(df_train['country'] == country_name) & (df_train['variety'] == hover) & (df_train['province'].isin([province]))],
                     x='price', y='points', color='rating_cat', opacity=.5,
                     hover_name='winery', hover_data=['title', 'country'],
                     log_x=True, height=500, title="Distribution clusters")
    fig.update_layout(showlegend=False)
    return fig


@ app.callback(
    dash.dependencies.Output('inf-1', 'children'),
    [dash.dependencies.Input('fig-2', 'hoverData')]
)
def update_tf_idf(clickData):
    # if not clickData:
    #     return dash.no_update

    country_name = clickData['points'][0]['customdata'][1]
    title_name = clickData['points'][0]['customdata'][0]
    # hover = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass
    # print()
    # print('teste df train', df_train[df_train['title'] == title_name])
    # print(df_train[(df_train['country'] == country_name)]['title'])
    dff = df_train[(df_train['country'] == country_name)
                   & (df_train['title'] == title_name)]

    var_aspects = dff['aspects'].values
    wineries = dff['winery'].nunique()
    # print('aspects selector', dff)

    return html.Div([
        html.Div([

                    html.Div(
                        children=f"Some Informations for the Wine title", className="font-md text-center"),

                    html.Div(children=[html.Div(
                        children=f"{title_name}")], style={'color': '#4f98ca'}, className="font-lg text-center bold")
                    ], className="bottom32"),


        dbc.Row([

            dbc.Col(
                children=[
                    html.Div(id='display-6',
                             children=[html.Div(
                                 children=len(var_aspects), className="font-lg text-center text-color bold")]
                             ),
                    html.Div(className="font-md text-color padding8 text-center",
                             children="Reviews",)
                ], width=4, className="bottom32"),


            dbc.Col(
                children=[
                    html.Div(id='display-7',
                             children=[html.Div(children=wineries, className="font-lg text-center text-color bold")],),
                    html.Div(className="font-md text-color padding8 text-center",
                             children="Wineries")
                ], width=4, className="bottom32"),
            dbc.Col(
                children=[
                    html.Div(id='display-8',
                             children=[html.Div(children=dff['taster_name'].nunique(), className="font-lg text-center text-color bold")]),
                    html.Div(className="font-md text-color padding8 text-center",
                             children="Tasters")
                ], width=4, className="bottom32"),
        ], className="boxed margin-auto text-color font-md bottom16"),
        html.Div(
            children=[
                html.Div(
                    children="Principal Wine Aspects", className="font-md text-center text-color bold bottom16",),
                html.Div(id='display-9',
                         children=[html.Div(className="font-lg text-color padding8 text-center",
                                            children=var_aspects[random.randrange(len(var_aspects))])]),

            ])

    ], style={'padding': '12px 36px'}
    )
    # 'display':'inline'
    # ] )
    #                         html.Div(style={'padding':'.5rem'},
    #   children=f"{var_aspects[random.randrange(len(var_aspects))]}")


@ app.callback(
    dash.dependencies.Output('display-1', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_tf_idf(clickData,):

    if clickData['points'][0]['customdata'] != 'General':
        country_name = clickData['points'][0]['customdata']
        variety_name = clickData['points'][0]['hovertext']
        country_name = ''.join(country_name)
#        dff = df_train[df_train['country'] == country_name]
        title = f'{country_name}, {variety_name}'

        return html.Div(style={'padding': '.5rem'},
                        children=title)
    else:
        dff = df_train
        title = "Total reviews (No country specific)"

        return html.Div(style={'padding': '.5rem'},
                        children=title)


@ app.callback(
    dash.dependencies.Output('display-2', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def _update_display(clickData,):

    if clickData['points'][0]['customdata'] != 'General':
        country_name = clickData['points'][0]['customdata']
        variety_name = clickData['points'][0]['hovertext']
        country_name = ''.join(country_name)

        dff = df_train[(df_train['country'] == country_name) &
                       (df_train['variety'] == variety_name)]
        title = f'{len(dff)}'
        return html.Div(style={'padding': '.5rem'},
                        children=title)

    else:
        dff = df_train
        title = f'{len(dff)}'
        return html.Div(style={'padding': '.5rem'},
                        children=title)


@ app.callback(
    dash.dependencies.Output('display-3', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_tf_idf(clickData,):

    if clickData['points'][0]['customdata'] != 'General':
        country_name = clickData['points'][0]['customdata']
        variety_name = clickData['points'][0]['hovertext']
        country_name = ''.join(country_name)
        dff = df_train[(df_train['country'] == country_name) &
                       (df_train['variety'] == variety_name)]
        title = f"{dff['winery'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)
    else:
        dff = df_train
        title = f"{dff['winery'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)


@ app.callback(
    dash.dependencies.Output('display-4', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_tf_idf(clickData,):

    if clickData['points'][0]['customdata'] != 'General':
        country_name = clickData['points'][0]['customdata']
        variety_name = clickData['points'][0]['hovertext']
        country_name = ''.join(country_name)
        dff = df_train[(df_train['country'] == country_name) &
                       (df_train['variety'] == variety_name)]
        title = f"{dff['title'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)
    else:
        dff = df_train
        title = f"{dff['title'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)


@ app.callback(
    dash.dependencies.Output('display-10', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_tf_idf(clickData, ):

    if clickData['points'][0]['customdata'] != 'General':
        country_name = clickData['points'][0]['customdata']
        variety_name = clickData['points'][0]['hovertext']
        country_name = ''.join(country_name)
        dff = df_train[(df_train['country'] == country_name) &
                       (df_train['variety'] == variety_name)]
        title = f"{dff['province'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)
    else:
        dff = df_train
        title = f"{dff['province'].nunique()}"
        return html.Div(style={'padding': '.5rem'},
                        children=title)


@ app.callback(
    dash.dependencies.Output('title-table', 'children'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'clickData')])
def update_tf_idf(clickData):

    country_name = clickData['points'][0]['customdata']
    variety_name = clickData['points'][0]['hovertext']

    if type(country_name) == list:
        country_name = ''.join(country_name)
    else:
        pass

    dff = df_train[df_train['country'] == country_name]
    return html.Div(className="padding8 ",
                    children=f'Table Data for: {country_name}, {variety_name}')


if __name__ == '__main__':
    app.run_server(debug=True)
