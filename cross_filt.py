import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash_table

import re
from app import app
from operator import itemgetter
def remove_year(year):
    #years= year.str.extract(r'([0-9]{4})')
    expr= re.compile('[0-9]{4}')
    line = re.sub(expr, '', year) 
    return line

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
#available_indicators = df['Indicator Name'].unique()


def _scatter_view(data, x_val, y_val, log):

    fig = px.scatter(data, x='price', y='points', 
                     title=f"Distribution of Wine Price and Ratings",
                     color=x_val, hover_name='variety', opacity=.4,
                     hover_data=['country'], height=600, 
                     log_x=log)

    fig.update_layout(showlegend=False, title_x=.5, 
                    #title=f"Distribution of {x_val} <br>by {color}",
                    xaxis_title=f"Price per Bottle (in USD)", 
                    yaxis_title=f"Points Range",
                    #xaxis={'type':'category'},
                    #margin=dict(t=100, l=50)
            )
    return fig


def tf_idf_words(df, country_var, variety, ngram, province):
    
    if ngram == 1:
        ngran_range=(1,2)
        title=(f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")
    elif ngram == 2:
        ngran_range=(2,2) 
        title=(f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")           
    elif ngram == 3:
        ngran_range=(3,3)
        title=(f"TfIdf Unigram for <br>{country_var}, {province}, {variety}")        

    vectorizer = TfidfVectorizer(ngram_range = ngran_range, min_df=1, 
                                 stop_words='english', max_features=10,
                                 max_df=500)

    #X2 = vectorizer.fit_transform(df_train.loc[(df_train.country == country_var)]['description']) 
    X2 = vectorizer.fit_transform(df['description_desc']) 
     
    features = (vectorizer.get_feature_names()) 
    scores = (X2.toarray()) 
    
    # Getting top ranking features 
    sums = X2.sum(axis = 0) 
    data1 = [] 
    
    for col, term in enumerate(features): 
        data1.append( (term, sums[0,col] )) 

    ranking = pd.DataFrame(data1, columns = ['term','rank']) 
    words = (ranking.sort_values('rank', ascending = False))[:15]
    
    fig = px.bar(words, x='term', y='rank', title=title, height=500)
    fig.update_layout(autosize=False, title_x=.5)
    return fig

columns=['province', 'winery', 'title', 'harvest', 'points', 'price']
# print(scatt.columns)
layout2 = html.Div([
        html.Div([
            html.Div([    
                html.P("Colored by: ", style={'color':'#272727', 'fontSize':'15px'}),
                dcc.Dropdown(
                    id='crossfilter-xaxis-column',
                    options=[#{'label': 'Review Cluster', 'value': 'clusters'},
                             {'label': 'Country', 'value': 'country'},
                             {'label': 'Harvest', 'value': 'harvest'},
                             {'label': 'Rating', 'value': 'rating_cat'}],
                    value='rating_cat', placeholder='Select a feature')
                    ], className='two columns', style={'textAlign':'center'}
            ),
            html.Div([
                html.P("Price Distribution", style={'color':'#272727', 'fontSize':'15px'}),
                dcc.RadioItems(
                    id='crossfilter-yaxis-type',
                    options=[{'label': 'Linear', 'value': 0},
                            {'label': 'Log', 'value': 1}],
                    value=1,
                    labelStyle={'display': 'inline-block'}, 
                )], className='two columns', style={'textAlign':'center'}),
            html.Div([    
                html.P("testetes", style={'color':'#272727', 'margin':'0 auto', 'fontSize':'15px'}),
                dcc.Dropdown(
                    id='crossfilter-yaxis-column',
                    options=[{'label': i, 'value': i} for i in ['points']],
                    value='points')], className='two columns', style={'margin':'0 auto', 'textAlign':'center'}
                ),
        ], className='row', style={'borderLeft': 'thin lightgrey solid',
                                   'backgroundColor': '#50d890',
                                   'padding': '10px','margin':'0 auto'}),

    ## Gráficos
    html.Div([
        # Scatter plot
        html.Div([
            ## Gráfico principal que tem o hover
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                clickData={'points': [{'customdata': 'Portugal', 'hovertext':'Portuguese White'}]},
                #hoverData={'points': [{'customdata': 'General', 'hovertext':'Ramisco'}]
                
                #clear_on_unhover=True, 
            )
        ], className='seven columns', style={'height':'600px', 'display':'inline'}),

        html.Div([
            html.Div([
                html.Div(
                    id="number-plate",
                    children=[
                        html.Div(
                            children=[
                                html.H5(style={'textAlign':'center','padding':'.1rem',
                                                'fontSize':'22px'}, className='title',
                                                    children="Information for Country/Variety"),                                     
                                html.H3(id='display-1', style={'textAlign':'center',
                                                'fontWeight':'bold','color':'#4f98ca'})
                                   
                                                    ], className='row'),
                        html.Div([
                                html.Div(
                                    children=[
                                        html.H3(id='display-2', style={'textAlign':'center',
                                                    'fontWeight':'bold','color':'#272727'}),
                                        html.H5(style={'textAlign':'center','color':'#272727','padding':'.1rem'},
                                                        children="Reviews")                                        
                                                ], className='three columns', style={'display':'inline-block'}),
                                html.Div(
                                    children=[
                                        html.H3(id='display-10', style={'textAlign':'center',
                                                    'fontWeight':'bold','color':'#272727'}),
                                        html.H5(style={'textAlign':'center','color':'#272727','padding':'.1rem'},
                                                        children="Provinces")                                        
                                                ], className='three columns', style={'display':'inline-block'}),
                                html.Div(
                                    children=[
                                        html.H3(id='display-3', style={'textAlign':'center',
                                                    'fontWeight':'bold','color':'#272727'}),
                                        html.H5(style={'textAlign':'center','color':'#272727','padding':'.1rem'},
                                                        children="Wineries")                                        
                                                ], className='three columns', style={'display':'inline-block'}),
                                html.Div(
                                    children=[
                                        html.H3(id='display-4', style={'textAlign':'center',
                                                'fontWeight':'bold','color':'#272727'}),
                                        html.H5(style={'textAlign':'center','color':'#272727','padding':'.1rem'},
                                                        children="Unique Titles")                                        
                                                ], className='three columns', style={'display':'inline-block'})
                                                
                                ], className='row')
              ], className='row', style={'padding': '24px 0',
                                         'backgroundColor':'#effffb',
                                         'height':'35%', 
                                         #'display':'inline'
                                            }, ),

                            html.Div([
                                html.Div(id='title-table'),
                                dash_table.DataTable(id='table',
                                                columns=[{"name": i, "id": i} for i in columns],
                                                page_current=0, 
                                                page_size=5,
                                                page_action='custom',
                                                filter_action='custom',
                                                sort_by=[],
                                                sort_mode='single',
                                                sort_action='custom',
                                               #sort_by='points', sort_mode='single',
                                               # sortDirection='ASC',
                                                filter_query='',
                                                #style_data={'whiteSpace':'normal'},
                                                #style_table={'overflowX': 'scroll'},
                                                fixed_columns={'headers': True, 'data': 0},
                                                fixed_rows={'headers': True, 'data': 5},
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
                                                        # {
                                                        #     'if': {'column_id': c},
                                                        #     'maxWidth': '90px',
                                                        #     'textAlign':'left',
                                                        # #    'overflowY': 'hidden',
                                                        #     'textOverflow':'hidden'
                                                        # } for c in ['province', 'winery', 'title']
                                                        {'if': {'column_id': 'variety'}, 'maxWidth': '50px'},
                                                    ], 
                        )], className="row", style={'height':'50%', 'margin':'0 auto'})
                ], className='five columns', style={'height':'600px'}),
        
]),        ], className="row", style={'padding':'24px 0'}),
            html.Div([
            html.Div(id='mark-1', children=[
                dcc.Markdown('''
                    ###  Reviews Specification
                    You can select in the Scatter plot on the right the Titles and see some aspects of the reviews of sommeliers.


                    #### Reviews Specications are composed by some parts 2 basic parts:
                    - TFIDF - You can select the N-gram that you desire for each Country/Variety pair. 
                    - Scatter Plot Chart where you can select an specific variety to see some informations and main aspects of sommelier reviews.  
          
                    The Aspects of reviews was extracted by SpaCy using some POS tagging techniques;
                     
                    NOTE: To avoid huge computational calculations, I'm limiting for only one review aspect, chosen randomly by title. 
                    '''),
            ], className='ten columns offset-by-one',style={'backgroundColor':'#effffb', 
                                                'padding':'24px 36px 12px',
                                                'textAlign':'center'}), 
            ], className='row', style={'margin':'0 auto'}),
            html.Div([
                html.Div([    
                    html.P("TfIdf N-grams: ", style={'color':'#272727', 'fontSize':'18px'}),
                    dcc.RadioItems(
                    id='crossfilter-xaxis-type', 
                    options=[{'label': 'Unigram', 'value': 1},
                            {'label': 'Bigram', 'value': 2},
                            {'label': 'Trigram', 'value': 3}],
                    value=1,
                    labelStyle={'display': 'inline-block'})
                    ],    className='three columns', style={'textAlign':'center'}),
                html.Div([    
                    html.P("Select the Province(s): ", style={'color':'#272727', 'fontSize':'18px'}),
                    dcc.Dropdown(
                        id='province-selector1', 
                        multi=False, #labelStyle={'display': 'inline-block'}
                    )
                    ],    className='three columns', style={'textAlign':'center'})
            ], className='row', style={'borderBottom': 'thin lightgrey solid',
                                       'backgroundColor': '#50d890',
                                       'padding': '30px 0','margin':'24px 0'}),
        # Table and tfidf plot
        html.Div([
            html.Div([
                dcc.Graph(id='fig-1')
                ], className='eight columns'),    # style={'width':'75%', 'margin':'0 auto'},     
            html.Div([
                dcc.Graph(id='fig-2',
                clickData={'points': [{'customdata': 'Portugal', 'hovertext':'Portuguese White'}]},
                hoverData={'points': [{'customdata': ['Seacampo  Reserva Red (Dão)', 'Portugal']}]},)
                ], className='four columns')

        ], className='row', style={'padding':'24px 0'}),

        html.Div([
            html.Div([
                html.Div(id='inf-1')
                ], className='twelve columns'),    # style={'width':'75%', 'margin':'0 auto'},     
            # html.Div([
            #     dcc.Graph(id='fig-2')
            #     ], className='four columns')

        ], className='row', style={'padding':'24px 0'})
               
])

