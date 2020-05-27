
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
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import os 


data_path = 'data/winemag-data-130k-v2.csv'
## Importing the dataset
df_train = pd.read_csv(data_path, index_col=0)

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
app.config.suppress_callback_exceptions = True


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn 

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

newStopWords = ['fruit', "Drink", "black", 'wine', 'drink']

stopwords.update(newStopWords)

def _wordcoud():
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=300,
        max_font_size=200, 
        width=1000, height=800,
        random_state=42,
    ).generate(" ".join(df_wine1['description'].astype(str)))

    print(wordcloud)
    fig = plt.figure(figsize = (12,14))
    plt.imshow(wordcloud)
    # plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)
    # plt.axis('off')
    # plt.show()

app.layout = html.Div([
    html.H1('Dash Tabs component demo'),
    dcc.Tabs(id="tabs-example",
             children=[

                    dcc.Tab(label='Home', value='tab-1-example'),
                    dcc.Tab(label='Profilling data', value='tab-2-example'),
                    dcc.Tab(label='Tab Three', value='tab-3-example'),
                    dcc.Tab(label='Tab Four', value='tab-4-example'),
                    dcc.Tab(label='Tab Five', value='tab-5-example')
            ],  
            value='tab-1-example', 
            className='ten columns offset-by-one'),
    
    html.Div(id='tabs-content-example')
], style={'margin':'0 auto'})



@app.callback(Output('tabs-content-example', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1-example':
        return html.Div([
                    html.Div([
                        dcc.Upload([
                            'Drag and Drop or ',
                            html.A('Select a File')
                        ], style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                    })
            ], className='row'),
                    html.Div(
                        dash_table.DataTable(
                            id='table-sorting-filtering',
                            columns=[
                                {'name': i, 'id': i, 'deletable': True} for i in df_train.columns
                            ],
                            #className='ten columns offset-by-one',
                            style_table={'overflowX': 'scroll'},
                            style_cell={
                                'height': '90',
                                # all three widths are needed
                                'minWidth': '140px',
                                'width': '140px', 
                                'maxWidth': '140px',
                                'whiteSpace': 'normal'
                            },
                            page_current= 0,
                            page_size= 20,
                            page_action='custom',
                            filter_action='custom',
                            filter_query='',
                            sort_action='custom',
                            sort_mode='multi',
                            sort_by=[]
                            
                ), className='ten columns offset-by-one')

])
        
    elif tab == 'tab-2-example':
        return html.Div([
            html.H3('Data Profiling: '),
            html.Iframe(
                id='graph-1-tabs',
                src=app.get_asset_url('profiling1.html'),
                #src=app.get_asset_url('trich-dash.jpg'),
                #className='ten columns offset-by-one',
                style={ 'margin': '0 auto',
                        'width': '85%',
                        'height': '700px'}
                )            
        ], className='ten columns offset-by-one')

    elif tab == 'tab-3-example':
        return html.Div(dash_table.DataTable(
                            id='table-sorting-filtering',
                            columns=[
                                {'name': i, 'id': i, 'deletable': True} for i in df_train.columns
                            ],
                            #className='ten columns offset-by-one',
                            style_table={'overflowX': 'scroll'},
                            style_cell={
                                'height': '90',
                                # all three widths are needed
                                'minWidth': '140px',
                                'width': '140px', 
                                'maxWidth': '140px',
                                'whiteSpace': 'normal'
                            },
                            page_current= 0,
                            page_size= 10,
                            page_action='custom',
                            filter_action='custom',
                            filter_query='',
                            sort_action='custom',
                            sort_mode='multi',
                            sort_by=[]
                ), className='six columns')

    elif tab == 'tab-4-example':
            return html.Div([
                dcc.Dropdown(
                            id='dropdown',
                            options=[{'label': i, 'value': i} for i in df_train['category'].unique()],
        value='US'
    ),
                        dcc.Graph(
                            id='graph-4-tabs',
                            figure=tf_idf_words(), style={'margin':'0 auto'}
                        )
            ], className='ten columns offset-by-one', style={'margin':'0 auto'})




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


@app.callback(Output('table-sorting-filtering', 'data'),
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

@app.callback(Output('table-sorting-filtering', 'data'),
             [Input('country-buttons', "page_current")])
def tf_idf_words(country_var):
    vectorizer = TfidfVectorizer(ngram_range = (3, 3), min_df=2, 
                                 stop_words='english', max_features=20,
                                 max_df=.5)

    X2 = vectorizer.fit_transform(df_train.loc[(df_train.country == country_var)]['description']) 
   
    features = (vectorizer.get_feature_names()) 
    scores = (X2.toarray()) 
    
    # Getting top ranking features 
    sums = X2.sum(axis = 0) 
    data1 = [] 
    
    for col, term in enumerate(features): 
        data1.append( (term, sums[0,col] )) 

    ranking = pd.DataFrame(data1, columns = ['term','rank']) 
    words = (ranking.sort_values('rank', ascending = False))[:15]
    
    fig = px.bar(words, x='term', y='rank')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

