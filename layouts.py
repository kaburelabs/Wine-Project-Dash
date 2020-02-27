

import dash_core_components as dcc
import dash_html_components as html

theme = {'font-family': 'Raleway', 'background-color': '#787878'}


def graph_1():
    graph = html.Div([   
            dcc.Graph(id='Graph1', 
                      className="six columns", 
                      #style={'border':'dotted'}
                    ),
            dcc.Graph(id='Graph4', 
            #style={'border-style': 'dotted', }, 
            className="six columns", )
            ])
    return [graph]


def graph2_3():
    graph_2 = html.Div([
                dcc.Graph(id='Graph2',
                        )], className="four columns", style={'margin':'0'})
    graph_5 = html.Div([
                dcc.Graph(id='Graph5',
                        )], className="four columns", style={'margin':'0'})
    graph_3 = html.Div([   
                dcc.Graph(id='Graph3',
                        )], className="four columns", style={'margin':'0'})

    # Return of the graphs in all the row
    return [graph_2, graph_3, graph_5]


def create_header(some_string):
    header_style = {
        'background-color':'rgb(191, 208, 247)',
        'padding': '1.5rem',
        'display':'inline-block',
        'width':'100%'
        
       # 'border-style': 'dotted'
    }
    logo_trich = html.Img(
                    src='/assets/fundo_transp.png',
                    className='three columns',
                    style={
                        'height': 'auto',
                        'width': '140px', # 'padding': 1
                        'float': 'right', #'position': 'relative'
                        'margin-right': '54px', #'border-style': 'dotted'
                        'display':'inline-block'})

    title = html.H1(children=some_string, className='eight columns', style={'margin':'0 0 0 24px', 'font-size':'32px'})

    header = html.Header(html.Div([title, logo_trich]), style=header_style)

    return header


def create_footer():
    p = html.P(
        children=[
            html.Span('Developed By: '),
            html.A('trich.ai | Data Intelligence Solutions', 
                   style={'text-decoration':'none', 'color':'black'},
                   href='https://trich.ai', target='_blank')
        ], style={'float':'right', 'margin-top':'8px', 
                  'font-size':'18px', 'color':'black' }
    )

    span_style = {'horizontal-align': 'right', 
                  'padding-left': '1rem', 
                  'font-size':'15px', 
                  'vertical-align':'middle'}

    kaggle = html.A(
        children=[
            html.I([], className='fab fa-kaggle'),
            html.Span('Kaggle', style=span_style)
        ], style={'text-decoration': 'none', 'color':'black', 'margin-right':'20px'},
        href="https://www.kaggle.com/kabure/kernels",
        target='_blank')

    mapbox = html.A(
        children=[
            html.I([], className='fab fa-python'),
            html.Span('Dash Plotly', style=span_style)
        ], style={'text-decoration': 'none', 'color':'black', 'margin-right':'20px'},
        href='https://plot.ly/dash/', target='_blank')

    font_awesome = html.A(
        children=[
            html.I([], className='fa fa-font-awesome'),
            html.Span('Font Awesome', style=span_style)
        ], style={'text-decoration': 'none', 'color':'black', 'margin-right':'20px'},
        href='http://fontawesome.io/', target='_blank')

    datatables = html.A(
        children=[
            html.I([], className='fa fa-github'),
            html.Span('trich.ai\n Github', style=span_style)
        ], style={'text-decoration': 'none', 'color':'black', 'margin-right':'20px'},
        href='https://github.com/kaburelabs/', target='_blank')

    ul1 = html.Div(
        children=[
            html.Li(mapbox, style={'display':'inline-block', 'color':'black'}),
            html.Li(font_awesome, style={'display':'inline-block', 'color':'black'}),
            html.Li(datatables, style={'display':'inline-block', 'color':'black'}),
            html.Li(kaggle, style={'display':'inline-block', 'color':'black'}),
        ],
        style={'list-style-type': 'none', 'font-size':'30px'},
    )

    hashtags = 'plotly,dash,trich.ai'
    tweet = 'trich.ai Customers Churn Dashboard, a cool dashboard with Plotly Dash!'
    twitter_href = 'https://twitter.com/intent/tweet?hashtags={}&text={}'\
        .format(hashtags, tweet)
    twitter = html.A(
        children=html.I(children=[], className='fa fa-twitter'),
        title='Tweet me!', href=twitter_href, target='_blank')

    github = html.A(
        children=html.I(children=[], className='fa fa-github', style={'color':'black'}),
        title='Repo on GitHub',
        href='https://github.com/kaburelabs/Dash-app-Churn-Customer', target='_blank')

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
            'font-size':'60px'
        }
    )
    div = html.Div([p, ul1, ul2])

    footer_style = {
        'font-size': '2.2rem',
        'background-color': 'rgb(191, 208, 247)',
        #'padding': '2.5rem',
        'margin-top': '3rem', 
        'display':'inline-block', 'padding':'16px 32px 8px'
    }
    footer = html.Footer(div, style=footer_style, className='twelve columns')
    return footer


def header_logo():
    h1_title = html.H1(
                    children='Churn Customer Prediction',
                    #className="eight columns", 
                    style={#'border-style': 'dotted', 
                           'margin-top':20,
                           #'float':'left',
                           #'margin-left': 100,
                           'font-size':'50px'
                            #'display':'inline-block'
                            })
    subtitle = html.Div(dcc.Markdown('''
      **Churn rate**, when applied to a customer base, refers to the proportion of contractual customers or subscribers who leave a supplier during a given time period. It is a possible indicator of customer dissatisfaction, cheaper and/or better offers from the competition, more successful sales and/or marketing by the competition, or reasons having to do with the customer life cycle.     
    When talking about subscribers or customers, sometimes the expression **"survival rate"** is used to mean 1 minus the churn rate. For example, for a group of subscribers, an annual churn rate of 25 percent is the same as an annual survival rate of 75 percent. Both imply a customer lifetime of four years. I.e., a customer lifetime can be calculated as the inverse of that customer's predicted churn rate. For a group or segment of customers, their customer life (or tenure) is the inverse of their aggregate churn rate. Gompertz distribution models of distribution of customer life times can therefore also predict a distribution of churn rates.
'''))
    return [h1_title, subtitle]

def paragraphs():
    div = html.H1("Revenue Churn", style={'width':'85%', 
                                          'margin':'0 auto', 
                                          'text-align':'center','padding':'24px 0px 10px'})
    paragra = html.P(dcc.Markdown("  **Revenue churn** is the monetary amount of recurring revenue lost in a period divided by the total revenue at the beginning of the period. Revenue churn is commonly used in Software as a Service (SaaS) and other business models that rely on recurring revenue models."), 
    style={'width':'85%', 'margin':'0 auto', 
                    'padding-bottom':'24px', 
                           # 'margin':'24px 0'
                            })

    return [div, paragra]

