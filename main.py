import base64
import json
import io
#import datetime

import dash
from dash import Dash
from dash import no_update
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.express as px
import dash_mantine_components as dmc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME])

def header():
    return dbc.Row([
        dbc.Col(dcc.Link(html.I(className="fa fa-home fa-2x", id="css-home-icon"), href='/'), width="auto"),
        dbc.Col(html.H3('MSD Transformer app', id="css-header-title"), width="auto"),
        dbc.Col(dcc.Link(html.I(className="fa fa-info fa-2x", id="css-info-icon"), href='/information'), width="auto")
    ], id="css-header")

def footer():
    github_url = "https://github.com/dabrze/topsis-msd-improvement-actions"
    return dash.html.Footer(children=[
        html.A(html.I(className="fab fa-github fa-2x", id="css-github-icon"), href=github_url, target="_blank"),
        html.Div(html.Img(src="assets/PP_znak_pełny_RGB.png", id="css-logo-img"), id="css-logo-div")
    ], id="css-footer")

def home_layout():
    return html.Div(children=[
        html.Div([
            html.Button(dcc.Link('Load your dataset using WIZARD', href='/wizard'), className='big-button'),
            html.Button(dcc.Link('Experiment with ready dataset', href='/main_dash_layout'), className='big-button'),
        ], className='button-container')
    ])

#==============================================================
#   WIZARD
#==============================================================

def wizard():
    return html.Div([
        
        # Data before Submit
        html.Div([
            html.Div([
                html.Div('Decimal:'),
                dcc.Input(id='wizard-data-input-decimal',
                            type = 'text',
                            placeholder='.',
                            minLength=0,
                            maxLength=1),
                html.Div('Delimiter:'),
                dcc.Input(id='wizard-data-input-delimiter',
                            type = 'text',
                            placeholder=',',
                            minLength=0,
                            maxLength=1),
                ]),
            html.Div('Upload data'),
            html.Div([
                dcc.Store(id='wizard_state_stored-data', data=None),
                dcc.Upload(
                    id='wizard-data-input-upload-data',
                    children=html.Div([
                        'Drag and Drop or Select Files'
                    ], id = 'wizard-data-output-upload-data-filename'),
                    multiple=False
                ),
                html.Div(id='wizard-data-input-remove-data'),
                html.Div(id='wizard_data_input_submit-button'),
                ], id = 'wizard-data-input-remove-upload-data'),

            html.Div('Upload parameters'),
            html.Div([
                dcc.Store(id='wizard_state_stored-params', data=None),
                dcc.Upload(
                    id='wizard-data-input-upload-params',
                    children=html.Div([
                        'Drag and Drop or Select Files'
                    ], id = 'wizard-data-output-upload-params-filename'),
                    multiple=False
                ),
                html.Div(id='wizard-data-input-remove-params'),
                ], id = 'wizard-data-input-remove-upload-params'),

            html.Div(id='wizard-data-output-parsed-data-before'),
            html.Div(id='wizard-data-output-parsed-params'),
            html.Div(id='data-preview'),
            html.Div(id='data-table', style={'display': 'none'}),
            #html.Div(id='wizard_data_input_submit-button'),
            #html.Div(id='wizard_data_input_submit-button')
            #html.Button("Submit", id='wizard_data_input_submit-button', n_clicks=0, style={'display':'none'})
            html.Button("Submit", id='wizard_data_input_submit-button', style={'display':'none'}),
        ], id='data_upload_layout', style={'display': 'block'}),

        # Data after Submit
        html.Div([
            html.Div([
                html.Div("project_title", id='wizard-data-input-title'),
                #html.Button(id='wizard-data-input-title-button', children='title')
            ], id='wizard-data-after-submit-output-project-title'),
            html.Div(id='wizard-data-output-parsed-data-after'),
            html.Button('Next', id='data-to-param', className='next-button')
        ], id='data_layout', style={'display': 'none'}),

        #Parameters
        html.Div([
            html.Div(id='wizard-parameters-output-params-table'),
            html.Div(id = 'wizard-parameters-output-warning', children = ''),
            html.Button('Back', id='param-to-data', className='back-button'),
            html.Button('Next', id='param-to-model', className='next-button')
        ], id='parameters_layout', style={'display': 'none'}),

        #Model
        html.Div([
            html.Div([
                dcc.RadioItems([
                    {
                    "label":[
                        html.Div("R description")
                    ],
                    "value": 'R',
                    },{
                    "label":[
                        html.Div("I description")
                    ],
                    "value": 'I',
                    },{
                    "label":[
                        html.Div("A description")
                    ],
                    "value": 'A',
                    },
                ], value='R', inline=False, id="wizard-model-input-radio-items"),
                html.Div(
                    style = {
                        'height': '50px',
                        'width' : '50px',
                        'background-color': '#FF0000'
                        },
                    id = "wizard-model-output-view"
                ),
            ], id="css-radio-items"),
            html.Div([
                dcc.RadioItems([
                    {
                    "label":[
                        html.Img(),
                        html.Div("link to image1")
                    ],
                    "value": 'color1',
                    },{
                    "label":[
                        html.Img(),
                        html.Div("link to image2")
                    ],
                    "value": 'color2',
                    },{
                    "label":[
                        html.Img(),
                        html.Div("link to image3")
                    ],
                    "value": 'color3',
                    },
                ], value='color1', inline=False, id="wizard-model-input-radio-items-color"),
            ], id="css-radio-items"),
            #https://dash.plotly.com/dash-core-components/radioitems
            html.Button('Back', id='model-to-param', className='back-button'),
            html.Button(dcc.Link('Finish', href='/main_dash_layout'), className='finish-button')
        ], id='model_layout', style={'display': 'none'})
    ])


@app.callback(Output('wizard-data-output-parsed-data-before', 'children', allow_duplicate=True),
              Output('wizard-data-output-parsed-data-after', 'children', allow_duplicate=True),
              Output('wizard-data-output-upload-data-filename', 'children'),
              Output('wizard-data-input-remove-data', 'children', allow_duplicate=True),
              Output('wizard_data_input_submit-button', 'style', allow_duplicate=True),
              Input('wizard-data-input-upload-data', 'contents'),
              Input('wizard-data-input-delimiter', 'n_submit'),
              State('wizard-data-input-delimiter', 'value'),
              Input('wizard-data-input-decimal', 'n_submit'),
              State('wizard-data-input-decimal', 'value'),
              State('wizard-data-input-upload-data', 'filename'),
              State('wizard-data-input-upload-data', 'last_modified'),
              prevent_initial_call=True)
def update_wizard_data_output_data(contents_data, enter_del, delimiter, enter_sep, decimal, name_data, date_data):

    if contents_data is not None:
        child = [
            parse_file_wizard_data_data(c, n, d, deli, dec) for c, n, d, deli, dec  in
            zip([contents_data], [name_data], [date_data], [delimiter], [decimal])]  
                
        remove = html.Button(id='wizard_data_input_remove-data-button', children='Remove')

        return child, child, name_data, remove, {'display':'block'}
    else:
        raise PreventUpdate


@app.callback(Output('wizard-data-input-remove-upload-data', 'children'),
              Output('wizard-data-output-parsed-data-before', 'children'),
              Output('wizard-data-output-parsed-data-after', 'children'),
              Output('wizard-data-input-remove-data', 'children'),
              Output('wizard_data_input_submit-button', 'style'),
              Input('wizard_data_input_remove-data-button','n_clicks'))
def remove_file_wizard_data_data_file(n):
    
    if n is None:
        return no_update

    child =  [
            dcc.Upload(
                id='wizard-data-input-upload-data',
                children=html.Div([
                    'Drag and Drop or Select Files'
                ], id = 'wizard-data-output-upload-data-filename'),
                multiple=False
            ),
            html.Div(id='wizard-data-input-remove-data'),
            ]
    table = None
    remove = None
    return child, table, table, remove, {'display':'none'}


@app.callback(Output('wizard-data-output-parsed-params', 'children', allow_duplicate=True),
              Output('wizard-data-output-upload-params-filename', 'children'),
              Output('wizard-data-input-remove-params', 'children', allow_duplicate=True),
              Input('wizard-data-input-upload-params', 'contents'),
              State('wizard-data-input-upload-params', 'filename'),
              State('wizard-data-input-upload-params', 'last_modified'),
              prevent_initial_call=True)
def update_wizard_data_output_params(contents_params, name_params, date_params):
    
    if contents_params is not None:
        child = [
            parse_file_wizard_data_params(c, n, d) for c, n, d in
            zip([contents_params], [name_params], [date_params])]
        
        remove = html.Button(id='wizard_data_input_remove-params-button', children='Remove')
        return child, name_params, remove
    else:
        raise PreventUpdate
 

@app.callback(Output('wizard-data-input-remove-upload-params', 'children'),
              Output('wizard-data-output-parsed-params', 'children'),
              Output('wizard-data-input-remove-params', 'children'),
              Input('wizard_data_input_remove-params-button','n_clicks'))
def remove_file_wizard_data_params_file(n):
    
    if n is None:
        return no_update

    child = [
            dcc.Store(id='wizard_state_stored-params', data=None),
            dcc.Upload(
                id='wizard-data-input-upload-params',
                children=html.Div([
                    'Drag and Drop or Select Files'
                ], id = 'wizard-data-output-upload-params-filename'),
                multiple=False
            ),
            html.Div(id='wizard-data-input-remove-params'),
            ]
    table = None
    remove = None
    return child, table, remove


def parse_file_wizard_data_data(contents, filename, date, delimiter, dec):
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    if not delimiter:
        delimiter = ','

    if not dec:
        dec = '.'
        
    try:
        if filename.endswith('.csv'):
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep = delimiter, decimal = dec)
        elif filename.endswith('.xls'):
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return "Please upload a file with the .csv or .xls extension"
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.Hr(),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=8
        ),
        #html.Button("Submit", id='wizard_data_input_submit-button'),
        #html.Button(id="wizard_data_input_submit-button", children="Submit", n_clicks=0),
        #html.Button('Submit', id='submit-button', n_clicks=0),
        dcc.Store(id='wizard_state_stored-data', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


def parse_file_wizard_data_params(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        if filename.endswith('.json'):
            content_dict = json.loads(decoded)
        else:
            return "Please upload a file with the .json extension"
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return html.Div([
        dcc.Store(id='wizard_state_stored-params', data=content_dict),

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


def check_parameters_wizard_data_files(data, params, param_keys):

    criteria = list(data[0].keys())

    df_data = pd.DataFrame.from_dict(data).set_index(criteria[0])
    df_params = pd.DataFrame.from_dict(params)

    n_alternatives = df_data.shape[0]
    m_criteria = df_data.shape[1]

    if param_keys[1] in df_params:
        if len(df_params[param_keys[1]]) != m_criteria:
            print("Invalid value 'weights'.")
            return -1
        if not all(type(item) in [int, float, np.float64] for item in df_params[param_keys[1]]):
            print("Invalid value 'weights'. Expected numerical value (int or float).")
            return -1
        if not all(item >= 0 for item in df_params[param_keys[1]]):
            print("Invalid value 'weights'. Expected value must be non-negative.")
            return -1
        if not any(item > 0 for item in df_params[param_keys[1]]):
            print("Invalid value 'weights'. At least one weight must be positive.")
            return -1
    else:
        return -1
    
    if param_keys[4] in df_params:
        if len(df_params[param_keys[4]]) != m_criteria:
            print("Invalid value 'objectives'.")
            return -1
        if not all(item in ["min", "max"] for item in df_params[param_keys[4]]):
            print("Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'.")
            return -1
    else:
        return -1
    
    if param_keys[2] in df_params and param_keys[3] in df_params:
        if len(df_params[param_keys[2]]) != m_criteria:
            print("Invalid value at 'expert_range'. Length of should be equal to number of criteria.")
            return -1
        if len(df_params[param_keys[3]]) != m_criteria:
            print("Invalid value at 'expert_range'. Length of should be equal to number of criteria.")
            return -1
        if not all(type(item) in [int, float, np.float64] for item in df_params[param_keys[2]]):
            print("Invalid value at 'expert_range'. Expected numerical value (int or float).")
            return -1
        if not all(type(item) in [int, float, np.float64] for item in df_params[param_keys[3]]):
            print("Invalid value at 'expert_range'. Expected numerical value (int or float).")
            return -1
        
        lower_bound = df_data.min() 
        upper_bound = df_data.max()

        for lower, upper, mini, maxi in zip(lower_bound, upper_bound, df_params[param_keys[2]], df_params[param_keys[3]]):
            if mini > maxi:
                print("Invalid value at 'expert_range'. Minimal value  is bigger then maximal value.")
                return -1
            if lower < mini:
                print("Invalid value at 'expert_range'. All values from original data must be in a range of expert_range.")
                return -1
            if upper > maxi:
                print("Invalid value at 'expert_range'. All values from original data must be in a range of expert_range.")
                return -1
    else:
        return -1
    
    return 1


def return_columns_wizard_parameters_params_table(param_keys):
    columns = [{
                    'id': 'criterion', 
                    'name': 'Criterion',
                    'type': 'text',
                    'editable': False
                },{
                    'id': param_keys[1], 
                    'name': 'Weight',
                    'type': 'numeric'
                },{
                    'id': param_keys[2], 
                    'name': 'Expert Min',
                    'type': 'numeric'
                },{
                    'id': param_keys[3], 
                    'name': 'Expert Max',
                    'type': 'numeric'
                },{
                    'id': param_keys[4], 
                    'name': 'Objective',
                    'presentation': 'dropdown'                    
                }]
    
    return columns


def fill_parameters_wizard_parameters_params(params, df, param_keys):

    if params is None:
        m_criteria = df.shape[1]
        return np.ones(m_criteria), df.min(), df.max(), np.repeat('max', m_criteria)
    else:
        weights = list(params[param_keys[1]].values())
        mins = list(params[param_keys[2]].values())
        maxs = list(params[param_keys[3]].values())
        objectives = list(params[param_keys[4]].values())

        return weights, mins, maxs, objectives
    
@app.callback([Output('data-preview', 'children'),
              Output('data-table', 'children', allow_duplicate=True),
              Output('wizard-parameters-output-params-table', 'children', allow_duplicate=True),
              Output('data_upload_layout', 'style', allow_duplicate=True),
              Output('data_layout', 'style', allow_duplicate=True),
              Output('parameters_layout', 'style', allow_duplicate=True),
              Output('model_layout', 'style', allow_duplicate=True)],
              [Input('wizard_data_input_submit-button', 'n_clicks')],
              [State('wizard_state_stored-data', 'data'),
              State('wizard_state_stored-params','data')],
              prevent_initial_call=True)
def submit(n_clicks, data, params):

    param_keys = ['criterion', 'weight', 'expert-min', 'expert-max', 'objective']

    if n_clicks:
        data_preview = dash_table.DataTable(
            data=data,
            columns=[{'name': i, 'id': i} for i in data[0].keys()],
            page_size=8
        )
        data_table = dcc.Store(id='wizard_state_stored-data', data=data)
        if params is not None and check_parameters_wizard_data_files(data, params, param_keys) == -1:
            print('Prevent update')
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
        
        
        columns = return_columns_wizard_parameters_params_table(param_keys)
        
        criteria = list(data[0].keys())
        df = pd.DataFrame.from_dict(data).set_index(criteria[0])

        weights, expert_mins, expert_maxs, objectives = fill_parameters_wizard_parameters_params(params, df, param_keys)

        data_params = []

        for id, c in enumerate(criteria[1:]):
            data_params.append(dict(criterion=c,
                        **{param_keys[1] : weights[id],
                        param_keys[2] : expert_mins[id],
                        param_keys[3] : expert_maxs[id],
                        param_keys[4] : objectives[id]}))
        
        if not data_params:
            print('Prevent update')
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update
            
        params_table = html.Div([
        #https://dash.plotly.com/datatable/editable
        #https://community.plotly.com/t/resolved-dropdown-options-in-datatable-not-showing/20366
        dash_table.DataTable(
            id = 'wizard-parameters-input-parameters-table',
            columns = columns,
            data = data_params,
            editable = True,
            dropdown={
                param_keys[4]: {
                    'options': [
                        {'label': i, 'value': i}
                        for i in ['min', 'max']
                    ],
                    'clearable': False
                },
             }
        ),
        html.Div("Set all to Min/Max"),
        dcc.Dropdown(['-', 'min', 'max'], '-', id = 'wizard-parameters-input-objectives-dropdown', clearable=False),
        ])
        return (data_preview, data_table, params_table, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'})
    else:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    
@app.callback([Output('data_upload_layout', 'style', allow_duplicate=True),
              Output('data_layout', 'style', allow_duplicate=True),
              Output('parameters_layout', 'style', allow_duplicate=True),
              Output('model_layout', 'style', allow_duplicate=True)],
              [Input('data-to-param', 'n_clicks')],
              prevent_initial_call=True)
def button_data_params(n_clicks):
    if n_clicks:
        return ({'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'})
    else:
        return no_update, no_update, no_update, no_update

@app.callback([Output('data_upload_layout', 'style', allow_duplicate=True),
              Output('data_layout', 'style', allow_duplicate=True),
              Output('parameters_layout', 'style', allow_duplicate=True),
              Output('model_layout', 'style', allow_duplicate=True)],
              [Input('param-to-data', 'n_clicks')],
              prevent_initial_call=True)
def button_params_data(n_clicks):
    if n_clicks:
        return ({'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'})
    else:
        return no_update, no_update, no_update, no_update

@app.callback([Output('data_upload_layout', 'style', allow_duplicate=True),
              Output('data_layout', 'style', allow_duplicate=True),
              Output('parameters_layout', 'style', allow_duplicate=True),
              Output('model_layout', 'style', allow_duplicate=True)],
              [Input('param-to-model', 'n_clicks')],
              prevent_initial_call=True)
def button_params_model(n_clicks):
    if n_clicks:
        return ({'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'})
    else:
        return no_update, no_update, no_update, no_update

@app.callback([Output('data_upload_layout', 'style', allow_duplicate=True),
              Output('data_layout', 'style', allow_duplicate=True),
              Output('parameters_layout', 'style', allow_duplicate=True),
              Output('model_layout', 'style', allow_duplicate=True)],
              [Input('model-to-param', 'n_clicks')],
              prevent_initial_call=True)
def button_model_params(n_clicks):
    if n_clicks:
        return ({'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'})
    else:
        return no_update, no_update, no_update, no_update


@app.callback(Output('wizard-data-after-submit-output-project-title', 'children', allow_duplicate=True),
             Input('wizard-data-input-title', 'n_clicks'),
             State('wizard-data-input-title', 'children'),
             prevent_initial_call=True)
def edit_title_wizard_data_after_submit(click, text):
    
    if click:
        return html.Div([
                dcc.Input(id='wizard-data-input-type-title',
                        type = 'text',
                        placeholder=text,
                        minLength=1,
                        maxLength=20),
            ])

    return no_update


@app.callback(Output('wizard-data-after-submit-output-project-title', 'children'),
             Input('wizard-data-input-type-title', 'n_submit'),
             State('wizard-data-input-type-title', 'value'))
def edit_title_wizard_data_after_submit(enter, text):
    
    if enter and text:
        return html.Div([
            html.Div(text, id='wizard-data-input-title')
            ])

    return no_update


def check_updated_params_wizard_parameters(df_data, df_params, param_keys):
    warnings = []

    #weights
    if (df_params[param_keys[1]] < 0).any():
        warnings.append("Weight must be a non-negative number")

    if df_params[param_keys[1]].sum() == 0:
        warnings.append("At least one weight must be greater than 0")

    #expert range
    lower_bound = df_data.min() 
    upper_bound = df_data.max()


    for lower, upper, mini, maxi in zip(lower_bound[1:], upper_bound[1:], df_params[param_keys[2]], df_params[param_keys[3]]):
        if mini > maxi:
            warnings.append("Min value must be lower or equal than max value")
        
        if lower < mini:
            warnings.append("Min value must be lower or equal than the minimal value of given criterion")

        if upper > maxi:
            warnings.append("Max value must be greater or equal than the maximal value of given criterion")
    
    return list(set(warnings))


def parse_warning(warning):
    return html.Div([
        warning
    ])

#Approach 2 - iterate through whole table
@app.callback(Output('wizard-parameters-output-params-table', 'children'),
              Output('wizard-parameters-output-warning', 'children'),
              Input('wizard-parameters-input-parameters-table', 'data_timestamp'),
              Input('wizard-parameters-input-objectives-dropdown', 'value'),
              State('wizard_state_stored-data','data'),
              State('wizard-parameters-input-parameters-table', 'data'),
              State('wizard-parameters-input-parameters-table', 'data_previous'))
def update_table_wizard_parameters(timestamp, objectives_val, data, params, params_previous):
    #https://community.plotly.com/t/detecting-changed-cell-in-editable-datatable/26219/3
    #https://dash.plotly.com/duplicate-callback-outputs


    param_keys = ['criterion', 'weight', 'expert-min', 'expert-max', 'objective']
    columns = return_columns_wizard_parameters_params_table(param_keys)

    criteria_params = list(params[0].keys())
    
    df_data = pd.DataFrame.from_dict(data)
    df_params = pd.DataFrame.from_dict(params).set_index(criteria_params[0])
     
    warnings = check_updated_params_wizard_parameters(df_data, df_params, param_keys)
                
    if warnings:
        children = [parse_warning(warning) for warning in warnings]
        params = params_previous
    else:
        children = html.Div([])

    if params_previous:
        df_params_prev = pd.DataFrame.from_dict(params_previous).set_index(criteria_params[0])
    
        if not df_params[param_keys[4]].equals(df_params_prev[param_keys[4]]):
            objectives_val = '-'

    if objectives_val != '-':
        for id, val in enumerate(params):
            params[id][param_keys[4]] = objectives_val

    #switches = [return_toggle_switch(id, o) for id, o in enumerate(df_params['objective'])]

    return html.Div([
        #https://dash.plotly.com/datatable/editable
        #https://community.plotly.com/t/resolved-dropdown-options-in-datatable-not-showing/20366
        dcc.Store(id='wizard_state_stored-data', data=data),
        dash_table.DataTable(
            id = 'wizard-parameters-input-parameters-table',
            columns = columns,
            data = params,
            editable = True,
            dropdown={
                param_keys[4]: {
                    'options': [
                        {'label': i, 'value': i}
                        for i in ['min', 'max']
                    ],
                    'clearable': False
                },
             }
        ),
        html.Div("Set all to Min/Max"),
        dcc.Dropdown(['-', 'min', 'max'], objectives_val, id = 'wizard-parameters-input-objectives-dropdown', clearable=False),
    ]), children


@app.callback(Output('wizard-model-output-view', 'children'),
              Input('wizard-model-input-radio-items', 'value'))
def show_view_wizard_model(agg):
    if agg == 'R':
        return html.Div(
                style = {
                    'height': '50px',
                    'width' : '50px',
                    'background-color': '#FF0000'
                    },
                id = "wizard-model-output-view"
            )
    if agg == 'I':
        return html.Div(
                style = {
                    'height': '50px',
                    'width' : '50px',
                    'background-color': '#00FF00'
                    },
                id = "wizard-model-output-view"
            )
    if agg == 'A':
        return html.Div(
                style = {
                    'height': '50px',
                    'width' : '50px',
                    'background-color': '#0000FF'
                    },
                id = "wizard-model-output-view"
            )
    

'''  
#CHECK PARAMETERS

#Approach 1 - use active cell
@app.callback( Output('wizard-parameters-output-warning', 'children'),
              Input('wizard-parameters-input-parameters-table', 'derived_virtual_row_ids'),
              Input('wizard-parameters-input-parameters-table', 'selected_row_ids'),
              Input('wizard-parameters-input-parameters-table', 'active_cell'),
              State('wizard-parameters-input-parameters-table', 'data'))
def update_table_wizard_parameters(row_ids, selected_row_ids, active_cell, data):
    #https://community.plotly.com/t/input-validation-in-data-table/24026

    criteria = list(data[0].keys())
    df = pd.DataFrame.from_dict(data).set_index(criteria[0])
    print(active_cell)

    warning = "Warning"

    if active_cell:
        warning = df.iloc[active_cell['row']][active_cell['column_id']]

    return html.Div([
        warning
    ])
 
'''

'''
#https://dash.plotly.com/dash-daq/toggleswitch
def return_toggle_switch(id, o):
    switch_id = 'switch-' + str(id)
    objective = True if o == 'max' else False
    return html.Div([
        daq.ToggleSwitch(
            id = switch_id,
            value = objective
        )
    ])
'''

#==============================================================
#   PLAYGROUND
#==============================================================

def model_setter():
    pass

def ranking_vizualization():
    #TO DO
    pass

def improvement_actions():
  #TO DO
  pass

def main_dash_layout():
    return html.Div(children=[
        dcc.Tabs(children=[
            dcc.Tab(label='Ranking vizualiazation', children=[
                ranking_vizualization()
            ]),
            dcc.Tab(label='Improvement actions', children=[
                improvement_actions()
            ]),
            dcc.Tab(label='analisis of parameters', children=[
                model_setter()
            ])
        ])
    ])


#==============================================================
#   MAIN
#==============================================================

app.layout = html.Div(children=[
    header(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='data-store', storage_type='memory'),
    dcc.Store(id='params-store', storage_type='memory'),
    footer()
], id="css-layout")


@app.callback(Output('page-content', 'children', allow_duplicate=True),
              Input('url', 'pathname'),
              prevent_initial_call=True)
def display_page(pathname):
    if pathname == '/':
        return home_layout()
    elif pathname == '/wizard':
        return wizard()
    elif pathname == '/main_dash_layout':
        return main_dash_layout()
    else:
        return '404 - Page not found'

if __name__ == "__main__":
    app.run_server(debug=True)