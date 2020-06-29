# -*- coding: utf-8 -*-
import base64
import io

import copy
import dash
import dash_auth
import dash_table
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import time

import pandas as pd
import numpy as np
import datetime
from utils import *

VALID_USERNAME_PASSWORD_PAIRS = {
    'caravel': 'assessment'
}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server

########## PPG
# production_df = pd.read_csv('data/work_cell_data.csv')
# margin_column = 'Actual vs Planned'
# groupby_primary = 'Batch Close Month'
# groupby_secondary = 'Inventory Org Name'
# descriptors = list(production_df.select_dtypes(exclude=np.number).columns)
##########

########## PPG2
# dates = ['Batch Completion Date',
#  'Move Order Created',
#  'First Inv Pick',
#  'Last Inv Pick',
#  'First Formulated Consumed Material',
#  'Last Formulated Consumed Material',
#  'First QC Consumed Material',
#  'Last QC Consumed Material',
#  'TO.80 Log Date',
#  'Preship And Fill Date',
#  'TO.80 Approval Date',
#  'Min SKU WIP Start Date',
#  'First WIP Completion Transaction',
#  'Last WIP Completion Transaction',f
#  'TO.90 Log Date',
#  'TO.90 Approval Date',
#  'Min Date',
#  'Max Date']
dates = ['Batch Completion Date', 'First Formulated Consumed Material', 'TO.80 Log Date']
production_df = pd.read_csv('data/Cleveland No Timedelta.csv', parse_dates=dates)
# whole_df = pd.read_csv('data/Cleveland.csv', parse_dates=dates)
descriptors = ['Family', 'Tank Number',
       'Cost Center', 'Technology', 'Product', 'Parent Batch Actual Qty', 'Site']
time_components = ['PA Time',
 'Tot. CM Time',
 '80 appv.',
 'Filling Time',
 'Tot. Time']
time_components = production_df.columns[6:11]
# for col in time_components:
#     production_df[col] = pd.to_timedelta(production_df[col])
time_column = time_components[-1]
volume_column = 'Parent Batch Actual Qty'
margin_column = "{} By {}".format(volume_column, time_column)
production_df[margin_column] = production_df[volume_column] /\
    (production_df[time_components[-1]])
groupby_primary = 'Technology'
groupby_secondary = descriptors[2]
production_df = production_df.loc[production_df[margin_column] < 1e2]
global_df = copy.copy(production_df)
##########


# production_df[descriptors] = production_df[descriptors].astype(str)
production_json = production_df.to_json()

def find_opportunity(df,
                     groupby_primary = "Cost Center",
                     groupby_secondary = "Technology",
                     groupby_tertiary = "Tank Number",
                     time_column=time_components[3],
                     volume_column='Parent Batch Actual Qty',
                     quant_target=0.75):
    groups=3
    if type(df.iloc[-1][time_column]) == pd._libs.tslibs.timedeltas.Timedelta:
        df[time_column] = df[time_column]
    margin_column = "{} By {}".format(volume_column, time_column)
    groupby = [groupby_primary, groupby_secondary, groupby_tertiary]
    df.loc[:, margin_column] = df[volume_column] / df[time_column]
    df = df.loc[df[[margin_column, volume_column, time_column]].notnull().all(axis=1)]
    # desc = df.groupby(groupby)[margin_column].quantile([0.5,quant_target]).unstack(level=groups)
    desc = df.groupby(groupby)[margin_column].describe()[['50%', '75%']]
    totals = df.groupby(groupby)[volume_column, time_column].agg(['sum', 'std', 'mean'])
    count = df.groupby(groupby)[volume_column].agg(['count'])
    count.columns = ['Count']
    desc = desc.join(totals).dropna()
    desc = desc.join(count)
    desc['Volume Opportunity, Gal'] = (desc[volume_column, 'sum'] / desc['50%'] * desc['75%']) - desc[volume_column, 'sum']
    desc['Time Opportunity, Hours'] = desc[time_column, 'sum'] - (desc[time_column, 'sum'] / desc['75%'] * desc['50%'])
    desc = desc.sort_values(by=[('Time Opportunity, Hours')], ascending=False)
    return desc

def make_primary_plot(production_df,
                   margin_column,
                   volume_column,
                   groupby_primary,
                   groupby_secondary,
                   time_column,
                   filter_selected=None,
                   filter_category=None,
                   results_df=None,
                   chart_type='Parallel Coordinates (Time)',
                   all_lines=True,
                   sort_by='mean',
                   data_type='Rate (Gal/Hr)',
                   quant=[0.02, 0.997],
                   dist_cutoff=2):
    ### Preprocessing
    if (data_type == 'Rate (Gal/Hr)') and (chart_type != 'Parallel Coordinates (Time)'):
        margin_column = "{} By {} (Gal/Hr)".format(volume_column, time_column)
        production_df[margin_column] = production_df[volume_column] / (production_df[time_column])
    elif data_type == 'Volume (Gal)':
        margin_column = "{} (Gal)".format(volume_column)
        production_df[margin_column] = production_df[volume_column]
    elif chart_type != 'Parallel Coordinates (Time)':
        margin_column = "{} (Hr)".format(time_column)
        production_df[margin_column] = production_df[time_column]
    production_df = production_df.loc[production_df[margin_column] < np.inf]
    production_df = production_df.loc[(production_df[margin_column] <
               production_df[margin_column].quantile(quant[1])) &
              (production_df[margin_column] >
               production_df[margin_column].quantile(quant[0]))]

    ### Charts
    if chart_type == 'Parallel Coordinates (Time)':
        df = production_df.groupby(groupby_primary)[time_components].agg(lambda x: x.median())
        df = df.reset_index()
        for col in time_components:
            df[col] = df[col]
        df[time_components] = np.round(df[time_components]) #px does not round automatically
        df = df.sort_values(by='Tot. Time').reset_index(drop=True)
        dimensions = list([
                    dict(tickvals = list(df['Tot. Time']),
                         ticktext = list(df[groupby_primary]),
                         label = groupby_primary, values = df['Tot. Time']),
                ])
        for col in time_components:
            dimensions.append(dict(label = col, values = df[col]))

        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = df['Tot. Time'],
                           colorscale = 'Electric',
                           showscale = True,
                           cmin = min(df['Tot. Time']),
                           cmax = max(df['Tot. Time'])),
                dimensions = dimensions,

            )
        )
    elif chart_type == 'Scatter':
        groupby = [groupby_primary, groupby_secondary]
        groupby = [i for i in groupby if 'None' not in i]
        if len(groupby) == 0:
            dff = production_df.sort_values(by=margin_column, ascending=False)
            dff = dff.reset_index(drop=True)

            fig = go.Figure()
            for data in px.scatter(
                    dff,
                    x=dff.index,
                    y=margin_column,
                    size=volume_column,
                    opacity=0.6).data:
                fig.add_trace(
                    data)
        else:
            if margin_column == volume_column:
                dff = pd.DataFrame(production_df.groupby(groupby)[[margin_column]]\
                     .median().sort_values(by=margin_column, ascending=False)).reset_index()
            else:
                dff = pd.DataFrame(production_df.groupby(groupby)[[margin_column, volume_column]]\
                     .median().sort_values(by=margin_column, ascending=False)).reset_index()
            dff['median'] = dff.groupby(groupby[-1])[margin_column].\
                    transform('median')

            dff = dff.sort_values(['median', margin_column],
                ascending=False).reset_index(drop=True)
            dff = dff[dff.columns[:-1]]
            if groupby_primary == 'Cost Center':
                dff[groupby_primary] = '_' + dff[groupby_primary]

            fig = go.Figure()
            for data in px.scatter(
                    dff,
                    x=groupby[0],
                    y=margin_column,
                    size=volume_column,
                    color=groupby[-1],
                    hover_data=[groupby[0]],
                    opacity=0.6).data:
                fig.add_trace(
                    data
                ),
    elif chart_type == 'Distribution':
        groupby = [groupby_primary, groupby_secondary]
        groupby = [i for i in groupby if 'None' not in i]
        fig = go.Figure()
        if len(groupby) != 0:
            dff = pd.DataFrame(production_df.groupby(groupby)[[margin_column]]\
                             .median().sort_values(by=margin_column, ascending=False)).reset_index()
            dff['median'] = dff.groupby(groupby[0])[margin_column].\
                    transform('median')

            dff = dff.sort_values(['median', margin_column],
                ascending=False).reset_index(drop=True)
            dff = dff[dff.columns[:-1]]
            for index in dff.index:
                if len(groupby) == 2:
                    trace = production_df.loc[(production_df[groupby_primary] == dff[groupby_primary][index]) &
                             (production_df[groupby_secondary] == dff[groupby_secondary][index])]
                    name = 'N: {}, Avg: {:.0f}, {}, {}'.format(trace.shape[0], dff[margin_column][index],
                        dff[groupby_primary][index], dff[groupby_secondary][index])
                elif len(groupby) == 1:
                    trace = production_df.loc[(production_df[groupby[0]] == dff[groupby[0]][index])]

                    name = 'N: {}, Avg: {:.0f}, {}'.format(trace.shape[0], dff[margin_column][index],
                        dff[groupby[0]][index])

                trace["Site"] = " "
                if trace.shape[0] > dist_cutoff:
                    fig.add_trace(go.Violin(x=trace[margin_column],
                                      y=trace["Site"],
                                      name=name,
                                    side='positive'))
        else:
            trace = production_df
            name = 'N: {}, Avg: {:.0f}'.format(trace.shape[0], trace[margin_column].median())
            if trace.shape[0] > dist_cutoff:
                fig.add_trace(go.Violin(x=trace[margin_column],
                                  y=trace["Site"],
                                  name=name,
                                side='positive'))
        fig.update_traces(meanline_visible=True, orientation='h')
        fig.update_xaxes(rangemode="nonnegative")


    elif "vs" in margin_column:
        margin_column = '{} (% by {}, {})'\
                       .format(margin_column, groupby_primary, groupby_secondary)
        dff = pd.DataFrame(((production_df.groupby([groupby_primary, groupby_secondary])\
                             ['Actual Qty In (KLG)'].sum() -
                         production_df.groupby([groupby_primary, groupby_secondary])\
                             ['Planned Qty In (KLG)'].sum()) /
                         production_df.groupby([groupby_primary, groupby_secondary])\
                            ['Planned Qty In (KLG)'].sum()) * 100).reset_index()
        dff.columns = [groupby_primary, groupby_secondary, margin_column]
        fig = px.bar(dff, dff[groupby_primary],
                 margin_column,
                 color=groupby_secondary,
                 barmode='group')

    fig.layout.clickmode = 'event+select'
    fig.update_layout({
            "height": 600,
            "plot_bgcolor": "#FFFFFF",
            "paper_bgcolor": "#FFFFFF",
    }
    )
    if chart_type != 'Parallel Coordinates (Time)':
        if chart_type != 'Distribution':
            fig.update_layout({
                "title": '{}'.format(margin_column),
                "yaxis.title": "{}".format(margin_column),
                "xaxis.title": "{}".format(groupby_primary),
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
    ),
                "xaxis.tickfont.size": 8,
                })
        else:
            fig.update_layout({
                    "title": '{}'.format(margin_column),
                    "xaxis.title": "{}".format(margin_column),
                    "yaxis.title": "{} + {}".format(groupby_primary,
                        groupby_secondary),
                    "margin": dict(
                           l=0,
                           r=0,
                           b=0,
                           t=30,
                           pad=4
        ),
                    "xaxis.tickfont.size": 8,
                    })
    return fig


def make_secondary_plot(production_df,
                   margin_column,
                   time_column,
                   groupby_primary,
                   groupby_secondary,
                   filter_selected=None,
                   filter_category=None,
                   results_df=None,
                   chart_type='Parallel Coordinates (Time)',
                   data_type='Rate (Gal/Hr)',
                   quant=[0.02, 0.997],
                   dist_cutoff=2,
                   start_date='First Formulated Consumed Material',
                   end_date='TO.80 Log Date'):
    ### Preprocessing
    if (data_type == 'Rate (Gal/Hr)') and (chart_type != 'Parallel Coordinates (Time)'):
        margin_column = "{} By {} (Gal/Hr)".format(volume_column, time_column)
        production_df[margin_column] = production_df[volume_column] / (production_df[time_column])
    elif data_type == 'Volume (Gal)':
        margin_column = "{} (Gal)".format(volume_column)
        production_df[margin_column] = production_df[volume_column]
    elif chart_type != 'Parallel Coordinates (Time)':
        margin_column = "{} (Hr)".format(time_column)
        production_df[margin_column] = production_df[time_column]
    production_df = production_df.loc[production_df[margin_column] < np.inf]
    production_df = production_df.loc[(production_df[margin_column] <
               production_df[margin_column].quantile(quant[1])) &
              (production_df[margin_column] >
               production_df[margin_column].quantile(quant[0]))]
    groupby = [groupby_primary, groupby_secondary]
    groupby = [i for i in groupby if 'None' not in i]
    if len(groupby) == 0:
        groupby = ['Product']
    if chart_type == 'Distribution':
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',\
                  '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        colors_cycle = cycle(colors)

        dff = pd.DataFrame(production_df.groupby(groupby)[[margin_column]]\
                             .median().sort_values(by=margin_column, ascending=False)).reset_index()
        dff['median'] = dff.groupby(groupby[0])[margin_column].\
                transform('median')

        dff = dff.sort_values(['median', margin_column],
            ascending=False).reset_index(drop=True)
        dff = dff[dff.columns[:-1]]
        fig = go.Figure()
        for index in dff.index:
            if len(groupby) == 2:
                trace = production_df.loc[(production_df[groupby_primary] == dff[groupby_primary][index]) &
                         (production_df[groupby_secondary] == dff[groupby_secondary][index])]
                name = 'N: {}, Avg: {:.0f}, {}, {}'.format(trace.shape[0], dff[margin_column][index],
                    dff[groupby_primary][index], dff[groupby_secondary][index])
            elif len(groupby) == 1:
                trace = production_df.loc[(production_df[groupby[0]] == dff[groupby[0]][index])]

                name = 'N: {}, Avg: {:.0f}, {}'.format(trace.shape[0], dff[margin_column][index],
                    dff[groupby[0]][index])

            trace["Site"] = " "
            if trace.shape[0] > dist_cutoff:
                trace = trace.reset_index(drop=True)
                color = next(colors_cycle)
                for sub_index in trace.index:
                    x1 = trace[start_date][sub_index]
                    x2 = trace[end_date][sub_index]
                    y1 = trace[margin_column][sub_index]
                    y2 = trace[margin_column][sub_index]
                    if sub_index == 0:
                        legend = True
                    else:
                        legend = False
                    fig.add_trace(go.Scatter(x=[x1, x2],
                                          y=[y1, y2],
                                          name=name,
                                             showlegend=legend,
                                             legendgroup=name,
                                            marker=dict(size=8, color=[color, color]),
                                             line=dict(color=color),

                                            mode='lines+markers'))
    else:
        production_df = production_df.sort_values(dates[-1]).reset_index()
        fig = px.scatter(production_df.loc[production_df[groupby[0]].dropna().index],
              x=dates[-1], y=volume_column, color=groupby[0])
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "paper_bgcolor": "#FFFFFF",
                # "title": '{} by {}'.format(volume_column,
                #  dates[-1]),
                "yaxis.title": "{}".format(margin_column),
                "height": 400,
                "legend_title_text": " ",
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4),
                "xaxis":{'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 1, 'label': '1Y', 'step': 'year', 'stepmode': 'backward'},
                                                          {'step': 'all'}])}},
                })
    return fig

def make_tertiary_plot(production_df,
                       margin_column,
                       descriptors,
                       clickData=None,
                       toAdd=None,
                       col=None,
                       val=None,
                       subdf=None,
                       family=None,
                       category_filter=None):
    desc = []
    if toAdd is not None:
        for item in toAdd:
            if item not in desc:
                desc.append(item)
    if subdf is not None:
        test = subdf
        title = 'Main Plot Selection'
    else:
        if clickData != None:
            val = clickData["points"][0]['x']
            production_df[descriptors] = production_df[descriptors].astype(str)
        elif col == None:
            col = 'Product'
            val = production_df[col].unique()[0]
        if col in desc:
            desc.remove(col)
        test = production_df.loc[production_df[col] == val]
        title = '{}: {}'.format(col,val)

    test = test.replace(np.nan, 'N/A', regex=True)
    test[descriptors] = test[descriptors].astype(str)

    fig = px.sunburst(test, path=desc, color=margin_column, title='{}: {}'.format(
        col, val), hover_data=desc,
        color_continuous_scale=px.colors.sequential.Viridis,
         )
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                "title": title,
                "paper_bgcolor": "#FFFFFF",
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
    ),
                })
    return fig

def make_results_distribution(df,
                 groupby_primary = "Cost Center",
                 groupby_secondary = "Technology",
                 groupby_tertiary = "Tank Number",
                 time_column=time_components[3],
                 data_type='Rate (Gal/Hr)',
                 volume_column='Parent Batch Actual Qty',
                 quant_target=0.75,
                 dist_cutoff = 1):
    groups = 3
    ### Preprocessing
    if (data_type == 'Rate (Gal/Hr)'):
        margin_column = "{} By {} (Gal/Hr)".format(volume_column, time_column)
        df[margin_column] = df[volume_column] / (df[time_column])
    elif data_type == 'Volume (Gal)':
        margin_column = "{} (Gal)".format(volume_column)
        df[margin_column] = df[volume_column]
    else:
        margin_column = "{} (Hr)".format(time_column)
        df[margin_column] = df[time_column]
    # margin_column = "{} By {} (Gal/Hr)".format(volume_column, time_column)
    groupby = [groupby_primary, groupby_secondary, groupby_tertiary]
    # df.loc[:, margin_column] = df[volume_column] / (df[time_column])
    # df = df.loc[df[[margin_column, volume_column, time_column]].notnull().all(axis=1)]
#     desc = df.groupby(groupby)[margin_column, volume_column, time_column].quantile([0.5,quant_target]).unstack(level=groups)
#     return desc


    dff = pd.DataFrame(df.groupby(groupby)[[margin_column]]\
                 .median().sort_values(by=margin_column, ascending=False)).reset_index()
    dff['median'] = dff.groupby(groupby)[margin_column].\
            transform('median')

    dff = dff.sort_values(['median', margin_column],
        ascending=False).reset_index(drop=True)
    dff = dff[dff.columns[:-1]]
#     return dff
    fig = go.Figure()
    for index in dff.index:
        trace = df.loc[(df[groupby_primary] == dff[groupby_primary][index]) &
                 (df[groupby_secondary] == dff[groupby_secondary][index]) &
                      (df[groupby_tertiary] == dff[groupby_tertiary][index])]
        if trace.shape[0] > dist_cutoff:
            name = 'Avg: {:.0f}, {}, {}, {}'.format(dff[margin_column][index],
                                                dff[groupby_primary][index],
                                                dff[groupby_secondary][index],
                                               dff[groupby_tertiary][index])
            fig.add_trace(go.Violin(x=trace[margin_column],
                              y=trace["Site"],
                              name=name,
                            side='positive'))
    fig.update_traces(meanline_visible=True, orientation='h')
    fig.update_xaxes(rangemode="nonnegative")
    fig.update_layout({
                "title": '{}'.format(margin_column),
                "xaxis.title": "{}".format(margin_column),
                "yaxis.title": "{}, {}, {}".format(groupby[0], groupby[1], groupby[2]),
                "height": 600})
    fig.update_layout({
                "plot_bgcolor": "#FFFFFF",
                # "title": title,
                "paper_bgcolor": "#FFFFFF",
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
    ),
                })
    return fig

UPLOAD = html.Div([
    html.Div([
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '95%',
                'height': '60px',
                # 'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'vertical-align': 'middle',
                'margin': '10px',

                'padding': '5px',
            },
            multiple=False
        ),],className='four columns',
            style={
            'margin-left': '40px',
            },
        id='up-option-1',),
        html.Div([
        html.P(' - or - ',
        style={
               'textAlign': 'center',
               'margin-top': '30px'}
               ),],className='four columns',
                   id='up-option-2',
                             ),
        html.Div([
        dcc.Dropdown(id='preset-files',
                     multi=False,
                     options=[{'label': i, 'value': i} for i in ['Cleveland Filtered', 'Cleveland', 'Oak Creek']],
                     # placeholder="Select Cloud Dataset",
                     className='dcc_control',
                     style={
                            'textAlign': 'center',
                            'width': '95%',
                            'margin': '10px',
                            }
                            ),],className='four columns',
                            id='up-option-3',
                                                        style={
                                                        'margin-right': '40px',
                                                               }
                                                               ),
        ], className='row flex-display',
        style={'max-height': '500px',
                 'margin-top': '200px'}
        ),
    # html.P('Margin Column'),
    # dcc.Dropdown(id='upload-margin',
    #              multi=False,
    #              options=[],
    #              className="dcc_control",
    #              style={'textAlign': 'center',
    #                     'margin-bottom': '10px'}),
    # html.P('Volume Column'),
    # dcc.Dropdown(id='upload-volume',
    #              multi=False,
    #              options=[],
    #              className="dcc_control",
    #              style={'textAlign': 'center',
    #                     'margin-bottom': '10px'}),
    # html.P('Descriptor-Attribute Columns'),
    # dcc.Dropdown(id='upload-descriptors',
    #              multi=True,
    #              options=[],
    #              className="dcc_control",
    #              style={'textAlign': 'left',
    #                     'margin-bottom': '10px'}),
    # html.P('p-Value Limit for Median Test', id='pvalue-number'),
    # dcc.Slider(id='p-value-slider',
    #            min=0.01,
    #            max=1,
    #            step=0.01,
    #            value=0.5),
    # html.Button('Process data file',
    #             id='datafile-button',
    #             style={'textAlign': 'center',
    #                    'margin-bottom': '10px'}),
],)

HIDDEN = html.Div([
    html.Div(id='production-df-upload',
             style={'display': 'none'},
             children=production_json),
    # html.Div(id='stat-df-upload',
    #          style={'display': 'none'},
    #          children=stat_json),
    html.Div(id='descriptors-upload',
             style={'display': 'none'},
             children=descriptors),
    html.Div(id='margin-upload',
             style={'display': 'none'},
             children=margin_column),
    html.Div(id='primary-upload',
             style={'display': 'none'},
             children=groupby_primary),
    html.Div(id='secondary-upload',
             style={'display': 'none'},
             children=groupby_secondary),
    html.Div(id='production-df-holding',
             style={'display': 'none'},
             children=None),
])

ABOUT = html.Div([dcc.Markdown('''

###### This dashboard evaluates Work Cell correlation with Planned vs Actual production ######

**KPIs:**

--KPI Description--

**Charts:**

--Primary, Secondary, Tertiary Chart Description--

**Controls:**

--Controls Tab Description--

Visualization Tab:

--Analytics Tab Description--

''')],style={'margin-top': '20px',
             'max-height': '500px',
             'overflow': 'scroll'})

search_bar = html.A(
    dbc.Row(
    [
        # dbc.Col(html.Img(src='assets/mfg_logo.png', height="40px")),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
),
href='https://mfganalytic.com/',
className="ml-auto flex-nowrap mt-3 mt-md-0",
)

NAVBAR = dbc.Navbar(
    [ html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src='assets/dashboard_logo.png', height="50px")),
                ],
                align="center",
                no_gutters=True,
            ),
            href='http://caravelsolutions.com/',
            ),
        dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="light",
    dark=False,
    sticky='top',
)

VISUALIZATION = html.Div([
    html.P('Filter'),
    dcc.Dropdown(id='filter_dropdown_1',
                 options=[{'label': i, 'value': i} for i in
                            descriptors],
                 value='Cost Center',
                 multi=False,
                 className="dcc_control"),
    dcc.Dropdown(id='filter_dropdown_2',
                 options=[{'label': i, 'value': i} for i in
                            production_df['Cost Center'].unique()],
                 value=production_df['Cost Center'].unique(),
                 multi=True,
                 className="dcc_control"),
    html.P('Groupby Primary'),
    dcc.Dropdown(id='primary_dropdown',
                 options=[{'label': i, 'value': i} for i in
                           descriptors + ['None']],
                 value='Technology',
                 multi=False,
                 className="dcc_control"),
    html.P('Groupby Secondary'),
    dcc.Dropdown(id='secondary_dropdown',
                 options=[{'label': i, 'value': i} for i in
                           descriptors + ['None']],
                 value='Tank Number',
                 multi=False,
                 className="dcc_control"),
    html.P('Process Times'),
    dcc.Dropdown(id='time_dropdown',
                 options=[{'label': i, 'value': i} for i in
                           time_components],
                 value=time_components[-1],
                 multi=False,
                 className="dcc_control"),
    # html.P('Quantile Range'),
    # dcc.RangeSlider(
    #     id='quantile-slider',
    #     min=0,
    #     max=1,
    #     step=0.05,
    #     value=[.1, .9]
    # ),
    html.P('Graph Type'),
    dcc.Dropdown(id='distribution',
                 options=[{'label': i, 'value': i} for i in
                           ['Scatter', 'Distribution', 'Parallel Coordinates (Time)']],
                 value='Distribution',
                 className="dcc_control"),
    html.P('Plot Metric'),
    dcc.Dropdown(id='data-type',
                 options=[{'label': i, 'value': i} for i in
                           ['Rate (Gal/Hr)', 'Volume (Gal)', 'Time (Hr)']],
                 value='Rate (Gal/Hr)',
                 className="dcc_control"),
      ],style={'max-height': '500px',
               'margin-top': '20px',
               'overflow': 'scroll'}
)

KPIS = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.H5(id='kpi-1'), html.H6(margin_column, id='margin-label')
            ], id='kpi1', className='six columns', style={'margin': '10px'}
            ),
            html.Div([
                html.Img(src='assets/money_icon_1.png', width='80px'),
            ], id='icon1', className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-rev',),
    html.Div([
        html.Div([
            html.Div([
                html.H5(id='kpi-2'), html.H6('Batches', id='margin-label2')
            ], className='six columns', style={'margin': '10px'}, id='kpi2',
            ),
            html.Div([
                html.Img(src='assets/product_icon_3.png', width='80px'),
            ], className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-rev-percent',),
    html.Div([
        html.Div([
            html.Div([
                html.H5(id='kpi-3'), html.H6('Volume', id='margin-label3')
            ], className='six columns', style={'margin': '10px'}, id='kpi3',
            ),
            html.Div([
                html.Img(src='assets/volume_icon_3.png', width='80px'),
            ], className='five columns',
                style={
                    'textAlign': 'right',
                    'margin-top': '20px',
                    'margin-right': '20px',
                    'vertical-align': 'text-bottom',
                }),
            ], className='row flex-display',
            ),
        ], className='mini_container',
           id='margin-products',),
    ], className='row container-display',
)

ANALYTICS = html.Div([
html.P('Groupby Primary'),
dcc.Dropdown(id='primary_dropdown_analytics',
             options=[{'label': i, 'value': i} for i in
                        descriptors],
             value='Cost Center',
             multi=False,
             className="dcc_control"),
html.P('Groupby Secondary'),
dcc.Dropdown(id='secondary_dropdown_analytics',
             options=[{'label': i, 'value': i} for i in
                       descriptors],
             value='Tank Number',
             multi=False,
             className="dcc_control"),
html.P('Groupby Tertiary'),
dcc.Dropdown(id='tertiary_dropdown_analytics',
             options=[{'label': i, 'value': i} for i in
                       descriptors],
             value='Product',
             multi=False,
             className="dcc_control"),
html.P('Process Times'),
dcc.Dropdown(id='time_dropdown_analytics',
             options=[{'label': i, 'value': i} for i in
                       time_components],
             value=time_components[1],
             multi=False,
             className="dcc_control"),
html.Button('Find opportunity',
            id='opportunity-button',
            style={'textAlign': 'center',
                   'margin-bottom': '10px'}),
html.P('Plot Metric'),
dcc.Dropdown(id='data-type-analytics',
             options=[{'label': i, 'value': i} for i in
                       ['Rate (Gal/Hr)', 'Volume (Gal)', 'Time (Hr)']],
             value='Rate (Gal/Hr)',
             className="dcc_control"),

    ], style={'max-height': '500px',
             'margin-top': '20px'}

    )

app.layout = html.Div([NAVBAR,
html.Div(className='pretty_container', children=[KPIS,
    html.Div([
        html.Div([
        dcc.Tabs(id='tabs-control', value='tab-2', children=[
            dcc.Tab(label='Upload', value='tab-4',
                    children=[UPLOAD]),
            dcc.Tab(label='Analytics', value='tab-2',
                    children=[ANALYTICS]),
            dcc.Tab(label='Visualization', value='tab-1',
                    children=[VISUALIZATION]),
            dcc.Tab(label='About', value='tab-3',
                    children=[ABOUT]),
                    ]),
            ], className='mini_container',
               id='descriptorBlock',
            ),
        html.Div([
            dcc.Graph(id='primary_plot',
                      # figure=make_primary_plot(production_df,
                      #   margin_column, volume_column, groupby_primary,
                      #   groupby_secondary, time_column)
                        ),
            ], className='mini_container',
               id='ebit-family-block',
               style={'display': 'block'},
            ),
    ], className='row container-display',
    ),
    html.Div([
        html.Div([
            dcc.Graph(className='inside_container',
                        id='secondary_plot',
                        style={'display': 'none'},
                        figure=make_secondary_plot(production_df,
                                           margin_column,
                                           time_column,
                                           groupby_primary,
                                           groupby_secondary,
                                           chart_type='Distribution')
                        ),
            html.Div([
            dcc.Loading(
                id="loading-1",
                type="default",
                children=dash_table.DataTable(id='opportunity-table',
                                    editable=True,
                                    # filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    column_selectable="single",
                                    row_selectable="multi",
                                    row_deletable=True,
                                    selected_columns=[],
                                    selected_rows=[0, 1, 2],
                                    page_action="native",
                                    page_current= 0,
                                    page_size= 10,),),
                    ],
                    id='opportunity-table-block',
                    style={'overflow': 'scroll'}),
            ], className='mini_container',
               id='violin',
               style={'display': 'block'},
                ),
        # html.Div([
        #     dcc.Dropdown(id='length_width_dropdown',
        #                 options=[{'label': i, 'value': i} for i in
        #                            descriptors],
        #                 value=descriptors[:-3],
        #                 multi=True,
        #                 placeholder="Include in sunburst chart...",
        #                 className="dcc_control"),
        #     dcc.Graph(
        #                 id='tertiary_plot',
        #                 figure=make_tertiary_plot(production_df, margin_column,
        #                                  descriptors, toAdd=descriptors[:-3])
        #                 ),
        #         ], className='mini_container',
        #            id='sunburst',
        #         ),
            ], className='row container-display',
               style={'margin-bottom': '10px'},
            ),
    ],
    ), HIDDEN,
    html.Div([], id='clickdump'),
    # html.Pre(id='relayout-data'),
],
)
app.config.suppress_callback_exceptions = True

# @app.callback(
#     Output('relayout-data', 'children'),
#     [Input('primary_plot', 'relayoutData')])
# def display_relayout_data(relayoutData):
#     return json.dumps(relayoutData, indent=2)

#### Tab control stuff
### UPLOAD TOOL ###
@app.callback(
    [Output('production-df-upload', 'children'),],
  [Input('upload-data', 'contents'),
   Input('preset-files', 'value')],
  [State('upload-data', 'filename'),
   State('upload-data', 'last_modified')])
def update_production_df_and_table(list_of_contents, preset_file, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = [parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        df = df[0]
        columns = [{'label': i, 'value': i} for i in df.columns]
        columns_table = [{"name": i, "id": i} for i in df.columns]
        return columns, columns, df.to_json(), columns
    elif preset_file is not None:

        dates = ['Batch Completion Date', 'First Formulated Consumed Material', 'TO.80 Log Date']
        # if preset_file == 'Cleveland':
        #     production_df = whole_df
        # else:
        production_df = pd.read_csv('data/{}.csv'.format(preset_file), parse_dates=dates)
        descriptors = ['Batch Completion Date', 'Batch Number', 'Tank Number',
               'Cost Center', 'Technology', 'Product', 'Inventory Category',
               'Equalization Lot Number', 'Parent Batch Planned Qty',
               'Parent Batch Actual Qty', 'Family']
        time_components = ['PA Time',
         'Tot. CM Time',
         '80 appv.',
         'Filling Time',
         'Tot. Time']
        time_components = production_df.columns[6:11]
        # for col in time_components:
        #     production_df[col] = pd.to_timedelta(production_df[col])
        time_column = time_components[-1]
        volume_column = 'Parent Batch Actual Qty'
        margin_column = "{} By {}".format(volume_column, time_column)
        production_df[margin_column] = production_df[volume_column] /\
            (production_df[time_components[-1]])
        groupby_primary = 'Technology'
        groupby_secondary = descriptors[2]
        # df = pd.read_csv('data/{}.csv'.format(preset_file))
        columns = [{'label': i, 'value': i} for i in production_df.columns]
        columns_table = [{"name": i, "id": i} for i in production_df.columns]

        return [production_df.to_json()]

# @app.callback(
#     [Output('production-df-upload', 'children'),
#     # Output('stat-df-upload', 'children'),
#     # Output('descriptors-upload', 'children'),
#     # Output('metric-upload', 'children'),
#     # Output('volume-upload', 'children'),
#     ],
#    [Input('production-df-holding', 'children'),
#     Input('upload-margin', 'value'),
#     Input('upload-descriptors', 'value'),
#     Input('datafile-button', 'n_clicks'),
#     Input('upload-volume', 'value'),
#     Input('p-value-slider', 'value')]
# )
# def update_main_dataframe(holding_df, margin, descriptors, button, volume, pvalue):
#     ctx = dash.callback_context
#     if ctx.triggered[0]['prop_id'] == 'datafile-button.n_clicks':
#         production_df = pd.read_json(holding_df)
#         for desc in descriptors: #9 is arbitrary should be a fraction of total datapoints or something
#             if (len(production_df[desc].unique()) > 9) and (production_df[desc].dtype == float):
#                 production_df[desc] = np.round(production_df[desc].astype(float),1)
#         stat_df = my_median_test(production_df,
#                    metric=margin,
#                    descriptors=descriptors,
#                    stat_cut_off=pvalue,
#                    continuous=False)
#         production_df[descriptors] = production_df[descriptors].astype(str)
#         production_df = production_df.sort_values(['Product Family', margin],
#                                                   ascending=False)
#         return production_df.to_json()

@app.callback(
    [Output('opportunity-table', 'data'),
    Output('opportunity-table', 'columns'),
    Output('opportunity-table', 'filter_action'),
    Output('opportunity-table', 'selected_rows')],
    [Input('opportunity-button', 'n_clicks'),
    Input('production-df-upload', 'children'),
    Input('primary_dropdown_analytics', 'value'),
    Input('secondary_dropdown_analytics', 'value'),
    Input('tertiary_dropdown_analytics', 'value'),
    Input('time_dropdown_analytics', 'value'),
    Input('tabs-control', 'value'),]
)
def display_opportunity_results(button, production_json, one, two, three, time, tab):
    production_df = global_df
    ctx = dash.callback_context

    if (ctx.triggered[0]['prop_id'] == 'opportunity-button.n_clicks') or\
       (ctx.triggered[0]['prop_id'] == 'tabs-control.value'):
        # production_df = pd.read_json(production_df)
        # for col in time_components:
        #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
            # production_df[col] = production_df[col]
        results = find_opportunity(production_df, one, two, three, time).reset_index()
        results.columns = [str(x).strip().replace('(', '').replace(')', '').replace("'", '') for x in results.columns]
        results = np.round(results)
        results = results[[i for i in results.columns if ('75%' not in i) and ('50%' not in i)]]
        metric = results['Time Opportunity, Hours']
        results['Rating'] = results['Time Opportunity, Hours'].apply(lambda x:
            '⭐⭐⭐⭐' if x > metric.quantile(0.99) else (
            '⭐⭐⭐' if x > metric.quantile(0.75) else (
            '⭐⭐' if x > metric.quantile(0.5) else (
            '⭐' if x > metric.quantile(0.25) else ''
        ))))
        columns=[{"name": i, "id": i} for i in results.columns]
        data = results.to_dict('rows')
        rows = [i for i in results.index if results['Time Opportunity, Hours'][i] > metric.quantile(0.99)]

        return data, columns, 'native', rows


@app.callback(
    [Output('secondary_plot', 'style'),
     Output('opportunity-table-block', 'style'),],
    [Input('tabs-control', 'value'),]
)
def display_violin_plot(tab):
    if (tab == 'tab-1') | (tab == 'tab-3') | (tab == 'tab-4'):
            return {'display': 'block',
             'margin': '10px',
             'padding': '15px',
             'position': 'relative',
             'border-radius': '5px',
             'width': '95%'}, {'display': 'none'}
    elif tab == 'tab-2':
            return {'display': 'none'}, \
            {'max-height': '500px',
               'overflow': 'scroll',
               'display': 'block',
               'padding': '0px 20px 20px 20px'}

@app.callback(
    [Output('kpi-1', 'children'),
     Output('kpi-2', 'children'),
     Output('kpi-3', 'children')],
    [Input('filter_dropdown_1', 'value'),
    Input('filter_dropdown_2', 'value'),
    Input('opportunity-table', 'derived_virtual_selected_rows'),
    Input('opportunity-table', 'derived_virtual_data'),
    Input('tabs-control', 'value'),
    Input('production-df-upload', 'children'),
    Input('margin-upload', 'children'),
    Input('primary_dropdown', 'value'),
    Input('secondary_dropdown', 'value'),
    Input('secondary_plot', 'clickData'),
    Input('primary_plot', 'selectedData'),
    Input('secondary_plot', 'relayoutData'),
    Input('time_dropdown', 'value'),
    Input('time_dropdown_analytics', 'value'),
    ]
)
def display_opportunity(filter_category, filter_selected, rows, data, tab,
                        production_json, margin_column, groupby_primary,
                        groupby_secondary, clickData, selectedData,
                        relayoutData, time_column, time):
    production_df = global_df
    ctx = dash.callback_context
    print(ctx.triggered[0]['prop_id'])
    if (tab == 'tab-2') and (data is not None) and (len(rows) > 0):
        # production_df = pd.read_json(production_df, convert_dates=dates)
#         total_volume = pd.DataFrame(data)['Parent Batch Actual Qty, sum'].sum()/1e6
        total_volume = production_df['Parent Batch Actual Qty'].sum()/1e6
        extra_volume = pd.DataFrame(data).iloc[rows]['Volume Opportunity, Gal'].sum()/1e6
        volume_increase = (total_volume + extra_volume) / total_volume * 100

#         total_batches = pd.DataFrame(data)['Count'].sum()
        total_batches = production_df.shape[0]
        mean_batch_size = pd.DataFrame(data).iloc[rows]['Parent Batch Actual Qty, mean'].mean()/1e6
        extra_batches = extra_volume / mean_batch_size
        batch_increase = (total_batches + extra_batches) / total_batches * 100

        rate = pd.DataFrame(data)['Parent Batch Actual Qty, sum'].sum() /\
            pd.DataFrame(data)['{}, sum'.format(time)].sum()
#         production_df[time] = pd.to_timedelta(production_df[time], unit='ms')
#         rate = production_df['Parent Batch Actual Qty'].sum() /\
#             (production_df[time])
        new_rate = (pd.DataFrame(data)['Parent Batch Actual Qty, sum'].sum() + (extra_volume*1e6) )/ pd.DataFrame(data)['{}, sum'.format(time)].sum()
        rate_increase = new_rate / rate * 100
        return "{:.1f} M Gal / Hr".format(new_rate), \
        "+ {:.0f} Batches ({:.2f}%)".format(extra_batches, batch_increase), \
        "+ {:.2f} M Gal ({:.2f}%)".format(extra_volume, volume_increase)
    if type(filter_selected) == str:
        filter_selected = [filter_selected]
    # production_df = pd.read_json(production_df, convert_dates=dates)
    production_df = production_df.loc[production_df[filter_category].isin(
        filter_selected)]
    # for col in time_components:
    #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
    production_df[margin_column] = production_df[volume_column] /\
        (production_df[time_column])
    # production_df = production_df.loc[production_df[margin_column] < np.inf]
    production_df = production_df.loc[(production_df[margin_column] <
        production_df[margin_column].quantile(0.995))]
    if relayoutData is not None:
        if 'xaxis.range[0]' in relayoutData.keys():
            start = pd.to_datetime(relayoutData['xaxis.range[0]'])
            end = pd.to_datetime(relayoutData['xaxis.range[1]'])
            production_df = production_df.loc[(production_df[dates[-1]] < end) &
                                              (production_df[dates[-1]] > start)]
    old_kpi_2 = production_df.shape[0]
    old_kpi_1 = production_df[margin_column].mean()
    old_kpi_3 = production_df[volume_column].sum()
    # old_kpi_1 = (production_df['Actual Qty In (KLG)'].sum() -
    #              production_df['Planned Qty In (KLG)'].sum()) * 5
    # old_kpi_3 = production_df['Actual Qty In (KLG)'].sum()
    return "{:.2f} Avg. Gal/Hr".format(old_kpi_1), \
    "{}".format(old_kpi_2),\
    "{:.2f} M Gal".format(old_kpi_3/1e6)

@app.callback(
    [Output('filter_dropdown_2', 'options'),
     Output('filter_dropdown_2', 'value'),
     Output('filter_dropdown_2', 'multi'),],
    [Input('filter_dropdown_1', 'value'),
     Input('distribution', 'value'),
     Input('production-df-upload', 'children'),]
)
def update_filter(category, type, production_json):
    production_df = global_df
    # production_df = pd.read_json(production_df)
    if type == 'Distribution':
        if len(production_df[category].unique()) > 1:
            return [{'label': i, 'value': i} for i in production_df[category].unique()],\
            production_df[category].unique()[-2], False
        else:
            return [{'label': i, 'value': i} for i in production_df[category].unique()],\
            production_df[category].unique()[0], False
    else:
        return [{'label': i, 'value': i} for i in production_df[category].unique()],\
        list(production_df[category].unique()), True

# @app.callback(
#      Output('filter_dropdown_2', 'multi'),
#     [Input('distribution', 'value')]
# )
# def update_filter(type):
#     if type == 'Distribution (Volume)':
#         return False
#     else:
#         return True

@app.callback(
    Output('margin-upload', 'children'),
    [Input('time_dropdown', 'value'),
    Input('time_dropdown_analytics', 'value'),
    Input('tabs-control', 'value'),]
)
def margin_column(time_column, time, tab):
    if tab == 'tab-2':
        return "{} By {}".format(volume_column, time)
    else:
        return "{} By {}".format(volume_column, time_column)

@app.callback(
    Output('margin-label', 'children'),
    [Input('time_dropdown', 'value'),
    Input('time_dropdown_analytics', 'value'),
    Input('tabs-control', 'value'),]
)
def margin_column(time_column, time, tab):
    if tab == 'tab-2':
        return "{} By {}".format(volume_column, time)
    else:
        return "{} By {}".format(volume_column, time_column)

### FIGURES ###
@app.callback(
    Output('primary_plot', 'figure'),
    [Input('filter_dropdown_1', 'value'),
    Input('filter_dropdown_2', 'value'),
    Input('opportunity-table', 'derived_virtual_selected_rows'),
    Input('opportunity-table', 'derived_virtual_data'),
    Input('tabs-control', 'value'),
    Input('production-df-upload', 'children'),
    Input('margin-upload', 'children'),
    Input('primary_dropdown', 'value'),
    Input('secondary_dropdown', 'value'),
    Input('secondary_plot', 'relayoutData'),
    Input('time_dropdown', 'value'),
    Input('distribution', 'value'),
    Input('data-type', 'value'),
    Input('primary_dropdown_analytics', 'value'),
    Input('secondary_dropdown_analytics', 'value'),
    Input('tertiary_dropdown_analytics', 'value'),
    Input('time_dropdown_analytics', 'value'),
    Input('data-type-analytics', 'value'),
    ]
)
def display_primary_plot(filter_category, filter_selected, rows, data, tab,
                        production_json, margin_column, groupby_primary,
                        groupby_secondary, relayoutData, time_column,
                        chart_type, data_type, one, two, three, time,
                        data_type_analytics):


    # production_df = pd.read_json(production_df, convert_dates=dates)
    ctx = dash.callback_context

    if (tab == 'tab-2') and (data is not None) and (len(rows) > 0):
        production_df = global_df
        margin_column = "{} By {}".format(volume_column, time_column)
        # for col in time_components:
        #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
        production_df[margin_column] = production_df[volume_column] /\
            (production_df[time_column])
        # for col in time_components:
        #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
        groupby_primary = one
        groupby_secondary = two
        groupby_tertiary = three
        opp_data = pd.DataFrame(data)
        groups = opp_data.iloc[rows]
        sub_df = pd.DataFrame()
        for row in rows:
            group_one = opp_data[groupby_primary][row]
            group_two = opp_data[groupby_secondary][row]
            group_three = opp_data[groupby_tertiary][row]
            sub_df = pd.concat([sub_df,
            production_df.loc[(production_df[groupby_primary] == group_one) &
                              (production_df[groupby_secondary] == group_two) &
                              (production_df[groupby_tertiary] == group_three)]])

        return make_results_distribution(sub_df, one, two, three, time,
            data_type_analytics)
    # elif (tab == 'tab-2'):
    #     return None


    elif (tab != 'tab-2') &\
        (ctx.triggered[0]['prop_id'] != 'opportunity-table.derived_virtual_data'):
        production_df = global_df
        margin_column = "{} By {}".format(volume_column, time_column)
        if type(filter_selected) == str:
            filter_selected = [filter_selected]

        production_df = production_df.loc[production_df[filter_category].isin(
            filter_selected)]
        # for col in time_components:
        #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
        production_df[margin_column] = production_df[volume_column] /\
            (production_df[time_column])
        production_df = production_df.loc[production_df[margin_column] < np.inf]
        production_df = production_df.loc[(production_df[margin_column] <
            production_df[margin_column].quantile(0.997))]
        print('prim filt ', production_df.shape)

        if relayoutData is not None:
            if 'xaxis.range[0]' in relayoutData.keys():
                start = pd.to_datetime(relayoutData['xaxis.range[0]'])
                end = pd.to_datetime(relayoutData['xaxis.range[1]'])
                production_df = production_df.loc[(production_df[dates[-1]] < end) &
                                                  (production_df[dates[-1]] > start)]
            return make_primary_plot(production_df,
              margin_column, volume_column, groupby_primary,
              groupby_secondary, time_column, chart_type=chart_type,
              data_type=data_type)

        return make_primary_plot(production_df,
          margin_column, volume_column, groupby_primary,
          groupby_secondary, time_column, chart_type=chart_type,
          data_type=data_type)
    # else:
    #     raise PreventUpdate

@app.callback(
    Output('secondary_plot', 'figure'),
    [Input('filter_dropdown_1', 'value'),
    Input('filter_dropdown_2', 'value'),
    Input('opportunity-table', 'derived_viewport_selected_rows'),
    Input('opportunity-table', 'data'),
    Input('tabs-control', 'value'),
    Input('production-df-upload', 'children'),
    Input('margin-upload', 'children'),
    Input('primary_dropdown', 'value'),
    Input('secondary_dropdown', 'value'),
    Input('time_dropdown', 'value'),
    Input('distribution', 'value'),
    Input('data-type', 'value'),
    ]
)
def display_secondary_plot(filter_category, filter_selected, rows, data, tab,
                        production_json, margin_column, groupby_primary,
                        groupby_secondary, time_column, chart_type, data_type):
    # ctx = dash.callback_context
    #
    # if (ctx.triggered[0]['prop_id'] == 'render-button.n_clicks') or\
    #         (ctx.triggered[0]['prop_id'] == 'margin-upload.children'):
    production_df = global_df
    # production_df = pd.read_json(production_df, convert_dates=dates)
    margin_column = "{} By {}".format(volume_column, time_column)
    if type(filter_selected) == str:
        filter_selected = [filter_selected]

    production_df = production_df.loc[production_df[filter_category].isin(
        filter_selected)]
    # for col in time_components:
    #     production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
    production_df[margin_column] = production_df[volume_column] /\
        (production_df[time_column])
    production_df = production_df.loc[production_df[margin_column] < np.inf]
    production_df = production_df.loc[(production_df[margin_column] <
        production_df[margin_column].quantile(0.997))]
    if tab == 'tab-2':
        production_df = production_df.iloc[:20]
    print('secon filt ', production_df.shape)
    return make_secondary_plot(production_df,
        margin_column, time_column, groupby_primary,
        groupby_secondary, chart_type=chart_type, data_type=data_type)

# @app.callback(
#     Output('tertiary_plot', 'figure'),
#     [Input('filter_dropdown_1', 'value'),
#     Input('filter_dropdown_2', 'value'),
#     Input('opportunity-table', 'derived_viewport_selected_rows'),
#     Input('opportunity-table', 'data'),
#     Input('tabs-control', 'value'),
#     Input('production-df-upload', 'children'),
#     Input('margin-upload', 'children'),
#     Input('primary_dropdown', 'value'),
#     Input('secondary_dropdown', 'value'),
#     Input('secondary_plot', 'clickData'),
#     Input('primary_plot', 'selectedData'),
#     Input('length_width_dropdown', 'value'),
#     Input('descriptors-upload', 'children'),
#     Input('secondary_plot', 'relayoutData'),
#     Input('time_dropdown', 'value')
#     ]
# )
# def display_tertiary_plot(filter_category, filter_selected, rows, data, tab,
#                         production_df, margin_column, groupby_primary,
#                         groupby_secondary, clickData, selectedData,
#                          toAdd, descriptors, relayoutData, time_column):
#
#     production_df = pd.read_json(production_df, convert_dates=dates)
#     production_df = production_df.loc[production_df[filter_category].isin(
#         filter_selected)]
#     for col in time_components:
#         production_df[col] = pd.to_timedelta(production_df[col], unit='ms')
#     production_df[margin_column] = production_df[volume_column] /\
#         (production_df[time_column])
#     production_df = production_df.loc[production_df[margin_column] < np.inf]
#     production_df = production_df.loc[(production_df[margin_column] <
#         production_df[margin_column].quantile(0.995))]
#     ctx = dash.callback_context
#     if ctx.triggered[0]['prop_id'] == 'primary_plot.selectedData':
#         dff = pd.DataFrame(selectedData['points'])
#         # dfff = pd.DataFrame(production_df[groupby_secondary].unique())
#         subdf = production_df.loc[(production_df[groupby_primary].isin(dff['x']))]# &
#                 # (production_df[groupby_secondary].isin(dfff.iloc
#                 # [dfff.index.isin(dff['curveNumber'])][0].values))]
#
#         return make_tertiary_plot(production_df, margin_column, descriptors,
#             toAdd=toAdd,
#             subdf=subdf)
#
#     col = groupby_primary
#     val = production_df[col].unique()[0]
#
#     return make_tertiary_plot(production_df, margin_column, descriptors,
#         clickData=clickData, toAdd=toAdd, col=col, val=val)

if __name__ == "__main__":
    app.run_server(debug=True)
