#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:01:52 2018

@author: matthew_green
"""
from mylistdir import mylistdir
import flask
import glob
import os
import pandas as pd
from flask import Flask
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

image_directory = '/Users/matthew_green/Desktop/version_control/huz_pig_project/data_pictures_png_augmentation/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

# %%

# Retrive the picture data
df = pd.read_csv('pic_data.csv', low_memory=False)
print(df.shape)

# %%

mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjaXhzYTB0bHcwOHNoMnFtOWZ3YWdreDB3In0.pjROwb9_CEuyKPE-x0lRUw'

# Layout of the main graph
layout_main = dict(
    autosize = True,
    height = 500,
    font = dict(color='#CCCCCC'),
    titlefont = dict(color='#CCCCCC', size='14'),
    margin = dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    
    # OPTIONAL
    hovermode = "closest",

    # COLOR THEME
    plot_bgcolor = "#191A1A",
    paper_bgcolor = "#020202",

    # LEGEND
    legend = dict(font=dict(size=12), orientation='h'),
    title = 'Satellite Overview',
    
    # MAPBOX
    mapbox = dict(
        accesstoken = mapbox_access_token,
        style = "dark",
        center = dict(
            lon = -98.7,
            lat = 28.64,
        ),
        zoom = 7.5,
    )
)

# %%

# Layout of the pie graph
layout_pie = dict(
    autosize = True,
    height = 500,
    font = dict(color='#CCCCCC'),
    titlefont = dict(color='#CCCCCC', size='14'),
    margin = dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode = 'closest',
    plot_bgcolor = '#191A1A',
    paper_bgcolor = '#020202',
    title = 'Mud Finding Breakdown',
    legend = dict(
        font=dict(size=10),
        orientation='h',
        bgcolor='rgba(0,0,0,0)'
    ),
    annotation = [
        dict(
            dict(
                font = {'size': 20},
                showarrow = False,
                text = 'GHG',
                x = 0.20,
                y = 0.5
            )
        )
    ]
)

# %%

# Main graph data type
types = dict(
        corrosive = 'Corrosive',
        no_mud = 'No_Mud',
        paraffin = 'Paraffin'
)

data_main = []
for mud_type, dff in df.groupby('Mud_Type'):
    trace = dict(
            type = 'scattermapbox',
            lon = dff['Long'],
            lat = dff['Lat'],
#            location = dff['Rel_id'],
            location = dff['Pic_Id'],
            text = dff['Num_Pic_Id'],
            name = types[mud_type],
            marker = dict(
                    size = 4,
                    opacity = 0.6,
            )
    )
    data_main.append(trace)
    
# %%

# Pie graph data  
data_pie = [dict(
    values = list(df.Mud_Type.value_counts()),
    labels = list(df.Mud_Type.value_counts().keys()),
    name = 'Mud Finding Breakdown',
    hoverinfo = 'labels+values',
    hole = 0.5,
    type = 'pie',
    marker = dict(colors=['rgb(44, 160, 44)', 'rgb(31, 119, 180)', 'rgb(255, 127, 14)'])
)]

# %%

# Create the app
app = dash.Dash()
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501

# Create app layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    'Pig Classification Dashboard',
                    className='eight columns',
                ),
                html.Img(
                    src="http://chiralityresearch.com/wp-content/uploads/2015/08/logo-small1.png",
                    className='one columns',
                    style={
                        'float': 'right',
                        'position': 'relative',
                    },
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.H5(
                    'No of Wells: {}'.format(len(df)),
                    className='two columns'
                ),
                html.H5(
                    'No Mud: {} | Paraffin: {} | Corrosive: {}'.format(df['Mud_Type'].value_counts()[2], df['Mud_Type'].value_counts()[0], df['Mud_Type'].value_counts()[1]),
                    className='eight columns',
                    style={'text-align': 'center'}
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id='main_graph',
                            figure=dict(
                                data=data_main,
                                layout=layout_main
                            )
                        )
                    ],
                    className='eight columns',
                    style={'margin-top': '20'}
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id='image-dropdown',
                            options=[{'label': i, 'value': i} for i in list_of_images],
                            disabled=True,
                            value=['']
                        ),
                        html.Img(id='image', height=464, width='100%')
                    ],
                    className='four columns',
                    style={'margin-top': '20'}
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Iframe(src=f'https://plot.ly/~mgreen1313/100.embed?autosize=True&link=false&logo=false', height=497, width='100%')
                    ],
                    className='eight columns',
                    style={'margin-top': '10'}
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id='pie_graph',
                            figure=dict(
                                data=data_pie,
                                layout=layout_pie
                            )
                        )
                    ],
                    className='four columns',
                    style={'margin-top': '10'}
                ), 
            ],
            className='row'
        ),
    ],
    className='ten columns offset-by-one'
)

# %%

@app.callback(Output('image-dropdown', 'value'),
              [Input('main_graph', 'hoverData')])
def make_individual_figure(pic):    
    return pic['points'][0]['text']

@app.callback(Output('image', 'src'),
              [Input('image-dropdown', 'value')])
def update_image_src(value):
    return static_image_route + value

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)
      
# %%
            
if __name__ == '__main__':
    app.run_server(debug=False, port=8000)