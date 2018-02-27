#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:58:03 2018

@author: matthew_green
"""
import plotly.plotly as py
from plotly.grid_objs import Grid, Column
from plotly.tools import FigureFactory as FF 

import time
import datetime
import numpy as np
import pandas as pd


valid = pd.read_csv('valid_accuracy.csv')

valid = valid[0:145]

table = FF.create_table(valid)
py.plot(table, filename='valid_run_data_table', auto_open=False)

# %%

def to_unix_time(dt):
    epoch =  datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000

valid_value = list(valid['Value'])
my_columns = []
for k in range(len(valid.index) - 1):
    my_columns.append(Column(valid.index[:k + 1], 'x{}'.format(k + 1)))   
    my_columns.append(Column(valid_value[:k + 1], 'y{}'.format(k + 1)))
grid = Grid(my_columns)
py.grid_ops.upload(grid, 'valid_run_data_table' + str(time.time()), auto_open=False)

# %%

data = [dict(
    type='scatter',
    xsrc=grid.get_column_reference('x1'),
    ysrc= grid.get_column_reference('y1'),
    name='Model Accuracy',
    mode='lines',
    line=dict(color= '#FFA14E'),
    fill='tozeroy',
    fillcolor='#BB773A')]

# %%

axis = dict(
    ticklen=4,
    mirror=True,
    zeroline=False,
    showline=True,
    autorange=False,
    showgrid=False)

# %%

layout = dict(
    title='Model Accuracy',
    paper_bgcolor='#000000',
    plot_bgcolor='#2E2E2E',
    font=dict(color='#CCCCCC'),
    titlefont=dict(color='#CCCCCC', font='14'),
    showlegend=False,
    autosize=False,
    width=800,
    height=400,
    xaxis=dict(axis, **{'title': 'Epoch', 'nticks':10, 'range':[0, 140]}),
    yaxis=dict(axis, **{'title': '% Accuracy', 'range':[0, 1]}),
    updatemenus=[dict(
        type='buttons',
        showactivate=False,
        y=1,
        x=1.1,
        xanchor='right',
        yanchor='top',
        pad=dict(t=0, r=10),
        bgcolor = '#000000',
        active = 100,
        bordercolor = '#CCCCCC',
        font = dict(size=11, color='#CCCCCC'),
        buttons=[dict(
            label='Play',
            method='animate',
            args=[
                None,
                dict(
                    frame=dict(duration=50, redraw=False),
                    transition=dict(duration=0),
                    fromcurrent=True,
                    mode='immediate'
                )
            ]
        )]
    )]
)

# %%
            
frames=[{'data':[{'xsrc': grid.get_column_reference('x{}'.format(k + 1)),
                  'ysrc': grid.get_column_reference('y{}'.format(k + 1))}],
         'traces': [0]
        } for k in range(len(valid.index) - 1)]

# %%

fig=dict(data=data, layout=layout, frames=frames)
py.create_animations(fig, 'valid_accuracy' + str(time.time()))
