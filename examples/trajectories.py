#%%
import numpy as np
from numpy import pi
from cuprates_transport.bandstructure import BandStructure
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity

import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc

import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px


params = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 1000.0,
    "tp": 0.,
    "tpp": 0.,
    "tz": 0.0,
    "tz2": 0.00,
    "tz3": 0.00,
    "tz4": -0.1,
    "mu": -0.80,
    "fixdoping": 0.1,
    "numberOfKz": 51,
    "mesh_ds": 1/30,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0],
    "gamma_0": 15,
    "gamma_k": 0,
    "N_time": 500,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}

params = {
    "bandname": "LargePocket",
    "a": 3.74767,
    "b": 3.74767,
    "c": 13.2,
    "t": 190,
    "tp": -0.154,
    "tpp": 0.074,
    "tz": 0.076,
    "tz2": 0.00,
    "tz3": 0.00,
    "tz4": 0.,
    "mu": -0.930,
    "fixdoping": 0.1,
    "numberOfKz": 51,
    "mesh_ds": 1/30,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0],
    "gamma_0": 15.1,
    "gamma_k": 84,
    "N_time": 500,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}


bandObject = BandStructure(**params)
# bandObject.figMultipleFS2D()  # will print the figure
bandObject.half_FS = False
bandObject.runBandStructure()

a = bandObject.a
c = bandObject.c
condObject = Conductivity(bandObject, **params)
admrObject = ADMR([condObject], **params)
admrObject.runADMR()


#%%

vv_dict = {}
vvrel_dict = {}
vvdos_dict = {}
vvreldos_dict = {}

vv_dict_max = 0
vvrel_dict_max = 0
vvdos_dict_max = 0
vvreldos_dict_max = 0

for i_theta, theta, in enumerate(admrObject.Btheta_array[:-1]):
    theta_i = admrObject.Btheta_array[i_theta]

    vv_dict[theta] = (admrObject.vproductDict["LargePocket", 0, theta_i])
    vvrel_dict[theta] = (admrObject.vproductDict["LargePocket", 0, theta_i] - admrObject.vproductDict["LargePocket", 0, 0])
    vvdos_dict[theta] = (admrObject.vproductDict["LargePocket", 0, theta_i]/bandObject.dos_k)
    vvreldos_dict[theta] = (admrObject.vproductDict["LargePocket", 0, theta_i] - admrObject.vproductDict["LargePocket", 0, 0])/bandObject.dos_k

    tmp_max = max(vv_dict[theta])
    vv_dict_max = max(vv_dict_max, tmp_max)

    tmp_max = max(vvrel_dict[theta])
    vvrel_dict_max = max(vvrel_dict_max, tmp_max)
    
    tmp_max = max(vvdos_dict[theta])
    vvdos_dict_max = max(vvdos_dict_max, tmp_max)
    
    tmp_max = max(vvreldos_dict[theta])
    vvreldos_dict_max = max(vvreldos_dict_max, tmp_max)
 
vz = bandObject.vf[2,:]/np.max(bandObject.vf[2,:])
minvz = np.min(vz)
maxvz = np.max(vz)


#%%

def fermiSurfacePlot(theta=0, view_selected=0):  
    print(theta)

    colormaps = [
        [vz, px.colors.diverging.balance, minvz, maxvz],
        [vv_dict[theta], px.colors.diverging.Portland, -vv_dict_max, vv_dict_max],
        [vvdos_dict[theta], px.colors.diverging.Portland, -vvdos_dict_max, vvdos_dict_max],
        [vvrel_dict[theta], px.colors.diverging.Portland, -vvrel_dict_max, vvrel_dict_max],
        [vvreldos_dict[theta], px.colors.diverging.Portland, -vvreldos_dict_max, vvreldos_dict_max]
    ]       
    return go.Scatter3d(
        x = bandObject.kf[0],
        y = bandObject.kf[1],
        z = bandObject.kf[2],
        mode='markers',
        marker=dict(
            size=4,
            color= colormaps[view_selected][0],
            colorscale= colormaps[view_selected][1],   # choose a colorscale
            opacity=0.8,
            cmin= colormaps[view_selected][2],
            cmax= colormaps[view_selected][3],
            colorbar=dict(
                title=dict(text="v<sub>z</sub>", font=dict(family="Arial", size=20),),
                titleside="right",
                tickmode="array",
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300,
                dtick=5,
                yanchor="top", y=0.8,
                xanchor="right", x=0.95,
            ),
        ),
        showlegend=False,
        hoverinfo='none',
    )


N_POINTS = 8051


def FSwithTrajectories(kstart=0, theta=0, nk=1, nk_jump=1, view_selected=0):
    kft = admrObject.kftDict["LargePocket", 0, theta]
    return [fermiSurfacePlot(theta, view_selected)] + [
        go.Scatter3d(
            x = kft[0, kstart+ii*nk_jump, :],
            y = kft[1, kstart+ii*nk_jump, :],
            z = kft[2, kstart+ii*nk_jump, :],
            mode='markers',
            marker=dict(
                color='black',
                size=5
            ),
            showlegend=False,
            hoverinfo='none',
        )
        for ii in range(nk)]


graph3D = {
    'data': FSwithTrajectories(),
    'layout': {
        'margin':{'l':0,'r':0,'t':20,'b':0},
        'scene': {
            'aspectmode': 'cube',
            'xaxis': {
                'title': 'kx',
                'range': [-(5/4.)*pi/a, (5/4.)*pi/a]
            },
            'yaxis': {
                'title': 'ky',
                'range': [-(5/4.)*pi/a, (5/4.)*pi/a]
            },
            'zaxis': {
                'title': 'kz',
                'range': [-(10/4.)*pi/c, (10/4.)*pi/c]
            }
        }
    }
}


def amroGraphData(highlight_idx):
    return [
        {
            'type': 'scatter',
            'x': 1.*admrObject.Btheta_array,
            'y': admrObject.rzz_array[0],
            'marker': {
                'size': 4,
            },
        },
        {
            'type': 'scatter',
            'x': 1.*admrObject.Btheta_array[highlight_idx:highlight_idx+1],
            'y': admrObject.rzz_array[0,highlight_idx:highlight_idx+1],
            'mode': 'markers',
            'marker': {
                'size': 12,
            },
        }
    ]

amroGraph = {
    'data': amroGraphData(1),
    'layout': {
        'height': 300,
        'xaxis': {
            'title': 'theta',
            'range': [0,90]
        },
        'yaxis': {
            'title': 'resitivity rel. to theta=0',
            # 'range': [0.99,1.01]
        },
        'margin':{'l':50,'r':10,'t':20,'b':80},
    }
}


app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                dcc.Slider(
                    id='viewSlider',
                    min=0,
                    max=4,
                    value=0,
                    marks={
                        0:'velocity',
                        1:'vv',
                        2:'vv/dos',
                        3:'vv-vv0',
                        4:'(vv-vv0)/dos',
                    },
                    className='center'
                ),
                dcc.Graph(
                    id='amroGraph',
                    figure=amroGraph,
                ),
            ],
            style={
                'width': '49%', 
                'display': 'inline-block', 
                'vertical-align': 'middle'
            }
        ),
        html.Div(
            children=[
                dcc.Graph(
                    id='3Dgraph', 
                    figure = graph3D, 
                    className='center'
                ),
            ],
            style={
                'width': '49%', 
                'height': '175%', 
                'display': 'inline-block', 
                'vertical-align': 'middle', 'horizontal': 'center'
            }
        ),
    ]
)



@app.callback(
    Output('3Dgraph', 'figure'),
    [
        Input('3Dgraph', 'clickData'),
        Input('amroGraph','clickData'),
        Input('viewSlider', 'value'),
    ],
    [
        State('3Dgraph', 'relayoutData'),
    ]
)
def update_3Dgraph(clickOn3DGraph, clickOn2DGraph, viewSlider, relayoutData):
    
    try: clicked_point = clickOn3DGraph['points'][0]['pointNumber']
    except TypeError: clicked_point = 0

    try: clicked_amro = clickOn2DGraph['points'][0]['pointNumber']
    except TypeError: clicked_amro = 0

    print(clicked_amro)
    updatedGraph = graph3D
    updatedGraph['data'] = FSwithTrajectories(
        kstart=clicked_point,
        theta=clicked_amro*5,
        view_selected=viewSlider
    )

    if relayoutData and 'scene.camera' in relayoutData:
        updatedGraph['layout']['scene']['camera']= relayoutData['scene.camera']
    return updatedGraph


@app.callback(Output('amroGraph', 'figure'), [Input('amroGraph', 'clickData')])
def update_2Dgraph(clickOn2DGraph):
    try: clicked_amro = clickOn2DGraph['points'][0]['pointNumber']
    except TypeError: clicked_amro = 0
    updatedGraph = amroGraph
    updatedGraph['data'] = amroGraphData(clicked_amro)
    return updatedGraph


if __name__ == '__main__':
    app.run_server()

