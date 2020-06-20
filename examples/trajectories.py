#%%
import os
import sys
import numpy as np
from numpy import pi
from cuprates_transport.bandstructure import BandStructure, Pocket
from cuprates_transport.admr import ADMR
from cuprates_transport.conductivity import Conductivity
from cuprates_transport.utils import easy_args, args_name

import plotly.graph_objs as go
import dash
import dash_core_components as dcc

import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px

try:
    switch = sys.argv[1]  ## get the first  the script
except IndexError:
    switch = "LargePocket"

band_switch = None
# band_switch = "yamaji"
# band_switch = "af"

tau_switch = None
tau_switch = "long"
# tau_swhitch = "medium"

default_params = {
    "bandname": "LargePocket",
    "t": 190,
    "tp": -0.154,
    "tpp": 0.074,
    "tz": 0.076,
    "tz2": 0.00,
    "tz3": 0.00,
    "tz4": 0.,
    "mu": -0.930,
    "fixdoping": 0.1,
    "numberOfKz": 31,
    "mesh_ds": 1/30,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0, 15, 30, 45],
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

if band_switch in ["hp", "hpocket", "hole", "af", "AF"]:
    default_params.update({
        "bandname": "hPocket",
        "mu": -0.495,
        "M": 0.025,
        "Bphi_array": [0, 45],
        "gamma_0": 24.2,
        "gamma_k": 0,
    })
elif band_switch in ["free", "yamaji"]:
    default_params.update({
        "bandname": "yamaji",
        "tp": -0.154,
        "tpp": 0.074,
        "tz": 0.,
        "tz4": 0.05,
        "free_term": 1.0,
        "mu": -0.7,
        "numberOfKz": 31,
        "mesh_ds": 1/100,
        "Bphi_array": [0],
    })

if tau_switch in ["long", "full"]:
    default_params.update({
        "gamma_0": 0.1,
        "gamma_k": 0,
        "N_time": 1000,
        "Btheta_step": 1,
    }) 
elif tau_switch in ["medium", "intermediate"]:
    default_params.update({
        "gamma_0": 1,
        "gamma_k": 0,
        "N_time": 500,
    }) 


params = easy_args(default_params, out_dict=True)

# pklname = args_name(params, default_params, ["pocket"]) + ".pkl"
# if os.path.exists(pklname):
#     print(f"loading results from {pklname}")
#     with open(pklname, 'rb') as f:
#         bandObject, condObject, admrObject = pickle.load(f)
#     print("done")
# else:

if switch in ["hp", "hpocket", "hole", "af", "AF"]:
    bandObject = Pocket(**params)
    bandObject.half_FS = False
    bandObject.runBandStructure()
else:
    bandObject = BandStructure(**params)
    bandObject.half_FS = False
    bandObject.runBandStructure()
# bandObject.figMultipleFS2D()
condObject = Conductivity(bandObject, **params)
admrObject = ADMR([condObject], **params)
admrObject.runADMR()


# print(f"saving results in {pklname}")
# with open(pklname, 'wb') as f:
#     pickle.dump((bandObject, condObject, admrObject), f)
# print("done")
    
fs_name = params["bandname"]
vv_dict = {}
vvrel_dict = {}
vvdos_dict = {}
vvreldos_dict = {}
gamma_dict = {}
dos_dict = {}

vv_dict_max = 0
vvrel_dict_max = 0
vvdos_dict_max = 0
vvreldos_dict_max = 0
gamma_dict_max = 0
dos_dict_max = 0

for i_phi, phi, in enumerate(admrObject.Bphi_array):
    for i_theta, theta, in enumerate(admrObject.Btheta_array):
        theta_i = admrObject.Btheta_array[i_theta]

        vv_dict[phi, theta] = (admrObject.vproductDict[fs_name, phi, theta_i])
        vvrel_dict[phi, theta] = (admrObject.vproductDict[fs_name, phi, theta_i] - admrObject.vproductDict[fs_name, phi, 0])
        vvdos_dict[phi, theta] = (admrObject.vproductDict[fs_name, phi, theta_i]*bandObject.dos_k)
        vvreldos_dict[phi, theta] = (admrObject.vproductDict[fs_name, phi, theta_i] - admrObject.vproductDict[fs_name, phi, 0])*bandObject.dos_k

        tmp_max = max(vv_dict[phi, theta])
        vv_dict_max = max(vv_dict_max, tmp_max)
        tmp_max = max(vvrel_dict[phi, theta])
        vvrel_dict_max = max(vvrel_dict_max, tmp_max)
        tmp_max = max(vvdos_dict[phi, theta])
        vvdos_dict_max = max(vvdos_dict_max, tmp_max)
        tmp_max = max(vvreldos_dict[phi, theta])
        vvreldos_dict_max = max(vvreldos_dict_max, tmp_max)
 
vz_on_kf = bandObject.vf[2,:]/np.max(bandObject.vf[2,:])
vz_on_kf_max = max(abs(np.min(vz_on_kf)),abs(np.max(vz_on_kf)))

gamma_on_kf = condObject.tau_total_func(
    bandObject.kf[0],
    bandObject.kf[1],
    bandObject.kf[2],
    bandObject.vf[0],
    bandObject.vf[1],
    bandObject.vf[2],
)
gamma_on_kf_max = max(abs(np.min(gamma_on_kf)),abs(np.max(gamma_on_kf)))

dos_on_kf = bandObject.dos_k
dos_on_kf_max = max(abs(np.min(dos_on_kf)),abs(np.max(dos_on_kf)))


color_labels = {  # copy-pasted from https://www.unicodeit.net
    0:"v",
    1:"vv̅",
    2:"vv̅ρ",
    3:"vv̅-(vv̅)₀",
    4:"vv̅ρ-(vv̅ρ)₀",
    5:"τ",
    6:"ρ",
    # ajouter v^2 tau (champ nul)
    # ajouter vv̅ρ-(vv̅ρ)₀ pour phi
}


def fermiSurfacePlot(theta=0, phi=0, view_selected=0):  
    print(theta)

    colormaps = [
        [vz_on_kf, px.colors.diverging.balance, -vz_on_kf_max, vz_on_kf_max],
        [vv_dict[phi, theta], px.colors.diverging.Portland, -vv_dict_max, vv_dict_max],
        [vvdos_dict[phi, theta], px.colors.diverging.Portland, -vvdos_dict_max, vvdos_dict_max],
        [vvrel_dict[phi, theta], px.colors.diverging.Portland, -vvrel_dict_max, vvrel_dict_max],
        [vvreldos_dict[phi, theta], px.colors.diverging.Portland, -vvreldos_dict_max, vvreldos_dict_max],
        [gamma_on_kf, px.colors.diverging.balance, -gamma_on_kf_max, gamma_on_kf_max],
        [dos_on_kf, px.colors.diverging.balance, -dos_on_kf_max, dos_on_kf_max],
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
                # title=dict(text="v<sub>z</sub>", font=dict(family="Arial", size=20),),
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


def FSwithTrajectories(kstart=0, theta=0, phi=0, nk=1, nk_jump=1, view_selected=0):
    kft = admrObject.kftDict[fs_name, phi, theta]
    return [fermiSurfacePlot(theta, phi, view_selected)] + [
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
                'range': [-(5/4.)*pi/bandObject.a, (5/4.)*pi/bandObject.a]
            },
            'yaxis': {
                'title': 'ky',
                'range': [-(5/4.)*pi/bandObject.a, (5/4.)*pi/bandObject.a]
            },
            'zaxis': {
                'title': 'kz',
                'range': [-(10/4.)*pi/bandObject.c, (10/4.)*pi/bandObject.c]
            }
        }
    }
}


def amroGraphData(highlight_idx, clicked_phi):
    return [
        {
            'type': 'scatter',
            'x': 1.*admrObject.Btheta_array,
            'y': admrObject.rzz_array[phi],
            'marker': {
                'size': 4,
            },
            'showlegend':False,
        } for phi in range(len(admrObject.Bphi_array))] + [
        {
            'type': 'scatter',
            'x': 1.*admrObject.Btheta_array[highlight_idx:highlight_idx+1],  # has to be an array
            'y': admrObject.rzz_array[clicked_phi,highlight_idx:highlight_idx+1],
            'mode': 'markers',
            'marker': {
                'size': 12,
            },
            'showlegend':False,
        }
    ]

amroGraph = {
    'data': amroGraphData(0,0),
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
        'hovermode':"closest",
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
                    max=len(color_labels)-1,
                    value=0,
                    marks=color_labels,
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

    clicked_theta = get_theta(clickOn2DGraph) * params["Btheta_step"]
    clicked_phi = params["Bphi_array"][get_phi(clickOn2DGraph)]

    print(clicked_theta)
    updatedGraph = graph3D
    updatedGraph['data'] = FSwithTrajectories(
        kstart=clicked_point,
        theta=clicked_theta,
        phi=clicked_phi,
        view_selected=viewSlider
    )

    if relayoutData and 'scene.camera' in relayoutData:
        updatedGraph['layout']['scene']['camera']= relayoutData['scene.camera']
    return updatedGraph


@app.callback(Output('amroGraph', 'figure'), [Input('amroGraph', 'clickData')])
def update_2Dgraph(clickOn2DGraph):

    clicked_theta = get_theta(clickOn2DGraph)
    clicked_phi = get_phi(clickOn2DGraph)
        
    updatedGraph = amroGraph
    updatedGraph['data'] = amroGraphData(clicked_theta, clicked_phi)
    return updatedGraph

def get_theta(clickOn2DGraph):
    try: return clickOn2DGraph['points'][0]['pointNumber']
    except TypeError: return 0

def get_phi(clickOn2DGraph):
    try: return clickOn2DGraph['points'][0]['curveNumber']
    except TypeError: return 0


if __name__ == '__main__':
    os.system("open http://127.0.0.1:8050/")
    app.run_server()
