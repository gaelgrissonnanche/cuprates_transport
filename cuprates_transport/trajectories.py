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
    "t": 190,
    "tp": -0.154,
    "tpp": 0.074,
    "tz": 0.,#0.076,
    "tz2": 0.00,
    "tz3": 0.00,
    "tz4": 0.1,
    "mu": -1.5,#-0.930,
    "fixdoping": 0.1,
    "numberOfKz": 51,
    "mesh_ds": 1/30,
    "T" : 0,
    "Bamp": 45,
    "Btheta_min": 0,
    "Btheta_max": 90,
    "Btheta_step": 5,
    "Bphi_array": [0],
    "gamma_0": 1,#15.1,
    "gamma_k": 0,#84,
    "gamma_dos_max": 0,
    "power": 12,
    "factor_arcs": 1,
    "seed": 72,
    "data_T": 25,
    "data_p": 0.24,
}

bandObject = BandStructure(**params)
bandObject.half_FS = False
bandObject.runBandStructure()
a = bandObject.a
c = bandObject.c
condObject = Conductivity(bandObject, **params)
admrObject = ADMR([condObject], **params)
admrObject.runADMR()

vproductDiffDict = {}
vproductDiff_max_list = []
vproductDiff_min_list = []
vproduct_max_list = []
vproduct_min_list = []

for i_theta, theta, in enumerate(admrObject.Btheta_array[:-1]):
    theta_i = admrObject.Btheta_array[i_theta]
    theta_ip1 = admrObject.Btheta_array[i_theta + 1]
    vproductDiffDict[theta] = (admrObject.vproductDict["LargePocket", 0, theta_i])
                            #    -admrObject.vproductDict["LargePocket", 0, 0])
    vproduct_max_list.append(np.max(admrObject.vproductDict["LargePocket", 0, theta_i]))
    vproduct_min_list.append(np.min(admrObject.vproductDict["LargePocket", 0, theta_i]))
    vproductDiff_max_list.append(np.max(vproductDiffDict[theta]))
    vproductDiff_min_list.append(np.min(vproductDiffDict[theta]))
 
vproduct_max = max(vproduct_max_list)
vproduct_min = min(vproduct_min_list)
vproductDiff_max = max(vproductDiff_max_list)
vproductDiff_min = min(vproductDiff_min_list)

maxboth = max(abs(vproductDiff_max), abs(vproductDiff_min))

vz = bandObject.vf[2,:]/np.max(bandObject.vf[2,:])
minvz = np.min(vz)
maxvz = np.max(vz)


#%%
def fermiSurfacePlot(theta=0, vz_selected=False):
    return go.Scatter3d(
        x = bandObject.kf[0],
        y = bandObject.kf[1],
        z = bandObject.kf[2],
        mode='markers',
        marker=dict(
            size=4,
            color= vz if vz_selected else vproductDiffDict[theta],
            colorscale= px.colors.diverging.balance if vz_selected else px.colors.diverging.Portland ,   # choose a colorscale
            opacity=0.8,
            cmin= minvz if vz_selected else -maxboth,
            cmax= maxvz if vz_selected else maxboth,
            colorbar=dict(
                title=dict(text="v<sub>z</sub>", font=dict(family="Arial", size=20),),
                titleside="right",
                tickmode="array",
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=300,
                dtick=5,
                yanchor="top", y=0.8,
                xanchor="right", x=0.95,
    #                     tickvals=[2, 50, 100],
    #                     ticktext=["Cool", "Mild", "Hot"],
    #                     ticks="outside"
                    ),
            ),
        showlegend=False,
        hoverinfo='none',
    )


N_POINTS = 8051


def FSwithTrajectories(kstart=0, theta=0, nk=1, nk_jump=1, vz_selected=False):
    kft = admrObject.kftDict["LargePocket", 0, theta]
    return [fermiSurfacePlot(theta, vz_selected)] + [
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


amroGraphData = go.Scatter(
    x = 1.*admrObject.Btheta_array,
    y = admrObject.rzz_array[0],
)


amroGraph = {
    'data': [amroGraphData],
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
                html.Button(
                    'Velocity toggle', 
                    id='FSbutton',
                    className='center'
                ),
                dcc.Graph(
                    id='amroGraph',
                    figure=amroGraph,
                ),
                dcc.Slider(
                    id='thetaSlider',
                    min=0,
                    max=90,
                    step=5,
                    value=5,
                    marks={
                        '0':0 , '10':10, '20':20, '30':30, '40':40,
                        '50':50, '60':60, '70':70, '80':80, '90':90
                    },
                    updatemode='drag',
                    className='center'
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


    # style={
    #     'width': '100%',
    #     'height': '800%',
    #     # 'display': 'inline-block',
    #     # 'vertical-align': 'middle',
    #     # 'horizontal': 'center',
    # },
)



@app.callback(
    Output('3Dgraph', 'figure'),
    [
        Input('3Dgraph', 'clickData'),
        Input('thetaSlider', 'value'),
        Input('FSbutton', 'n_clicks') 
    ],
    [State('3Dgraph', 'relayoutData'),]
)
def update_3Dgraph(click3D, thetaSlider, FSclick, relayoutData):
    updatedGraph = graph3D
    
    try: clicked_point = click3D['points'][0]['pointNumber']
    except TypeError: clicked_point = 0

    updatedGraph['data'] = FSwithTrajectories(
        kstart=clicked_point,
        theta=thetaSlider,
        vz_selected=FSclick%2 if FSclick else False
    )
    
    if relayoutData and 'scene.camera' in relayoutData:
        updatedGraph['layout']['scene']['camera']= relayoutData['scene.camera']
    return updatedGraph

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server()

