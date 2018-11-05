import numpy as np
from numpy import pi
import json
import time

import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


# from scipy.spatial import Delaunay
from skimage import measure
from textwrap import dedent as d

from band import BandStructure
from chambers import amroPoint

startTime = time.time()
print('discretizing fermi surface')
band = BandStructure()
band.half_FS_z = False
band.discretize_FS()
print("discretizing time : %.6s s\n" % (time.time() - startTime))

def computeAMROpoints(B_amp,B_phi_a,B_theta_a):
    amroPointList = []
    rho_zz_a = np.empty((B_phi_a.shape[0], B_theta_a.shape[0]), dtype = np.float64)
    for i in range(B_phi_a.shape[0]):
        for j in range(B_theta_a.shape[0]):
            currentPoint = amroPoint(band, B_amp, B_theta_a[j], B_phi_a[i])
            currentPoint.solveMovementFunc()
            currentPoint.chambersFunc()
            amroPointList.append(currentPoint)
            
            rho_zz_a[i, j] = 1 / currentPoint.sigma_zz # dim (phi, theta)
    return rho_zz_a,amroPointList

startTime = time.time()
print('computing AMRO curves')
B_theta_a = np.linspace(0, 11*pi/18, 31)
rho_zz_a, amroPointList = computeAMROpoints(45,np.array([0, 15, 30, 45]) * pi / 180, B_theta_a)
rho_zz_0 = rho_zz_a[:,0]
print("AMRO time : %.6s s\n" % (time.time() - startTime))

currentPoint = amroPointList[0]
refPoint = currentPoint

mesh_xy_graph = 61
mesh_z_graph = 61
kx_a = np.linspace(-5*pi/4/band.a, 5*pi/4/band.a, mesh_xy_graph)
ky_a = np.linspace(-5*pi/4/band.b, 5*pi/4/band.b, mesh_xy_graph)
kz_a = np.linspace(-2*pi/band.c, 2*pi/band.c, mesh_z_graph) # 2*pi/c because bodycentered unit cell
kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a)



xFS = band.kf[0,:]
yFS = band.kf[1,:]
zFS = band.kf[2,:]
fsData = go.Scatter3d(
    x = xFS, y = yFS, z = zFS,
    mode='markers',
    marker=dict(
        color='#0000ff',
        size=2
    ),
    hoverinfo='closest'
)

xTraj = currentPoint.kft[0,0,:]
yTraj = currentPoint.kft[1,0,:]
zTraj = currentPoint.kft[2,0,:]
trajectoryData = go.Scatter3d(
    x = xTraj, y = yTraj, z = zTraj,
    mode='lines',
    line=dict(
        color='#ff0000',
        width=10
    )
)

bands = band.e_3D_func(kxx, kyy, kzz)
vertices, simplices = measure.marching_cubes_classic(bands, 0)
x = (vertices[:,0]/(mesh_xy_graph-1)-0.5)*(5/2.)*pi/band.a
y = (vertices[:,1]/(mesh_xy_graph-1)-0.5)*(5/2.)*pi/band.b
z = (vertices[:,2]/(mesh_z_graph-1)-0.5)*4*pi/band.c
def vzFunction(kx,ky,kz):
    return abs(band.v_3D_func(kx, ky, kz)[2])
colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
fermiSurface = ff.create_trisurf(
    x = x, y = y, z = z, 
    simplices = simplices,
    plot_edges = False,
    color_func = vzFunction
)


# could use 'type': 'scatter3d',
graph3D = {
    'data':[
        fsData,
        trajectoryData,
        fermiSurface['data'][0]
        ],
    'layout': {
        'margin':{'l':0,'r':0,'t':20,'b':0},
        'scene': {
            'aspectmode': 'cube',
            'xaxis': {
                'title': 'kx',
                'range': [-(5/4.)*pi/band.a, (5/4.)*pi/band.a]
            },
            'yaxis': {
                'title': 'ky',
                'range': [-(5/4.)*pi/band.b, (5/4.)*pi/band.b]
            },
            'zaxis': {
                'title': 'kz',
                'range': [-2*pi/band.c, 2*pi/band.c]
            }
        },
        'hoverinfo':'closest',
        }
    }

amroGraph = {
    'data': [{
        'x': B_theta_a * 180 / pi,
        'y': rho_zz_a[i,:]/rho_zz_0[i],
        'type': 'scatter',
        'mode': 'lines',
        'name': 'phi='+str(i*15)
        } for i in range(rho_zz_a.shape[0])],
    'layout': {
        'height': 300,
        'xaxis': {
            'title': 'polar angle',
            'range': [0,105]
        },
        'yaxis': {
            'title': 'rho_zz(theta)/rho_zz(0)',
            'range': [0.99,1.01]
        },
        'margin':{'l':50,'r':10,'t':20,'b':80},
        'hovermode':'closest'
    }
}

weightGraph = {
    'data': [{
        'x': np.arange(currentPoint.vz_product().shape[0]),
        'y': currentPoint.vz_product(),
        'type': 'scatter',
        'mode': 'lines'
        },{
        'x': np.arange(2),
        'y': currentPoint.vz_product()[:2],
        'type': 'scatter',
        'mode': 'markers'
        }
    ],
    'layout': {
        'height': 300,
        'xaxis': {
            'title': 'kf number',
            'range': [0,xFS.shape[0]]
        },
        'yaxis': {
            'title': 'v(kf)*vbar(kf)',
        },
        'margin':{'l':50,'r':10,'t':20,'b':80},
        'hovermode':'closest',
        'showlegend':False
    }
}


app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div(children=[
            html.Div([ html.H1('AMRO'),
                dcc.Graph(id='amroGraph', figure = amroGraph),
                dcc.Graph(id='weightGraph', figure = weightGraph)
            ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),
            html.Div(children=[
                html.Button('Toggle surface (to click points)', id='FSbutton'),
                dcc.Graph(id='3Dgraph', figure = graph3D)
            ], style={'width': '49%', 'height': '150%', 'display': 'inline-block', 'vertical-align': 'middle', 'horizontal': 'center'}),
        html.Div([ dcc.Markdown(d("**Click Data**Click on points.")),
            html.Pre(id='click-data')]),
        ])
    ]),
    html.Div([], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


activeWeight = 22
lastClicked2D = None
def updateActivePoint(click2D,click3D):
    global activeWeight
    global lastClicked2D
    point2D = None
    point3D = None
    if click2D :
        point2D = click2D['points'][0]['pointNumber']
    if click3D and click3D['points'][0]['curveNumber']==0:
        point3D = click3D['points'][0]['pointNumber']
    
    if point2D and point3D:
        if lastClicked2D == point2D:
            activeWeight = point3D
        else :
            activeWeight = point2D
            lastClicked2D = point2D
    elif point2D:
        activeWeight = point2D
        lastClicked2D = point2D
    elif point3D and click3D['points'][0]['curveNumber']==0:
        activeWeight = point3D

activeAMRO = 0
lastClickedAMRO = 0
def updateCurrentPoint(clickAmro):
    global activeAMRO
    global lastClickedAMRO
    global currentPoint

    if clickAmro:
        ii = clickAmro['points'][0]['curveNumber']
        jj = clickAmro['points'][0]['pointNumber']
        nn = ii*rho_zz_a.shape[0]+jj
        activeAMRO = nn

    if activeAMRO != lastClickedAMRO:
        lastClickedAMRO = activeAMRO
        currentPoint = amroPointList[activeAMRO]


@app.callback(Output('3Dgraph', 'figure'),
            [Input('amroGraph', 'clickData'), Input('weightGraph', 'clickData'),Input('3Dgraph', 'clickData'),Input('FSbutton', 'n_clicks')],
            [State('3Dgraph', 'relayoutData')])
def update_3Dgraph(clickAmro,click2D,click3D,n_clicks,relayoutData):
    global activeAMRO
    global activeWeight
    updatedGraph = graph3D

    xTraj = currentPoint.kft[0,activeWeight,:]
    yTraj = currentPoint.kft[1,activeWeight,:]
    zTraj = currentPoint.kft[2,activeWeight,:]

    # for i in range(xTraj.size):
    #     while xTraj[i] >   pi/band.a  : xTraj[i] = xTraj[i]-2*pi/band.a
    #     while xTraj[i] <= -pi/band.a  : xTraj[i] = xTraj[i]+2*pi/band.a
    #     while yTraj[i] >   pi/band.b  : yTraj[i] = yTraj[i]-2*pi/band.b
    #     while yTraj[i] <= -pi/band.b  : yTraj[i] = yTraj[i]+2*pi/band.b
    #     while zTraj[i] >  2*pi/band.c : zTraj[i] = zTraj[i]-4*pi/band.c
    #     while zTraj[i] <=-2*pi/band.c : zTraj[i] = zTraj[i]+4*pi/band.c

    trajectoryData = go.Scatter3d(
        x=xTraj, y=yTraj, z=zTraj,
        mode='lines',
        line=dict(
            color='#ff0000',
            width=10
        )
    )
    updatedGraph['data'] = [fsData, trajectoryData, fermiSurface['data'][0]]
    if n_clicks:
        if n_clicks%2: updatedGraph['data'] = [fsData, trajectoryData]
        
    if relayoutData and 'scene.camera' in relayoutData:
        updatedGraph['layout']['scene']['camera']= relayoutData['scene.camera']

    return updatedGraph


@app.callback(Output('weightGraph', 'figure'),
              [Input('amroGraph', 'clickData'), Input('weightGraph', 'clickData'),Input('3Dgraph', 'clickData')],
              [State('weightGraph', 'relayoutData')])
def update_2Dgraph(clickAmro,click2D,click3D,relayoutData):
    global activeWeight

    updatedGraph = weightGraph
    updatedGraph['data'][0] = {
        'x': np.arange(currentPoint.vz_product().shape[0]),
        'y': currentPoint.vz_product() / refPoint.vz_product(),
        'type': 'scatter',
        'mode': 'lines'
        }
    updatedGraph['data'][1] = {
        'x': np.array([activeWeight]),
        'y': np.array([(currentPoint.vz_product() / refPoint.vz_product())[activeWeight]]),
        'type': 'scatter',
        'mode': 'markers'
        }
    if relayoutData:
        if 'xaxis.range[0]' in relayoutData:
            updatedGraph['layout']['xaxis'] = {'range': [relayoutData['xaxis.range[0]'],relayoutData['xaxis.range[1]']]}
        if 'yaxis.range[0]' in relayoutData:
            updatedGraph['layout']['yaxis'] = {'range': [relayoutData['yaxis.range[0]'],relayoutData['yaxis.range[1]']]}
            
    return updatedGraph


@app.callback(Output('click-data', 'children'),
                [Input('amroGraph','clickData'), Input('weightGraph', 'clickData'),Input('3Dgraph', 'clickData')],
                [State('weightGraph', 'relayoutData')])
def display_click_data(clickAmro,click2D,click3D,relayoutData):
    global activeWeight
    global activeAMRO
    updateActivePoint(click2D,click3D)
    updateCurrentPoint(clickAmro)

    return json.dumps(clickAmro, indent=2)


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server()