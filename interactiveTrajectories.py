import numpy as np
from numpy import sqrt, exp, log, pi, ones
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")

from numba import jit, prange, config, threading_layer, guvectorize, float64
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
from skimage import measure
from copy import deepcopy

from band import BandStructure
from chambers import amroPoint

band = BandStructure()
band.mesh_ds = 1
band.numberOfKz = 15
band.half_FS_z = False
band.discretize_FS()

currentPoint = amroPoint(band, 45, pi/8, pi/16)
currentPoint.solveMovementFunc()

mesh_xy_graph = 40
mesh_z_graph = 40
kx_a = np.linspace(-5*pi/4/band.a, 5*pi/4/band.a, mesh_xy_graph)
ky_a = np.linspace(-5*pi/4/band.b, 5*pi/4/band.b, mesh_xy_graph)
kz_a = np.linspace(-2*pi/band.c, 2*pi/band.c, mesh_z_graph) # 2*pi/c because bodycentered unit cell
kxx, kyy, kzz = np.meshgrid(kx_a, ky_a, kz_a)

bands = band.e_3D_func(kxx, kyy, kzz)
vertices, simplices = measure.marching_cubes_classic(bands, 0)
x = (vertices[:,0]/(mesh_xy_graph-1)-0.5)*(5/2.)*pi/band.a
y = (vertices[:,1]/(mesh_xy_graph-1)-0.5)*(5/2.)*pi/band.b
z = (vertices[:,2]/(mesh_z_graph-1)-0.5)*4*pi/band.c

def vzFunction(kx,ky,kz):
    return abs(band.v_3D_func(kx, ky, kz)[2])

colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
fermiSurface = ff.create_trisurf(x=x, y=y, z=z, 
                        simplices=simplices,
                        plot_edges=False,
                        color_func = vzFunction
                        )

xTraj = currentPoint.kft[0,0,:]
yTraj = currentPoint.kft[1,0,:]
zTraj = currentPoint.kft[2,0,:]
trajectoryData = go.Scatter3d(
    x=xTraj, y=yTraj, z=zTraj,
    mode='lines',
    line=dict(
        color='#ff0000',
        width=10
    )
)

xFS = band.kf[0,:]
yFS = band.kf[1,:]
zFS = band.kf[2,:]
fsData = go.Scatter3d(
    x=xFS, y=yFS, z=zFS,
    mode='markers',
    marker=dict(
        color='#0000ff',
        size=3
    ),
    hoverinfo='closest'
)

trace = go.Scatter(
    x = np.arange(currentPoint.vz_product().shape[0]),
    y = currentPoint.vz_product(),
    mode = 'lines'
)

trace2 = go.Scatter(
    x = np.arange(2),
    y = currentPoint.vz_product()[:2],
    mode = 'markers'
)

# fermiSurface['data'][0].update(opacity=0.90)

dataToPlot = [fsData, trajectoryData] #fermiSurface.data[0], 

# py.iplot(dataToPlot,filename="myFirstPlot")



import json
from textwrap import dedent as d
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Div(children=[
            html.Div([
            html.H1('AMRO'),
            html.Div([
                dcc.Markdown(d("""
                    **Click Data**
                    Click on points.
                    """)),
                html.Pre(id='click-data'),
                ]),
            ]),
            dcc.Graph(
            id='2Dgraph',
            figure={
                'data': trace2
                }
            )
        ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'middle'}),
        html.Div(children=[
            dcc.Graph(
                id='3Dgraph',
                figure={
                    'data': dataToPlot,
                    'layout': {
                        'height': 700,
                        'margin':{'l':0,'r':0,'t':0,'b':0},
                        'scene': {'aspectmode':'cube'},
                        'clickmode':'none'
                        }
                    }
                )
        ], style={'width': '49%', 'height': '150%', 'display': 'inline-block', 'vertical-align': 'middle'})
    ]),
    html.Div([
        
    ], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

activePoint = 22
activeKvec = np.array([0,0,0])
lastClicked2D = None
def updateActivePoint(click2D,click3D):
    global activePoint
    global lastClicked2D
    global activeKvec
    point2D = None
    point3D = None
    if click2D :
        point2D = click2D['points'][0]['pointNumber']
    if click3D and click3D['points'][0]['curveNumber']==0:
        point3D = click3D['points'][0]['pointNumber']
    
    if point2D and point3D:
        if lastClicked2D == point2D:
            activePoint = point3D
            activeKvec = np.array([click3D['points'][0]['x'],click3D['points'][0]['y'],click3D['points'][0]['z']])
        else :
            activePoint = point2D
            activeKvec = np.array([click2D['points'][0]['x'],click3D['points'][0]['y'],click3D['points'][0]['z']])
            lastClicked2D = point2D
    elif point2D:
        activePoint = point2D
        activeKvec = np.array([click2D['points'][0]['x'],click3D['points'][0]['y'],click3D['points'][0]['z']])
        lastClicked2D = point2D
    elif point3D and click3D['points'][0]['curveNumber']==0:
        activePoint = point3D
        activeKvec = np.array([click3D['points'][0]['x'],click3D['points'][0]['y'],click3D['points'][0]['z']])
    
@app.callback(Output('3Dgraph', 'figure'),
              [Input('2Dgraph', 'clickData'),Input('3Dgraph', 'clickData')],
              [State('3Dgraph', 'relayoutData')])
def update_3Dgraph(click2D,click3D,relayoutData):
    global activePoint
    
    xTraj = currentPoint.kft[0,activePoint,:]
    yTraj = currentPoint.kft[1,activePoint,:]
    zTraj = currentPoint.kft[2,activePoint,:]

    # for i in range(xTraj.size):
    #     while xTraj[i] >   pi  : xTraj[i] = xTraj[i]-2*pi
    #     while xTraj[i] <= -pi  : xTraj[i] = xTraj[i]+2*pi
    #     while yTraj[i] >   pi  : yTraj[i] = yTraj[i]-2*pi
    #     while yTraj[i] <= -pi  : yTraj[i] = yTraj[i]+2*pi
    #     while zTraj[i] >  2*pi : zTraj[i] = zTraj[i]-4*pi
    #     while zTraj[i] <=-2*pi : zTraj[i] = zTraj[i]+4*pi

    trajectoryData = go.Scatter3d(
        x=xTraj, y=yTraj, z=zTraj,
        mode='lines',
        line=dict(
            color='#ff0000',
            width=10
        )
    )
    dataToPlot = [fsData, trajectoryData, fermiSurface.data[0]] 
    
    if relayoutData and 'scene.camera' in relayoutData:
        return {'data': dataToPlot,
                'layout': {
                    'margin':{'l':0,'r':0,'t':0,'b':0},
                    'scene': {
                        'camera': relayoutData['scene.camera'] ,
                        'aspectmode': 'cube'
                        },
                    'hoverinfo':'closest'
                    }
                }
    else:
        return {'data': dataToPlot,
                'layout': {
                    'margin':{'l':0,'r':0,'t':0,'b':0},
                    'scene': {
                        'aspectmode': 'cube'
                        },
                    'hoverinfo':'closest'
                    }
                }

@app.callback(Output('2Dgraph', 'figure'),
              [Input('2Dgraph', 'clickData'),Input('3Dgraph', 'clickData')],
              [State('2Dgraph', 'relayoutData')])
def update_2Dgraph(click2D,click3D,relayoutData):
    global activePoint
    
    newTrace2 = go.Scatter(
            x = np.array([activePoint]),
            y = np.array([currentPoint.vz_product()[activePoint]]),
            mode = 'markers'
        )
    layout={
        'margin':{'l':30,'r':10},
        'hovermode':'closest'
        }
    if relayoutData:
        if 'xaxis.range[0]' in relayoutData:
            layout['xaxis'] = {'range': [relayoutData['xaxis.range[0]'],relayoutData['xaxis.range[1]']]}
        if 'yaxis.range[0]' in relayoutData:
            layout['yaxis'] = {'range': [relayoutData['yaxis.range[0]'],relayoutData['yaxis.range[1]']]}
            

    return {'data': [trace, newTrace2], 'layout': layout}
    
@app.callback(Output('click-data', 'children'),
                [Input('2Dgraph', 'clickData'),Input('3Dgraph', 'clickData')],
                [State('2Dgraph', 'relayoutData')])
def display_click_data(click2D,click3D,relayoutData):
    global activePoint
    updateActivePoint(click2D,click3D)
    clicked = {'active point': activePoint}
    # if click2D : clicked["2D"] = click2D
    # if click3D : clicked["3D"] = click3D
    return json.dumps(click3D, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)