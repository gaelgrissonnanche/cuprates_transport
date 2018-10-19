import numpy as np
from numpy import sqrt, exp, log, pi, ones
np.set_printoptions(6,suppress=True,sign="+",floatmode="fixed")
from numba import jit, prange, config, threading_layer, guvectorize, float64
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from band_structure import *
from diff_equation import *
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from scipy.spatial import Delaunay
from copy import deepcopy


start_total_time = time.time()

## Constant //////
# hbar = 1.05e-34
# e = 1.6e-19
# m0 = 9.1e-31
# kB = 1.38e-23
# c = 13.2
# a = 5.3 / sqrt(2)
# b = 5.3 / sqrt(2)
# mu = 805 # VHs = 600
# t = 525
# tp = -115
# tpp = 35
# tz = 11
# tau = 1e-3
# t   =  1.
# tp  = -0.209 * t
# tpp =  0.062 * t
# tz  =  0.0209 * t
# mu  = 1.123 * t

e = 1
hbar = 1
m = 1
a = 1
b = 1
c = 1
t   =  1.
tp  = -0.14 * t
tpp =  0.07 * t
tz  =  0.07 * t
mu  = 0.9 * t # van Hove 0.84
band_parameters = np.array([a, b, c, mu, t, tp, tpp, tz])
tau =  25 / t * hbar
B_amp = 0.02 * t
half_FS_z = False
mesh_xy = 56 # 28 must be a multiple of 4
mesh_z = 30 # 11 ideal to be fast and accurate
mesh_B_theta = 31
B_theta_max = 180
B_phi_a = np.array([0, 15, 30, 45]) * pi / 180
B_theta_a = np.linspace(0, B_theta_max * pi / 180, mesh_B_theta)
## Make mesh_xy a multiple of 4 to respect the 4-order symmetry
mesh_xy = mesh_xy - (mesh_xy % 4)



## Discretize FS
start_time_FS = time.time()
kf, vf, dkf, number_contours = discretize_FS(band_parameters, mesh_xy, mesh_z, half_FS_z)
print("Discretize FS time : %.6s seconds" % (time.time() - start_time_FS))


## Solve differential equation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

@jit(nopython = True, cache = True)
def solve_movement_func(B_amp, B_theta, B_phi, kf, band_parameters, tmax):

    dt = tmax / 300
    t = np.arange(0, tmax, dt)

    ## Compute B ////#
    B = B_func(B_amp, B_theta, B_phi)

    ## Run solver ///#
    kft = rgk4_algorithm(kf, t, B, band_parameters)
    vft = np.empty_like(kft, dtype = np.float64)
    vft[0,:,:], vft[1,:,:], vft[2,:,:] = v_3D_func(kft[0,:,:], kft[1,:,:], kft[2,:,:], band_parameters)

    return kft, vft, t

@jit(nopython = True, parallel = True)
def calculate_vz_product(vf, vzft, kf, dkf, t, tau):

    prefactor = e**2 / ( 4 * pi**3 )

    ## Velocity components
    vxf = vf[0,:]
    vyf = vf[1,:]
    vzf = vf[2,:]

    # Time increment
    dt = t[1] - t[0]
    # Density of State
    dos = hbar * sqrt( vxf**2 + vyf**2 + vzf**2 )

    # First the integral over time
    vz_product = np.empty(vzf.shape[0], dtype = np.float64)
    for i0 in prange(vzf.shape[0]):
        vz_sum_over_t = np.sum( ( 1 / dos[i0] ) * vzft[i0,:] * exp(- t / tau) * dt ) # integral over t
        vz_product[i0] = vzf[i0] * vz_sum_over_t # integral over z

    return vz_product




# mesh_graph = 1001
# kx = np.linspace(-pi/a, pi/a, mesh_graph)
# ky = np.linspace(-pi/b, pi/b, mesh_graph)
# kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')


### go interactive and do
### exec(open("interactiveTrajectories.py").read())


kft, vft, t = solve_movement_func(0.08, pi/2, 0, kf, band_parameters, tmax = 10 * tau)
tmax = 10 * tau
vz_product = calculate_vz_product(vf, vft[2,:,:], kf, dkf, t, tau)

kft = np.reshape(kft,(3,mesh_z,mesh_xy,300))
kft0 = np.reshape(kft[:,:,0,:],(3,mesh_z,1,300))
kft = np.append(kft,kft0,axis=2)
kft = np.transpose(kft,(0,2,1,3))
kft = np.reshape(kft,(3,mesh_z*(mesh_xy+1),300))

vz_product = np.reshape(vz_product,(mesh_z,mesh_xy))
vz_product0 = np.reshape(vz_product[:,0],(mesh_z,1))
vz_product = np.append(vz_product,vz_product0,axis=1)
vz_product = np.transpose(vz_product,(1,0))
vz_product = np.reshape(vz_product,mesh_z*(mesh_xy+1))

x = kft[0,:,0]
y = kft[1,:,0]
z = kft[2,:,0]

## make simplices (triangles of defining the surface)
v = np.arange(0, mesh_xy +1)
u = np.arange(0, mesh_z)
u,v = np.meshgrid(u,v)
u = u.flatten()
v = v.flatten()
points2D = np.vstack([u,v]).T
tri = Delaunay(points2D)
simplices = tri.simplices

def vzFunction(kx,ky,kz):
    return abs(v_3D_func(kx,ky,kz,band_parameters)[2])

colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
fermiSurface = ff.create_trisurf(x=x, y=y, z=z, 
                        simplices=simplices,
                        plot_edges=False,
                        color_func = vzFunction,
                        title="Isosurface")

xTraj = kft[0,0,:]
yTraj = kft[1,0,:]
zTraj = kft[2,0,:]
trajectoryData = go.Scatter3d(
    x=xTraj, y=yTraj, z=zTraj,
    mode='lines',
    line=dict(
        color='#ff0000',
        width=10
    )
)

trace = go.Scatter(
    x = np.arange(vz_product.shape[0]),
    y = vz_product,
    mode = 'lines'
)

trace2 = go.Scatter(
    x = np.arange(2),
    y = vz_product[:2],
    mode = 'markers'
)

# fermiSurface['data'][0].update(opacity=0.75)

dataToPlot = [fermiSurface.data[0], trajectoryData]

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
                        'scene': {'aspectmode':'cube'}
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

activePoint = 743
lastClicked2D = None
def updateActivePoint(click2D,click3D):
    global activePoint
    global lastClicked2D
    point2D = None
    point3D = None
    if click2D :
        point2D = click2D['points'][0]['pointNumber']
    if click3D and 'i' in click3D['points'][0]:
        point3D = click3D['points'][0]['pointNumber']
    
    if point2D and point3D:
        if lastClicked2D == point2D:
            activePoint = point3D
        else :
            activePoint = point2D
            lastClicked2D = point2D
    elif point2D:
        activePoint = point2D
        lastClicked2D = point2D
    elif point3D:
        activePoint = point3D
    
@app.callback(Output('3Dgraph', 'figure'),
              [Input('2Dgraph', 'clickData'),Input('3Dgraph', 'clickData')],
              [State('3Dgraph', 'relayoutData')])
def update_3Dgraph(click2D,click3D,relayoutData):
    global activePoint
    
    xTraj = kft[0,activePoint,:]
    yTraj = kft[1,activePoint,:]
    zTraj = kft[2,activePoint,:]

    for i in range(xTraj.size):
        print(zTraj[i])
        while xTraj[i] >   pi  : xTraj[i] = xTraj[i]-2*pi
        while xTraj[i] <= -pi  : xTraj[i] = xTraj[i]+2*pi
        while yTraj[i] >   pi  : yTraj[i] = yTraj[i]-2*pi
        while yTraj[i] <= -pi  : yTraj[i] = yTraj[i]+2*pi
        while zTraj[i] >  2*pi : zTraj[i] = zTraj[i]-4*pi
        while zTraj[i] <=-2*pi : zTraj[i] = zTraj[i]+4*pi

    trajectoryData = go.Scatter3d(
        x=xTraj, y=yTraj, z=zTraj,
        mode='lines',
        line=dict(
            color='#ff0000',
            width=10
        )
    )
    dataToPlot = [fermiSurface.data[0], trajectoryData]
    
    if relayoutData and 'scene.camera' in relayoutData:
        return {'data': dataToPlot,
                'layout': {
                    'margin':{'l':0,'r':0,'t':0,'b':0},
                    'scene': {
                        'camera': relayoutData['scene.camera'] ,
                        'aspectmode': 'cube'
                        }
                    }
                }
    else:
        return {'data': dataToPlot,
                'layout': {
                    'margin':{'l':0,'r':0,'t':0,'b':0},
                    'scene': {
                        'aspectmode': 'cube'
                        }
                    }
                }

@app.callback(Output('2Dgraph', 'figure'),
              [Input('2Dgraph', 'clickData'),Input('3Dgraph', 'clickData')],
              [State('2Dgraph', 'relayoutData')])
def update_2Dgraph(click2D,click3D,relayoutData):
    global activePoint
    
    newTrace2 = go.Scatter(
            x = np.array([activePoint]),
            y = np.array([vz_product[activePoint]]),
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
    return json.dumps(clicked, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)