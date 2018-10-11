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
half_FS_z = True
mesh_xy = 56 # 28 must be a multiple of 4
mesh_z = 11 # 11 ideal to be fast and accurate
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

kft, vft, t = solve_movement_func(B_amp, 0.15, 0, kf, band_parameters, tmax = 10 * tau)

mesh_graph = 1001
kx = np.linspace(-pi/a, pi/a, mesh_graph)
ky = np.linspace(-pi/b, pi/b, mesh_graph)
kxx, kyy = np.meshgrid(kx, ky, indexing = 'ij')



### go interactive and do
### exec(open("interactiveTrajectories.py").read())

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np
from scipy.spatial import Delaunay

## to connect the first xy point with the last
## also rebuild the two halves in z of the FS, although there is a probleme at the middle
## and it doesn't work at the vHS
FSnodes = kft[:,:,0]

x = kft[0,:,0]
# xStackedInZ = np.reshape(x,(mesh_z,mesh_xy))
# x0stackedInZ = np.reshape(xStackedInZ[:,0],(mesh_z,1))
# newXstackedInZ = np.hstack([xStackedInZ,x0stackedInZ])
# x = np.reshape(newXstackedInZ ,(mesh_xy+1)*mesh_z )
# x = np.append(np.flip(x,0),x)

y = kft[1,:,0]
# yStackedInZ = np.reshape(y,(mesh_z,mesh_xy))
# y0stackedInZ = np.reshape(yStackedInZ[:,0],(mesh_z,1))
# newystackedInZ = np.hstack([yStackedInZ,y0stackedInZ])
# y = np.reshape(newystackedInZ ,(mesh_xy+1)*mesh_z )
# y = np.append(np.flip(y,0),y)

z = kft[2,:,0]
# zStackedInZ = np.reshape(z,(mesh_z,mesh_xy))
# z0stackedInZ = np.reshape(zStackedInZ[:,0],(mesh_z,1))
# newzstackedInZ = np.hstack([zStackedInZ,z0stackedInZ])
# z = np.reshape(newzstackedInZ ,(mesh_xy+1)*mesh_z )
# z = np.append(-np.flip(z,0),z)

## simplices
u = np.arange(0, mesh_xy)# +1)
v = np.arange(0, mesh_z)
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

dataToPlot = [fermiSurface.data[0],fermiSurface.data[1], trajectoryData]

# py.iplot(dataToPlot,filename="myFirstPlot")



import json
from textwrap import dedent as d

import dash
import dash_core_components as dcc
import dash_html_components as html

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1('AMRO'),
        html.Div([
            dcc.Markdown(d("""
                **Click Data**
                Click on points.
                """)),
            html.Pre(id='click-data'),
            ])
    ],className="four columns"),
    html.Div(children=[
        dcc.Graph(
            id='mainGraph',
            figure={
                'data': dataToPlot,
                'layout': {
                    'title': 'Trajectories on the Fermi surface'
                    }
                },
            style={'height': '700px','width': '600px'}
            )
    ],className="four columns")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


@app.callback(dash.dependencies.Output('mainGraph', 'figure'),
              [dash.dependencies.Input('mainGraph', 'clickData')])

def update_graph(clickData):
    if clickData and clickData['points'][0]['i']:
        
        ii = clickData['points'][0]['pointNumber']%((mesh_xy)*mesh_z)
        xTraj = kft[0,ii,:]
        yTraj = kft[1,ii,:]
        zTraj = kft[2,ii,:]
        trajectoryData = go.Scatter3d(
            x=xTraj, y=yTraj, z=zTraj,
            mode='lines',
            line=dict(
                color='#ff0000',
                width=10
            )
        )
        dataToPlot = [fermiSurface.data[0],fermiSurface.data[1], trajectoryData]
    else:
        dataToPlot = [fermiSurface.data[0],fermiSurface.data[1], trajectoryData]

    return {'data': dataToPlot}

@app.callback(dash.dependencies.Output('click-data', 'children'),[dash.dependencies.Input('mainGraph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)