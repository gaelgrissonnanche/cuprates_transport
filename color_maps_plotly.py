import matplotlib
import matplotlib.cm
import numpy as np

def magma_colormap():

  magma_cmap = matplotlib.cm.get_cmap('magma')
  magma_rgb = []

  norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

  for i in range(0, 255):
         k = matplotlib.colors.colorConverter.to_rgb(magma_cmap(norm(i)))
         magma_rgb.append(k)

  def matplotlib_to_plotly(cmap, pl_entries):
      h = 1.0/(pl_entries-1)
      pl_colorscale = []

      for k in range(pl_entries):
          C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
          pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

      return pl_colorscale

  # magma = matplotlib_to_plotly(magma_cmap, 255)

  return magma_rgb