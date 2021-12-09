"""
DKRZ matplotlib script:  matplotlib_contour_filled.py

   - filled contour over map plot
   - rectilinear grid (lat/lon)
   - colorbar

   12.06.15  meier-fleischer(at)dkrz.de
"""
from   mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
from   netCDF4 import Dataset as open_ncfile
import numpy as np

#-- open netcdf file
nc = open_ncfile('rectilinear_grid_2D.nc')

#-- read variable
var = nc.variables['tsurf'][0,:,:]
lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]

#-- create figure and axes instances
dpi = 100
fig = plt.figure(figsize=(1100/dpi, 1100/dpi), dpi=dpi)
ax  = fig.add_axes([0.1,0.1,0.8,0.9])

#-- create map
map = Basemap(projection='cyl',llcrnrlat= -90.,urcrnrlat= 90.,\
              resolution='c',  llcrnrlon=-180.,urcrnrlon=180.)

#-- draw coastlines, state and country boundaries, edge of map
map.drawcoastlines()
map.drawstates()
map.drawcountries()

#-- create and draw meridians and parallels grid lines
map.drawparallels(np.arange( -90., 90.,30.),labels=[1,0,0,0],fontsize=10)
map.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1],fontsize=10)

#-- convert latitude/longitude values to plot x/y values
x, y = map(*np.meshgrid(lon,lat))

#-- contour levels
clevs = np.arange(210,320,5)

#-- draw filled contours
cnplot = map.contourf(x,y,var,clevs,cmap=plt.cm.jet)

#-- add colorbar
cbar = map.colorbar(cnplot,location='bottom',pad="10%")      #-- pad: distance between map and colorbar
cbar.set_label('deg K')                                      #-- add colorbar title string

#-- add plot title
plt.title('Temperature')

#-- displa on screen
#plt.show()

#-- maximize and save PNG file
plt.savefig('plot_matplotlib_contour_filled_rect.png', bbox_inches='tight', dpi=dpi)
