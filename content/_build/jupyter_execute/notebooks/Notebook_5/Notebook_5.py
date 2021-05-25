# Notebook 5: Rational Method and Travel Times

In this notebook, we look at two ways of estimating a runoff hydrograph from precipitation data.

First, we'll use the rational method to approximate peak flow and the maximum water level at the basin outlet, and then we'll use an open-source library to make a higher resolution estimate of flow accumulation paths and stream networks using a digital elevation model.

# import required packages
import pandas as pd
import numpy as np
import math

# advanced statistics library
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mp
import matplotlib.colors as colors

# SEE COMMENTS ABOUT PYSHEDS LIBRARY IN NEXT CELL
from pysheds.grid import Grid
import warnings
warnings.filterwarnings('ignore')

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LinearColorMapper, LogTicker, ColorBar, BasicTickFormatter
from bokeh.io import output_notebook
output_notebook()

%matplotlib inline

### Import Precipitation Data

For this exercise, we will use historical climate data from the Meteorological Service of Canada (MSC) station at Whistler, BC.

# calibration data
df = pd.read_csv('../../data/Whistler_348_climate.csv', 
                 index_col='Date/Time', parse_dates=True)
# note that the 'head' command shows the first five rows of data, 
# but in this case the columns are abbreviated. 
# print(df.head())

# list all the columns
# print('')
# print('__________')
for c in df.columns:
    print(c)
    
stn_name = df['Station Name'].values[0]

### Plot the Data

It's always a good idea to begin by visualizing the data we're working with.

# plot flow at Stave vs. precip at the closest climate stations
p = figure(width=900, height=400, x_axis_type='datetime')
p.line(df.index, df['Total Precip (mm)'], alpha=0.8,
         legend_label="Total Precip [mm]", color='dodgerblue')

p.line(df.index, df['Total Snow (cm)'], alpha=0.8,
         legend_label="Total Snow [cm]", color="firebrick")

p.line(df.index, df['Total Rain (mm)'], alpha=0.8,
         legend_label="Total Rain [mm]", color='green')

p.legend.location = 'top_left'
p.legend.click_policy = 'hide'
p.xaxis.axis_label = 'Date'#
p.yaxis.axis_label = 'Daily Rain [mm] / Snow[cm] Volume'

show(p)

### Simplified Version of Rainfall-Runoff

First, isolate a single precipitation event to use for estimating a runoff hydrograph.  Let's find a nice week for skiing:  

fig, ax = plt.subplots(1, 1, figsize=(16,4))

sample_start = pd.to_datetime('2014-12-01')
sample_end = pd.to_datetime('2014-12-15')

sample_df = df[(df.index > sample_start) & (df.index < sample_end)][['Total Precip (mm)', 'Total Snow (cm)', 'Total Rain (mm)']]

# print(sample_df.head())

ax.plot(sample_df.index, sample_df['Total Precip (mm)'], label="Total Precip [mm]")
ax.plot(sample_df.index, sample_df['Total Snow (cm)'], label="Total Snow [cm]")
ax.plot(sample_df.index, sample_df['Total Rain (mm)'], label="Total Rain [mm]")

ax.set_xlabel('Date')
ax.set_ylabel('Precipitation')
ax.set_title('{}'.format(stn_name))
plt.legend()

First, imagine we are some unfortunate parking lot attendant working a shift in Whistler Village at Parking Lot 5, and we are told by our cruel supervisor we have stand at the lowest point of the parking lot: a catchment basin with an area of $1 km^2$ where water runs off into FitzSimmons Creek.  The sky looks angry, but we're running late for work and put on our running shoes instead of our sturdy waterproof boots.  

Next, assume the travel time is effectively zero across our entire basin (precipitation takes no time to travel to the outlet once it falls on the parking lot surface).  Is this a reasonable assumption in general?  

Under these assumptions, lets reconstruct a runoff hydrograph at the outlet.  First, look at the precipitation data over the twelve days of the big storm. 

print(sample_df)

### Convert Volume to volmeteric flow units

Runoff is typically measured in $\frac{m^3}{s}$, so convert $\frac{mm}{day}$ precipitation to $\frac{m^3}{s}$ runoff.

$$1 \frac{mm}{day} \times \frac{1 m}{1000 mm} \times \frac{1 day}{24 h} \times \frac{1 h}{ 3600 s} \times 1 km^2 \times \frac{1000 m \times 1000 m}{1 km^2}= \frac{1}{86.4} \frac{m^3}{s}$$

# convert to runoff volume
drainage_area = 1 # km^2

# runoff is typically measured in m^3/s (cms for short -- cubic metres per second), 
# so express the runoff in cms
sample_df['runoff_cms'] = sample_df['Total Rain (mm)'] / 86.4
print(sample_df)

If the channel outlet has a rectangular shape of width 2m, how tall should our boots be?  Assume a 2% slope, and find a reasonable assumption for the roughness of asphalt.

Recall the Manning equation:

$$Q = \frac{1}{n} A R^{2/3} S^{1/2}$$

Where:
* **n** is the manning roughness
* **A** is cross sectional area of the flow
* **R** hydraulic radius (area / wetted perimeter)
* **S** is the channel slope

w_channel = 1.5 # m
S = 0.005 # channel slope
n_factor = 0.017  # rough asphalt

def calc_Q(d, w, S, n):
    """
    Calculate flow from the Manning equation.
    """
    A = d * w
    wp = w + 2 * d  # wetted perimeter
    R = A / wp
    return (1/n) * A * R**(2/3) * S**(1/2)

def solve_depth(w, n_factor, Q, S):
    """
    Given a flow, a roughness factor, a channel slope, and a channel width, 
    calculate flow depth. 
    """
    e = 1 / 100  # solve within 1%
    d = 0
    Q_est = 0
    n = 0
    while (abs(Q_est - Q) > e) & (n < 1000):
        Q_est = calc_Q(d, w, S, n_factor)
#         print(Q, Q_est, abs(Q_est - Q))
        d += 0.001
        n += 1
#     print('solved in {} iterations'.format(n))
    return d 
    

# For each timestep, we want to solve for the depth of water at our outlet
sample_df['flow_depth_m'] = sample_df['runoff_cms'].apply(lambda x: solve_depth(w_channel, n_factor, x, S))


plt.plot(sample_df.index, sample_df['flow_depth_m'])
plt.ylabel('Flow depth [m]')

>**Not only are our feet wet, but if we happen to be there the peak it's potentially dangerous.  As little as 10-15cm of water moving fast enough can sweep you off your feet.**

![Recalculating Life](img/recalculating.png)

## More Complex Implementation: Spatial Data

As discussed in class, precipitation takes time to travel from where it fell to the basin outlet.  Next we will estimate the runoff response in a real catchment, just upstream from the parking lot example in the FitzSimmons Creek basin.

### Step 1: Instantiate a grid from a DEM raster
Some sample data is already included, but for extra data, see the [USGS hydrosheds project](https://www.hydrosheds.org/).

# grid = Grid.from_raster('data/n45w125_con_grid/n45w125_con/n45w125_con', data_name='dem')
grid = Grid.from_ascii(path='../../data/notebook_5_data/n49w1235_con_grid.asc', 
                       data_name='dem')

# reset the nodata from -32768 so it doesn't throw off the 
# DEM plot
grid.nodata = 0

# store the extents of the map
map_extents = grid.extent
min_x, max_x, min_y, max_y = map_extents

### Plot the DEM

**NOTE:** The cell below may take up to 30 seconds to load.  Please be patient, it is thinking really hard. 

The code below will plot the Digital Elevation Model (DEM).  

Do you recognize any features of the terrain?  Can you locate where it is?

Hover over the map (or touch if using a touchscreen) to see the coordinates in decimal degree units.

What does the [precision of the coordinates represent](http://wiki-1-1930356585.us-east-1.elb.amazonaws.com/wiki/index.php/Decimal_degrees)?  
* i.e. what does 5 decimal places in decimal degrees equate to in kilometers?

You can interact with the plot by using the tools on the left (in vertical order from top to bottom):
* **pan:** move around the map
* **box zoom:** draw a square to zoom in on
* **wheel zoom:** use the mousewheel (or pinch gesture on a touchscreen) to zoom in
* **box zoom:** draw a square to zoom in on
* **tap**: not yet implemented (but you can see the coordinates)
* **refresh**: reset the map
* **hover**: see the coordinates when hovering over the map with a mouse or pointer

# set bokeh plot tools
tools = "pan,wheel_zoom,box_zoom,reset,tap"

# show the precision of the decimal coordinates
# in the plot to 5 decimal places
TOOLTIPS = [
    ("(x,y)", "($x{1.11111}, $y{1.11111})"),
]

# create a figure, setting the x and y ranges to the appropriate data bounds
p1 = figure(title="DEM of the Lower Mainland of BC.  Hover to get coordintes.", plot_width=600, plot_height=int(400),
            x_range = map_extents[:2], y_range = map_extents[2:], 
            tools=tools, tooltips=TOOLTIPS)

# map elevation to a colour spectrum
color_mapper = LinearColorMapper(palette="Magma256", low=-200, high=2400)

# np.flipud flips the image data on a vertical axis
adjusted_img = [np.flipud(grid.dem)]  

p1.image(image=adjusted_img,   
         x=[min_x],               # lower left x coord
         y=[min_y],               # lower left y coord
         dw=[max_x-min_x],        # *data space* width of image
         dh=[max_y-min_y],        # *data space* height of image
         color_mapper=color_mapper
)

color_bar = ColorBar(color_mapper=color_mapper, #ticker=Ticker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

p1.add_layout(color_bar, 'right')


show(p1)

# -123.15512, 49.41293
#-123.14657, 49.41080
-123.14350, 49.40251

### Resolve flats in DEM

grid.resolve_flats('dem', out_name='inflated_dem')

### Specify flow direction values

#         N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)

grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)

fig = plt.figure(figsize=(8,6))
fig.patch.set_alpha(0)

plt.imshow(grid.dir, extent=grid.extent, cmap='viridis', zorder=2)
boundaries = ([0] + sorted(list(dirmap)))
plt.colorbar(boundaries= boundaries,
             values=sorted(dirmap))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow direction grid')
plt.grid(zorder=-1)
plt.tight_layout()
# plt.savefig('data/img/flow_direction.png', bbox_inches='tight')

# view the values of the raster as an array
grid.dir


# check the size of the raster
grid.dir.size

### Delineate a Catchment

Note that once you've executed the code in the cells below,
if you change the Point of Concentration (POC), you'll
need to go back to Step 1 and execute the code from there again.

This needs to be done to re-initialize the extents of the data 
that are loaded into memory.  The intermediary steps trim the extent
of the DEM and you will get an error message saying:


>`ValueError: Pour point (-123.94307, 49.40783) is out of bounds for dataset with bbox (-123.195000000122, 49.39999999984, -123.15333333347199, 49.421666666498).`



# Specify the Point of Concentration (POC) / Catchment Outlet (a.k.a. pour point) 
# This location is a tributary of the Capilano River, just above the reservoir above
# Cleveland Dam.
x, y = -123.14657, 49.41080
x, y = -123.14350, 49.40251

# And just for good measure, here's the little tributary in the south-west corner.
# Note the instructions above about reloading the original data to re-initialize
# the DEM
x, y = -123.15512, 49.41293

# Delineate the catchment
grid.clip_to('dem')
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)

# Clip the bounding box to the catchment we've chosen
grid.clip_to('catch', pad=(1,1,1,1))

# Create a view of the catchment
catch = grid.view('catch', nodata=np.nan)

# check the shape to see if we've estimated close enough to the 
# actual river to delineate the catchment successfully
print(catch.shape)
# if we get dimensions of < 10, we've missed and instead pointed at some 
# little hillslope

print(grid.extent)
ext_1 = grid.extent

# Plot the catchment
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
im = ax.imshow(catch, extent=grid.extent, zorder=1, cmap='viridis')
plt.colorbar(im, ax=ax, boundaries=boundaries, values=sorted(dirmap), label='Flow Direction')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Delineated Catchment')
# plt.savefig('data/img/catchment.png', bbox_inches='tight')

### Get flow accumulation

grid.accumulation(data='catch', dirmap=dirmap, out_name='acc')

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
acc_img = np.where(grid.mask, grid.acc + 1, np.nan)
im = ax.imshow(acc_img, extent=grid.extent, zorder=2,
               cmap='cubehelix',
               norm=colors.LogNorm(1, grid.acc.max()))
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Flow Accumulation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.savefig('data/img/flow_accumulation.png', bbox_inches='tight')


### Calculate distances to upstream cells

grid.flow_distance(data='catch', x=x, y=y, dirmap=dirmap, out_name='dist',
                   xytype='label', nodata_out=np.nan)

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
im = ax.imshow(grid.dist, extent=grid.extent, zorder=2,
               cmap='cubehelix_r')
plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Flow Distance')
# plt.savefig('data/img/flow_distance.png', bbox_inches='tight')

area_threshold=20
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)
plt.grid('on', zorder=0)
streamnetwork_img = np.where(acc_img>area_threshold, 100, 1+acc_img*0)
# print([['nan' if np.isnan(i) else cmap[i] for i in j] for j in streamnetwork_img])
labels = {1:'Catchment', 2: 'Outside Catchment', 100: 'Stream Network'}
cmap = {1: [0.247, 0.552, 0.266, 0.5], 
        100: [0.074, 0.231, 0.764, 0.8],
       2: [0.760, 0.760, 0.760, 0.8]}

arrayShow = np.array([[cmap[2] if np.isnan(i) else cmap[i] for i in j] for j in streamnetwork_img])    
## create patches as legend

patches =[mp.Patch(color=cmap[i],label=labels[i]) for i in cmap]

#streamnetwork_img = np.where(grid.mask, > 100, 10 , 1)
# im = ax.imshow(streamnetwork_img, extent=grid.extent, zorder=2,
#                 cmap='cubehelix')
im = ax.imshow(arrayShow)
plt.legend(handles=patches, loc=1, borderaxespad=0.)
# plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Stream network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
# plt.savefig('data/img/stream_network.png', bbox_inches='tight')

### Calculate weighted travel distance

Assign a travel time to each cell based on the assumption that water travels at one speed (slower) until it reaches a stream network, at which point its speed increases dramatically.

fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

plt.grid('on', zorder=0)
grid.clip_to('catch', pad=(1,1,1,1))
acc = grid.view('acc')

# calculate weights.
# assume the threshold is 100 accumulation cells 
# (of roughly 500mx500m) results in stream
weights = (np.where(acc, 0.1, 0)
               + np.where((0 < acc) & (acc <= 100), 1, 0)).ravel()

weighted_dist = grid.flow_distance(data='catch', x=x, y=y, weights=weights,
                   xytype='label', inplace=False)

im = ax.imshow(weighted_dist, extent=grid.extent, zorder=2,
               cmap='cubehelix_r')
# plt.legend(handles=patches, loc=1, borderaxespad=0.)
plt.colorbar(im, ax=ax, label='Upstream Cells')
plt.title('Stream network')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

## Develop Rainfall-Runoff Model

Now that we have weighted flow distances for each cell in the delineated catchment, we can apply precipitation to each 'cell' in order to reconstruct a flow hydrograph.  

First, we must figure out the cell dimensions.  From the [USGS Hydrosheds information](https://hydrosheds.cr.usgs.gov/datadownload.php), we know the resolution is 15 (degree) seconds.  Because of the odd shape of the earth, and the projection of coordinate systems onto the earth, there is a little bit of work involved in converting the DEM resolution to equivalent distances.  For the purpose of this exercise, we will assume cells are 300x300m.  You can check the approximation [here](https://opendem.info/arc2meters.html) for a latitude of 49 degrees.

# cells can be grouped by their weighted distance to the outlet to simplify 
# the process of calculating the contribution of each cell to flow at the outlet

dist_df = pd.DataFrame()
dist_df['weighted_dist'] = weighted_dist.flatten()
# trim the distance dataframe to include only the cells in the catchment,
# and round the travel time to the nearest one (hour)
dist_df = dist_df[dist_df['weighted_dist'] > 0].round(0)

start_date = sample_df.index.values[0]
end_date = sample_df.index.values[-1]
# create an hourly dataframe based on the sample precipitation event
# then resample the values to evenly distribute the total daily 
# precipitation to hourly precipitation
resampled_df = sample_df.resample('1H').pad() / 24

Note that our 'weighted distance' has just provided a relative difference between the flow accumulation cells and non-flow-accumulation cells.  We still must convert these values to some time-dependent form.

For this exercise, we will assume the average velocity of water is 1 m/s value for the flow accumulation cells, and 0.1 m/s for the other cells.  Therefore precipitation will take on average 300s (0.0833 h) and 3000s (0.833 h) to travel to the outlet for flow-accumulation and non-flow-accumulation cells, respectively.

# get the number of cells of each distance
grouped_dists = pd.DataFrame(dist_df.groupby('weighted_dist').size())
grouped_dists.columns = ['num_cells']

# create unit hydrographs for each timestep
runoff_df = pd.DataFrame(np.zeros(len(resampled_df)))
runoff_df.columns = ['Total Precip (mm)']
runoff_df.index = resampled_df.index.copy()
end_date = pd.to_datetime(runoff_df.index.values[-1]) + pd.DateOffset(hours=1)
max_distance = max(grouped_dists.index)

extended_df = pd.DataFrame()
extended_df['Total Precip (mm)'] = [0 for e in range(int(max_distance) + 1)]
extended_df.index = pd.date_range(end_date, periods=max_distance + 1, freq='1H')

# append the extra time to the runoff dataframe
runoff_df = runoff_df.append(extended_df)
runoff_df['Runoff (cms)'] = 0


**NOTE**: if you re-run the cell below, you need to run the cell above as well, or the runoff dataframe will not reset and the values will keep increasing.

cell_size = 300 # assume each pixel represents 300m x 300m 
runoff_coefficient = 0.3


for ind, row in resampled_df.iterrows():
    this_hyd = resampled_df[['Total Precip (mm)']].copy()
    for weight_dist, num_cells in grouped_dists.iterrows():
        try: 
            outlet_time = ind + pd.DateOffset(hours=weight_dist)
            precip = num_cells.values[0] * row['Total Precip (mm)']
            runoff_vol = precip * runoff_coefficient / 1000 * cell_size**2 
            runoff_rate = runoff_vol  / 3600  # convert to m^3/s from m^3/h
            runoff_df.loc[outlet_time, 'Runoff (cms)'] += runoff_rate
        except KeyError as err:
            print('error')
            break
#             print(err)
#             print(ind, row)

fig, ax = plt.subplots(1, 1, figsize=(16,4))

sample_start = pd.to_datetime('2014-12-01')
sample_end = pd.to_datetime('2014-12-15')

sample_df = df[(df.index > sample_start) & (df.index < sample_end)][['Total Precip (mm)', 'Total Snow (cm)', 'Total Rain (mm)']]

# print(sample_df.head())
# plot the original daily precip
# ax.plot(sample_df.index, sample_df['Total Precip (mm)'], label="Total Precip [mm]")
# ax.plot(sample_df.index, sample_df['Total Snow (cm)'], label="Total Snow [cm]")
# ax.plot(sample_df.index, sample_df['Total Rain (mm)'], label="Total Rain [mm]")
ax.plot(runoff_df.index, runoff_df['Runoff (cms)'], label="Runoff")


ax.set_xlabel('Date')
ax.set_ylabel('Runoff [cms]', color='blue')
ax.set_title('Example Rainfall-Runoff Model')
ax.legend(loc='upper left')
ax.tick_params(axis='y', colors='blue')

ax1 = ax.twinx()
ax1.plot(sample_df.index, sample_df['Total Rain (mm)'], 
         color='green',
         label="Total Rain")
ax1.set_ylabel('Precipitation [mm]', color='green')
ax1.tick_params(axis='y', colors='green')
ax1.legend(loc='upper right')

### Determine the Peak Unit Runoff

First, estimate the drainage area.  Then, find the peak hourly flow.

DA = round(grouped_dists.sum().values[0] * 0.3 * 0.3, 0)
max_UR = runoff_df['Runoff (cms)'].max() / DA * 1000
print('The drainage area is {} km^2 and the peak Unit Runoff is {} L/s/km^2'.format(DA, int(max_UR)))

Discuss the limitations of the approach.  Where do uncertainties exist?

* assumed precipitation is constant across days
* assumed constant runoff coefficient
* assumed two weights for travel time, constant across time

## Question for Reflection

For the first part where we estimated the water level at the parking lot outlet based on an assumption that there was zero infiltration, assuming all else is equal, how could we reduce the maximum water level to 5 cm?  

