#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import a few useful libraries
import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt


# # Example: Clausius-Clapeyron Equation
# 
# ## Introduction
# 
# This notebook is an interactive development environment (IDE) where you can run Python code to do calculations, numerical simuluation, and much more.
# 
# For this example, we'll plot the atmospheric saturation water vapour pressure as a function of air temperature.

# The numpy library has a `linspace()` function that creates arrays of numbers with specific properties.
# 
# Here we're interested in looking at saturation vapour pressure for a range of temperatures we want to explore.  Say **0 degrees to 30 degrees Celsius** (the relationship does not hold below zero).  We can also specify how many points we want between the minimum and maximum we've set.  Let's say 50 for now.

# In[ ]:


min_temp = 1
max_temp = 35
temperature_range = np.linspace(min_temp, max_temp, 50)

# alternatively we could specify the step size


# When executing the cell block above, did you get an error that says `NameError: name 'np' is not defined`?
# 
# Recall that code cells must be executed in order to load the requisite libraries, variables, etc. into memory.  The error above suggests the very first cell in this notebook wasn't executed, so the numpy library is not yet accessible in the variable `np`.  Note the line `import numpy as np` loads the numpy library and makes its many functions available from the variable `np`. 

# In[ ]:


temperature_range


# ## Markdown
# 
# ![Block-type dropdown menu](img/block_type.png)
# 
# This cell/block is set to "Markdown" which is an easy way to format text nicely.  
# 
# More information on formatting text blocks using Markdown [can be found here](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html).
# 
# Most academic writing is formatted using a system called [LaTeX](https://www.latex-project.org/).  
# 
# >**Note:** If you are thinking about grad school, you will most likely end up learning LaTeX for publishing papers.  If you can work with Markdown (hint: you can!), it isn't much further to preparing your work using LaTeX.  [Overleaf](https://www.overleaf.com/) is a great web application for storage and collaborative editing of LaTeX documents.
# 

# Let's write the Clausius-Clapeyron equation in a print-worthy format:

# ## Clausius-Clapeyron Equation
# 
# The change in saturation vapour pressure of air as a function of temperature is given in differential form by:
# 
# $$\frac{de_s}{dT} = \frac{L_v(T)\cdot e_s}{R_v \cdot T^2}$$ 
# 
# Assuming $L_v$ is constant yields the approximation$^{[1]}$:
# 
# $$e_s(T) = e_{s0} \cdot exp \left(\frac{L_v}{R_v} \left[ \frac{1}{T_0} - \frac{1}{T} \right] \right) $$
# 
# 
# Where:
# * $L_v$ is the latent heat of vaporization, (constant approximation 0-35 Celsius = $2.5\times10^6 \frac{J}{kg \cdot K}$)
# * $R_v$ is the vapor pressure gas constant ($461 \frac{J}{kg \cdot K}$)
# * $T$ is air temperature in Kelvin
# * $T_0$ and $e_{s0}$ are constants ($273 K$ and $0.611 kPa$)
# 
# 1.  Margulis, S. *Introduction to Hydrology*. 2014.

# We can write this as a function in Python:

# In[ ]:


def calculate_saturation_vapour_pressure(T):
    """
    Given T (temperature) as an input in Celsius,
    return the saturation vapour pressure of air.
    Output units are in kiloPascals [kPa].
    """
    e_s0 = 0.611
    L_v = 2.5E6
    R_v = 461
    T_0 = 273.16
    T_k = T + T_0
    return e_s0 * math.exp( (L_v/R_v) * (1/T_0 - 1/T_k))
    


# It's good practice to write functions into simple components so they can be reused and combined.  
# 
# Calculate the saturation vapour pressure for the temperature range we defined above:

# In[ ]:


# create an empty array to store the vapour pressures we will calculate
vapour_pressures = []
# iterate through the temperature array we created above
for t in temperature_range:
    sat_vapour_pressure = calculate_saturation_vapour_pressure(t)
    vapour_pressures.append(sat_vapour_pressure)


# In[ ]:


# now plot the result 
# note in the matplotlib plotting library the figsize is defined in inches by default
# here we're saying 10" wide by 6" high

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(temperature_range, vapour_pressures, 'b-')
ax.set_title('Saturation Vapour Pressure vs. Temperature')
ax.set_xlabel('Temperature (Celsius)')
ax.set_ylabel('Saturation Vapour Pressure (kPa)')


# Below we'll create a function to calculate dewpoint temperature that uses the vapour pressure function above.

# In[ ]:


def calculate_dewpoint_temperature(rh, T):
    """
    Given relative humidity and ambient temperature (in Celsius), 
    return the dewpoint temperature in Celsius.
    """
    # declare constants
    L_v = 2.5E6
    R_v = 461
    e_s0 = 0.611
    T_0 = 273.16
    
    e_s = calculate_saturation_vapour_pressure(T)
    # calculate the (actual) vapour pressure
    e_a = rh * e_s
    # calculate the dewpoint temperature    
    T_dk = 1 / (1/T_0 - (R_v / L_v) * np.log(e_a / e_s0))
    
    T_d = T_dk - T_0
    
    # if the dewpoint temperature is below zero, return NaN
    if T_d < 0:
        return np.nan
    else:
        return T_d


# Let's assume we want to explore the dewpoint temperature as a function of relative humidity.

# In[ ]:


# create an array to represent the relative humidity from 10% to 100%
rh_range = np.linspace(0.1, 1, 10)


# This time we'll use a *list comprehension* instead of a "for" loop to calculate the dewpoint temperature where we assume temperature is constant but we want to evaluate a range of relative humidity.  When might we encounter such a situation?

# In[ ]:


t_amb = 25
dewpt_temps = [calculate_dewpoint_temperature(rh, t_amb) for rh in rh_range]


# In[ ]:


# now plot the result 
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(rh_range, dewpt_temps, 'b-')
ax.set_title(f'Dewpoint Temperatures by Relative Humidity for Ambient Temperature = {t_amb} Celsius')
ax.set_xlabel('Relative Humidity [/100]')
ax.set_ylabel('Dewpoint Temperature (Celsius)')


# Let's get really fancy and create a heat map to express the relationship between ambient air temperature, relative humidity, and dewpoint temperature.
# 
# See this [gist example](https://gist.github.com/Kautenja/f9d6fd3d1dee631200bc11b8a46a76b7) used as a template.

# In[ ]:


ambient_temp_range = np.linspace(0, 50, 100)
rh_range = np.linspace(0.01, 1.0, 100)

# create an empty dataframe to store results
dewpt_df = pd.DataFrame()
dewpt_df['T_amb'] = ambient_temp_range 
dewpt_df.set_index('T_amb', inplace=True)

for r in rh_range:
    dewpt_df[r] = [calculate_dewpoint_temperature(r, t) for t in dewpt_df.index]


# In[ ]:


data = dewpt_df.T


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 6))
img = ax.imshow(data, cmap='inferno')
fig.colorbar(img)

ambient_temp_labels = np.arange(0, 50, 5)
rh_labels = np.linspace(0.01, 1.0, 10).round(1)

label_locs = np.arange(0, 100, 10)

# Show all ticks and label them with the respective list entries
# set x ticks to ambient temperature and y to relative humidity
ax.set_yticks(label_locs)
ax.set_xticks(label_locs)
ax.set_yticklabels(ambient_temp_labels)
ax.set_xticklabels(rh_labels)

ax.set_xlabel('Relative Humidity [/100]')
ax.set_ylabel('Ambient Temperature (Celsius)')

ax.set_title("Dewpoint Temperature [degrees Celsius]")


# ### Other examples of functions related to atmospheric energy balance

# In[ ]:


def calculate_blackbody_radiation(T):
    """
    Given T (temperature) as an input in Kelvin,
    return the unit area (area = 1m^2) 
    **black body** radiative power in W/m^2/K^4.
    """
    sigma=5.670374419e-8
    return sigma*(T^4)


# In[ ]:


def calculate_greybody_radiation(T,emiss):
    """
    Given T (temperature) as an input in Kelvin,
    return the unit area (area = 1m^2) 
    **grey body** radiative power in W/m^2/K^4..
    """
    sigma=5.670374419e-8
    return sigma*emiss*(T^4)


# Stefan-Bolzmann Constant ($\sigma$):
# $\sigma = 5.670374419×10^8$ $W m^{−2}⋅K^{−4}$

# In[ ]:


calculate_blackbody_radiation(5500)


# In[ ]:





# In[ ]:




