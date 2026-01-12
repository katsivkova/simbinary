#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 09:06:39 2026

@author: esivkova
"""

import numpy as np
import simbinary as sim
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd
import numpy as np

root = '/Users/esivkova/phd/code/gaia_sim/'
plot_dir = '/Users/esivkova/phd/code/gaia_sim/plots/' # optional
data_dir = '/Users/esivkova/phd/code/gaia_sim/simDR4/' # optional
fit_dir = '/Users/esivkova/phd/code/gaia_sim/notebook_fit/' # optional

params = {'Object': 'V1334 Cyg', #simbad resolved name
# 'id3': 11111..., if no gaia resolved name, write gaia dr3 id here
 'type': 'cepheid', # can be 'binary' or 'BH'
 'P': 1932.8,
 'a': 8.54,
 'e': 0.233,
 'i': 124.94,
 'Omega': 213.17,
 'w': 229.8,
 'T0': 2453316.75,
 'q': 0.942,
 'pll': 1.388,
 'Vcomp': 7.99 # mandatory for types cepheid and binary
}

sb = sim.SimBinary(params, DataRelease = 4) # DataRelease from 1 to 5

w_bs = sb.SimDR4() # simulated along scan points
sim_astrometry = sb.get_dataframe(data_dir) # a dataframe

sb.PlotSim(plot_dir) # plot. plot_dir is optional

nt = sim.notebookDR4(params['Object'], sb.id3) # initialize fitting
nt.load_dataframe(sim_astrometry) # give data to fit
fit_parameters = nt.fitthething(fit_dir) # fit. fit_dir is optional

w_fit = nt.keplerian_model.model() # fitted along scan
ra_fit, dec_fit = sim.orbit2(fit_parameters, sb.reltimes.value) # fitted
ra_ph, dec_ph = sb.ra_ph, sb.dec_ph # sim data

fig, axs = plt.subplots(1,2, figsize=(14, 5), constrained_layout=True)
ax1, ax2 = axs
fig.suptitle('Simulated data vs fit')
ax1.plot(ra_ph, dec_ph, label = 'Simulated data', marker='.', lw = 4, color='lightpink')
ax1.plot(ra_fit, dec_fit, label = 'Fit', marker='.', color = 'black')
ax1.set_aspect('equal', adjustable='datalim')
ax1.xaxis.set_inverted(True)
ax1.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
ax1.set_ylabel(r'$\Delta \delta$ [mas]')
ax1.legend()

ax2.plot(sb.reltimes.value, w_bs, marker='.', lw = 4, color='lightpink')
ax2.plot(sb.reltimes.value, w_fit, marker='.', color = 'black')
ax2.set_xlabel(f'Time, Tref={sb.Tref}, days')
ax2.set_ylabel('Along scan, [mas]')