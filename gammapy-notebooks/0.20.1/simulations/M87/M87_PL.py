import gammapy
from astropy import units as u
import numpy as np
from astropy.io import ascii
import collections
import sys, os
import matplotlib.pyplot as plt
from gammapy.makers import SpectrumDatasetMaker, SafeMaskMaker, ReflectedRegionsBackgroundMaker
from gammapy.modeling import Fit
from gammapy.data import Observation, Observations
from gammapy.datasets import SpectrumDatasetOnOff, SpectrumDataset, Datasets
from gammapy.irf import load_cta_irfs
from gammapy.maps import MapAxis, RegionGeom

from gammapy.modeling.models import (
    EBLAbsorptionNormSpectralModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

from gammapy.irf import EffectiveAreaTable2D

from numpy.random import RandomState

from scipy.stats import chi2, norm

from gammapy.estimators import FluxPointsEstimator
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset

# astropy imports
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column

from gammapy.estimators import SensitivityEstimator

# astropy affiliated packages imports
from regions import CircleSkyRegion

from gammapy.stats import WStatCountsStatistic
from gammapy.stats import CashCountsStatistic
from scipy.stats import sem
from gammapy.maps import Map
from regions import PointSkyRegion

import math

# Define simulation parameters parameters
livetimes = 5 * u.h
emin = 25 * u.GeV
emax = 100. * u.TeV

srcposition = SkyCoord(187.73, 12.36, unit="deg", frame="icrs")

offset = 0.5 * u.deg
pointing = SkyCoord(srcposition.ra, srcposition.dec + offset, unit="deg", frame="icrs")

# Reconstructed and true energy axis
energy_reco = MapAxis.from_energy_bounds(emin, emax, nbin=5, per_decade=True, name="energy")
    # true energy should be wider than reco energy
energy_true = MapAxis.from_energy_bounds(0.3*emin, 3*emax, nbin=8, per_decade=True, name="energy_true")

on_region_radius = Angle("0.11 deg")

on_region = CircleSkyRegion(center=srcposition, radius=on_region_radius)

# Define spectral model - a simple Power Law in this case
specmodel = PowerLawSpectralModel(
    index=2.24,
    amplitude=6.47e-13 * u.Unit("cm-2 s-1 TeV-1"),
    reference=0.1 * u.TeV,
)

# we set the sky model used in the dataset

absorption = EBLAbsorptionNormSpectralModel.read_builtin("dominguez", redshift=0.0043)

absspecmodel = specmodel * absorption

skymodel = SkyModel(spectral_model=absspecmodel, name="model_simu")

obs = Observation.create(pointing=pointing, livetime=livetimes, irfs=load_cta_irfs('/home/luiz/gammapy/irfs/caldb/data/cta/prod5-v0.1/bcf/North_z20_5h/irf_file.fits'))

geom = RegionGeom.create(region=on_region, axes=[energy_reco])
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_true)
maker = SpectrumDatasetMaker(containment_correction=True, selection=["edisp", "background", "exposure"])
safe_maker = SafeMaskMaker(methods=["bkg-peak"])

datasets = Datasets()

dataset = maker.run(dataset_empty, obs)
dataset = safe_maker.run(dataset, obs)
low_energies_safe = dataset.energy_range_safe[0]
energy_threshold = low_energies_safe.get_by_coord({'skycoord': srcposition})[0] * low_energies_safe.unit
dataset_onoff = SpectrumDatasetOnOff.from_spectrum_dataset(dataset=dataset, acceptance=1, acceptance_off=5)
dataset_onoff.fake(random_state='random-seed', npred_background=dataset.npred_background())
print(dataset_onoff)
datasets.append(dataset_onoff)

significance = WStatCountsStatistic(n_on=sum(dataset_onoff.counts.data), n_off=sum(dataset_onoff.counts_off.data), alpha=0.2).sqrt_ts
print(significance)

n_obs = 100
datasets = Datasets()

for idx in range(n_obs):
    dataset_onoff.fake(random_state=idx, npred_background=dataset.npred_background())
    dataset_fake = dataset_onoff.copy(name=f"obs-{idx}")
    dataset_fake.meta_table["OBS_ID"] = [idx]
    datasets.append(dataset_fake)

table = datasets.info_table()
    
fix, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].hist(table["counts"])
axes[0].set_xlabel("Counts")
axes[0].set_ylabel("Frequency")
axes[1].hist(table["counts_off"])
axes[1].set_xlabel("Counts Off")
axes[2].hist(table["excess"])
axes[2].set_xlabel("excess");
axes[3].hist(table["sqrt_ts"])
axes[3].set_xlabel(r"significance ($\sigma$)");

plt.savefig('./M87_gammapy_counts.png', bbox_inches='tight')
plt.savefig('./M87_gammapy_counts.pdf', bbox_inches='tight')

#Compute sensitivity

sensitivity_estimator = SensitivityEstimator(gamma_min=5, n_sigma=3, bkg_syst_fraction=0.10)
sensitivity_table = sensitivity_estimator.run(dataset_onoff)
print(sensitivity_table)

# Plot the sensitivity curve
t = sensitivity_table

fix, axes = plt.subplots(figsize=(12, 8))

axes.plot(t["energy"], t["e2dnde"], "s-", color="red")
axes.loglog()
axes.set_xlabel(f"Energy ({t['energy'].unit})", size=12)
axes.set_ylabel(f"Sensitivity ({t['e2dnde'].unit})", size=12)

plt.savefig('./M87_gammapy_sensitivity.png', bbox_inches='tight')
plt.savefig('./M87_gammapy_sensitivity.pdf', bbox_inches='tight')

#Compute flux points

datasets.models = [skymodel]

#fit_joint = Fit()
#result_joint = fit_joint.run(datasets=datasets)

# we make a copy here to compare it later
model_best_joint = skymodel.copy()

energy_edges = MapAxis.from_energy_bounds("0.1 TeV", "30 TeV", nbin=12).edges

fpe = FluxPointsEstimator(energy_edges=energy_edges, source="model_simu", selection_optional="all")
flux_points = fpe.run(datasets=datasets)

print(flux_points.to_table(sed_type="dnde", formatted=True))

#flux_points_dataset = FluxPointsDataset(data=flux_points, models=model_best_joint)

e_ref = []
e_min = []
e_max = []
dnde = []
dnde_err = []
subtract_emin = []
subtract_emax = []

for j in range(len(flux_points["energy_max"].value)):
    if flux_points["dnde"].data[j] > 0:
       e_ref.append(flux_points["energy_ref"].value[j])
       e_min.append(flux_points["energy_min"].value[j])
       e_max.append(flux_points["energy_max"].value[j])
       dnde.append(flux_points["dnde"].data[j][0])
       dnde_err.append(flux_points["dnde_err"].data[j][0])
       
flux_TEV = np.hstack((dnde))#*1e+06
flux_err_TEV = np.hstack((dnde_err))#*1e+06  
energy = np.hstack((e_ref))

#print(len(energy), len(flux_TEV))
#print(energy)
#print(np.asarray(flux_TEV))
#print(np.hstack((e_ref))-np.hstack((e_min)), np.hstack((e_max))-np.hstack((e_ref)))

energy_bounds = [0.1, 50] * u.TeV
plt.figure()
absspecmodel.plot(energy_bounds, label='intrinsic spectrum + EBL')
xerr = [np.hstack((e_ref))-np.hstack((e_min)), np.hstack((e_max))-np.hstack((e_ref))]
plt.errorbar(e_ref, flux_TEV, color='red', marker='o', xerr = xerr, yerr = flux_err_TEV, linestyle='', label='measured spectrum')
plt.grid(which="both")
plt.ylim(1e-24, 1e-8)
plt.legend(loc="best")
plt.title("M87")
plt.savefig('./spectrum_srcM87.png', bbox_inches='tight')

plt.savefig('./M87_gammapy_flux_point.png', bbox_inches='tight')
plt.savefig('./M87_gammapy_flux_point.pdf', bbox_inches='tight')

plt.show()
