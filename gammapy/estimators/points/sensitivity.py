# Licensed under a 3-clause BSD style license - see LICENSE.rst
import warnings
import numpy as np
from astropy.table import Column, Table
from gammapy.maps import Map
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.modeling.selection import TestStatisticNested
from gammapy.modeling.parameter import restore_parameters_status
from gammapy.stats import WStatCountsStatistic
from gammapy.stats.utils import ts_to_sigma
from ..core import Estimator
from ..utils import apply_threshold_sensitivity
from gammapy.utils.deprecation import deprecated_renamed_argument
from gammapy.utils.roots import find_roots

__all__ = ["SensitivityEstimator", "ParameterSensitivityEstimator"]


class SensitivityEstimator(Estimator):
    """Estimate sensitivity.

    This class allows to determine for each reconstructed energy bin the flux
    associated to the number of gamma-ray events for which the significance is
    ``n_sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent
    larger than the number of background events in the ON region.


    Parameters
    ----------
    spectral_model : `~gammapy.modeling.models.SpectralModel`, optional
        Spectral model assumption. Default is power-law with spectral index of 2.
    n_sigma : float, optional
        Minimum significance. Default is 5.
    gamma_min : float, optional
        Minimum number of gamma-rays. Default is 10.
    bkg_syst_fraction : float, optional
        Fraction of background counts above which the number of gamma-rays is. Default is 0.05.

    Examples
    --------
    For a usage example see :doc:`/tutorials/analysis-1d/cta_sensitivity` tutorial.

    """

    tag = "SensitivityEstimator"

    @deprecated_renamed_argument("spectrum", "spectral_model", "v1.3")
    def __init__(
        self,
        spectral_model=None,
        n_sigma=5.0,
        gamma_min=10,
        bkg_syst_fraction=0.05,
    ):
        if spectral_model is None:
            spectral_model = PowerLawSpectralModel(
                index=2, amplitude="1 cm-2 s-1 TeV-1"
            )

        self.spectral_model = spectral_model
        self.n_sigma = n_sigma
        self.gamma_min = gamma_min
        self.bkg_syst_fraction = bkg_syst_fraction

    def estimate_min_excess(self, dataset):
        """Estimate minimum excess to reach the given significance.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset.

        Returns
        -------
        excess : `~gammapy.maps.RegionNDMap`
            Minimal excess.
        """
        n_off = dataset.counts_off.data

        stat = WStatCountsStatistic(
            n_on=dataset.alpha.data * n_off, n_off=n_off, alpha=dataset.alpha.data
        )
        excess_counts = stat.n_sig_matching_significance(self.n_sigma)

        excess_counts = apply_threshold_sensitivity(
            dataset.background.data,
            excess_counts,
            self.gamma_min,
            self.bkg_syst_fraction,
        )

        excess = Map.from_geom(geom=dataset._geom, data=excess_counts)
        return excess

    def estimate_min_e2dnde(self, excess, dataset):
        """Estimate dnde from a given minimum excess.

        Parameters
        ----------
        excess : `~gammapy.maps.RegionNDMap`
            Minimal excess.
        dataset : `~gammapy.datasets.SpectrumDataset`
            Spectrum dataset.

        Returns
        -------
        e2dnde : `~astropy.units.Quantity`
            Minimal differential flux.
        """
        energy = dataset._geom.axes["energy"].center

        dataset.models = SkyModel(spectral_model=self.spectral_model)
        npred = dataset.npred_signal()

        phi_0 = excess / npred

        dnde_model = self.spectral_model(energy=energy)
        dnde = phi_0.data[:, 0, 0] * dnde_model * energy**2
        return dnde.to("erg / (cm2 s)")

    def _get_criterion(self, excess, bkg):
        is_gamma_limited = excess == self.gamma_min
        is_bkg_syst_limited = excess == bkg * self.bkg_syst_fraction
        criterion = np.chararray(excess.shape, itemsize=12)
        criterion[is_gamma_limited] = "gamma"
        criterion[is_bkg_syst_limited] = "bkg"
        criterion[~np.logical_or(is_gamma_limited, is_bkg_syst_limited)] = (
            "significance"
        )
        return criterion

    def run(self, dataset):
        """Run the sensitivity estimation.

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to compute sensitivity for.

        Returns
        -------
        sensitivity : `~astropy.table.Table`
            Sensitivity table.
        """
        energy = dataset._geom.axes["energy"].center
        excess = self.estimate_min_excess(dataset)
        e2dnde = self.estimate_min_e2dnde(excess, dataset)
        criterion = self._get_criterion(
            excess.data.squeeze(), dataset.background.data.squeeze()
        )

        return Table(
            [
                Column(
                    data=energy,
                    name="e_ref",
                    format="5g",
                    description="Energy center",
                ),
                Column(
                    data=dataset._geom.axes["energy"].edges_min,
                    name="e_min",
                    format="5g",
                    description="Energy edge low",
                ),
                Column(
                    data=dataset._geom.axes["energy"].edges_max,
                    name="e_max",
                    format="5g",
                    description="Energy edge high",
                ),
                Column(
                    data=e2dnde,
                    name="e2dnde",
                    format="5g",
                    description="Energy squared times differential flux",
                ),
                Column(
                    data=np.atleast_1d(excess.data.squeeze()),
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=np.atleast_1d(dataset.background.data.squeeze()),
                    name="background",
                    format="5g",
                    description="Number of background counts in the bin",
                ),
                Column(
                    data=np.atleast_1d(criterion),
                    name="criterion",
                    description="Sensitivity-limiting criterion",
                ),
            ]
        )


class ParameterSensitivityEstimator:
    """Estimate the sensitivity to a given parameter

    Computes the TS distribution in the non-null hypothesis using the
    log likelihood of the Asimov dataset (i.e. a dataset with counts = npred)
    and the non central chi2 distribution.
    Once the TS distribution under the testing hypothesis is known,
    one can compute the required parameter value
    to have 50% of measurements above a given significance threshold.


    Parameters
    ----------
    parameter : `~gammapy.modeling.Parameter`
       Parameter to test
    null_value : float or `~gammapy.modeling.Parameter`
        Value of the parameter for the null hypothesis.
    n_sigma : int, default=5
        Number of required significance level.
    rtol : float
        Relative precision of the estimate. Used as a stopping criterion.
        Default is 0.01.
    max_niter : int
        Maximal number of iterations used by the root finding algorithm.
        Default is 100.

    References
    ----------
        Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.

    """

    tag = "ParameterSensitivityEstimator"

    def __init__(
        self,
        parameter,
        null_value,
        n_sigma=5,
        n_free_parameters=None,
        rtol=0.01,
        max_niter=100,
    ):
        self.test = TestStatisticNested(
            [parameter], [null_value], n_free_parameters=n_free_parameters
        )
        self.parameter = parameter
        self.n_sigma = n_sigma
        self.rtol = rtol
        self.max_niter = max_niter

    def _fcn(self, value, datasets):
        """Call the Test Statistics function."""
        self.parameter.value = value
        ts_asimov = self.test.ts_asimov(datasets)
        return ts_to_sigma(ts_asimov) - self.n_sigma
        # return ts_to_sigma(ts_asimov, methos="cowan") - self.n_sigma # requires #5638

    def parameter_matching_significance(self, datasets):
        """Parameter value  matching the target significance"""

        if ~np.isfinite(self.parameter.min):
            vmin = self.parameter.value / 1000
        else:
            vmin = self.parameter.min
        if ~np.isfinite(self.parameter.max):
            vmax = self.parameter.value * 1000
        else:
            vmax = self.parameter.max

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots, res = find_roots(
                self._fcn,
                vmin,
                vmax,
                args=(datasets,),
                nbin=1,
                maxiter=self.max_niter,
                rtol=self.rtol,
                points_scale=self.parameter.interp,
            )
            # Where the root finding fails NaN is set as norm

        return roots[0]

    def run(self, datasets):
        """Parameter sensitivity
        given as the diffrence between vaue matching the target significance and the null value.
        """
        with restore_parameters_status(self.test.parameters):
            value = self.parameter_matching_significance(datasets)

        return value - self.test.null_values[0]
