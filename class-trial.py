from nbodykit.lab import cosmology
import matplotlib.pyplot as plt
import numpy as np
import numpy

available = ['CLASS', 'EisensteinHu', 'NoWiggleEisensteinHu']

# minimum CLASS value to represent k->0
KMIN = 1e-8


a = 1e-3
redshift = (1. - a)/a

class CLASS(object):
    """
    The linear matter transfer function using the CLASS Boltzmann code.

    Parameters
    ----------
    cosmo : :class:`Cosmology`
        the cosmology instance
    redshift : float
        the redshift of the power spectrum
    """
    def __init__(self, cosmo, redshift):

        # make sure we have Pk
        if not cosmo.has_pk_matter:
            cosmo = cosmo.clone(output='mPk dTk vTk')
        self.cosmo = cosmo

        # find the low-k amplitude to normalize to 1 as k->0 at z = 0
        self._norm = 1.0; self.redshift = 0
        self._norm = 1. / self(KMIN)

        # the proper redshift
        self.redshift = redshift


    def __call__(self, k):
        r"""
        Return the CLASS linear transfer function at :attr:`redshift`. This
        computes the transfer function from the CLASS linear power spectrum
        using:

        .. math::

            T(k) = \left [ P_L(k) / k^n_s  \right]^{1/2}.

        We normalize the transfer function :math:`T(k)` to unity as
        :math:`k \rightarrow 0` at :math:`z=0`.

        Parameters
        ---------
        k : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`

        Returns
        -------
        Tk : float, array_like
            the transfer function evaluated at ``k``, normalized to unity on
            large scales
        """
        k = numpy.asarray(k)
        nonzero = k>0
        linearP = self.cosmo.get_pklin(k[nonzero], self.redshift) / self.cosmo.h**3 # in Mpc^3
        primordialP = (k[nonzero]*self.cosmo.h)**self.cosmo.n_s # put k into Mpc^{-1}

        # return shape
        Tk = numpy.ones(nonzero.shape)

        # at k=0, this is 1.0 * D(z), where T(k) = 1.0 at z=0
        Tk[~nonzero] = self.cosmo.scale_independent_growth_factor(self.redshift)

        # fill in all k>0
        Tk[nonzero] = self._norm * (linearP / primordialP)**0.5
        return Tk

plt.loglog(CLASS.__call__)
plt.show()



plt.close()
