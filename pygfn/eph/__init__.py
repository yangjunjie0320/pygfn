import inspect

import pyscf, numpy
from pyscf import gto, fci

import pygfn
from pygfn.eph import _fci
# from pygfn.eph.fci import FullConfigurationInteractionDirectSpin1
#
# def EPH(gf_obj, h1p=None, h1e1p=None):
#     assert isinstance(gf_obj, pygfn.fci.GreensFunctionMixin)
#     assert not isinstance(gf_obj, pygfn.eph.fci.WithPhonon)
#
#     if isinstance(gf_obj, pygfn.fci.FullConfigurationInteractionDirectSpin1):
#         return FullConfigurationInteractionDirectSpin1(gf_obj._base, h1p=h1p, h1e1p=h1e1p)
#
#     raise NotImplementedError

def FCI(m_or_mf, mo=None, singlet=None, h1p=None, h1e1p=None, nph_max=4):
    """FCI electronic-phonon solver. This function is a wrapper of the
    functions in pygfn.eph._fci.

    Note: the h1e1p will be transformed if mo is provided.

    Args:
        m_or_mf (Mole or SCF): The Mole or SCF object.
        mo (ndarray): The molecular orbital coefficients.
        singlet (bool): Whether to use singlet wave function.
        h1p (ndarray): The phonon Hamiltonian.
        h1e1p (ndarray): The phonon-electron Hamiltonian.
        nph_max (int): The maximum number of phonons/phontons.

    Returns:
        EPH-FCI object.
    """

    fci_obj = pyscf.fci.FCI(m_or_mf, mo=mo, singlet=singlet)

    h1e = inspect.signature(fci_obj.kernel).parameters['h1e'].default
    eri = inspect.signature(fci_obj.kernel).parameters['eri'].default
    norb = inspect.signature(fci_obj.kernel).parameters['norb'].default
    nelec = inspect.signature(fci_obj.kernel).parameters['nelec'].default
    ecore = inspect.signature(fci_obj.kernel).parameters['ecore'].default

    nsite = norb
    nmode = h1e1p.shape[2]
    assert h1p.shape == (nmode, nmode)

    if mo is not None:
        assert mo.ndim == h1e.ndim
        assert mo.shape[-1] == norb

        if h1e.ndim == 2:
            h1e1p = numpy.einsum('mp,nq,mnI->pqI', mo, mo, h1e1p)

        elif h1e.ndim == 3:
            assert mo.ndim == 3
            h1e1p = numpy.einsum('smp,snq,mnI->spqI', mo, mo, h1e1p)
            assert h1e1p.shape == (2, norb, norb, nmode)

    class CISolver(pyscf.lib.StreamObject):
        nroots = 1
        def __init__(self, mol=None):
            fci_obj.__class__.__init__(self, mol)

        def kernel(self, h1e=h1e, eri=eri, h1e1p=h1e1p, h1p=h1p,
                   nmode=nmode, norb=norb, nsite=None,
                   nelec=nelec, ci0=None, ecore=ecore,
                   **kwargs):

            if nsite is None:
                nsite = norb
            assert nsite == norb

            res = pygfn.eph._fci.kernel(
                h1e=h1e, eri=eri, h1e1p=h1e1p, h1p=h1p,
                nsite=nsite, nmode=nmode,
                nelec=nelec, nph_max=nph_max,
                ci0=ci0, h0=ecore, fci_obj=fci_obj,
                nroots=self.nroots,
                **kwargs
            )

            return res

    cisolver = CISolver(fci_obj.mol)
    cisolver.__dict__.update(fci_obj.__dict__)
    return cisolver