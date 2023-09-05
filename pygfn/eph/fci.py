import numpy

import pyscf
from pyscf import fci, lib
from pyscf.fci.direct_spin1 import contract_1e
from pyscf.fci.direct_spin1 import contract_2e

from pyscf.fci.cistring import num_strings
from pyscf.fci.cistring import gen_linkstr_index
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.direct_ep import contract_pp

import pygfn

def _pack(h, omega, eta, v=None, comp="ip"):
    assert h.shape == v.shape
    pm = 1.0 if comp == "ip" else -1.0
    omega_eta = (omega - pm * eta * 1.0j)
    return pm * h + omega_eta * v

def _gen_hop_direct(gfn_obj, comp="ip", verbose=None):
    assert comp in ["ip", "ea"]

    norb = gfn_obj.norb
    nelec0 = gfn_obj.nelec0
    nelec = gfn_obj._nelec_ip if comp == "ip" else gfn_obj._nelec_ea
    assert nelec[0] >= 0 and nelec[1] >= 0
    assert nelec[0] <= norb and nelec[1] <= norb

    h1e = gfn_obj._h1e
    eri = gfn_obj._eri
    ene0 = gfn_obj.ene0

    h1p = gfn_obj.h1p
    h1e1p = gfn_obj.h1e1p

    vec_hdiag = gfn_obj._base.make_hdiag(h1e, eri, norb, nelec)
    vec_size = vec_hdiag.size
    vec_ones = numpy.ones(vec_size)

    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])
    assert na * nb == vec_size

    if eri is None:
        def gen_hv0(omega, eta):
            def hv0(v):
                c = v.reshape(na, nb)
                hc = contract_1e(h1e, c, norb, nelec)
                hc0 = hc - ene0 * c
                return _pack(hc0.reshape(-1), omega, eta, v=v, comp=comp)

            return hv0

        def gen_hd0(omega, eta):
            return _pack(vec_hdiag - ene0, omega, eta, v=vec_ones, comp=comp)

    else:
        h2e = gfn_obj._base.absorb_h1e(h1e, eri, norb, nelec, .5)
        def gen_hv0(omega, eta):
            def hv0(v):
                c = v.reshape(na, nb)
                hc_real = contract_2e(h2e, c.real, norb, nelec)
                hc_imag = contract_2e(h2e, c.imag, norb, nelec)
                hc0 = hc_real + 1j * hc_imag - ene0 * c
                return _pack(hc0.reshape(-1), omega, eta, v=v, comp=comp)
            return hv0

        def gen_hd0(omega, eta):
            return _pack(vec_hdiag - ene0, omega, eta, v=vec_ones, comp=comp)

        return gen_hv0, gen_hd0

class WithPhonon(lib.StreamObject):
    nmode = None # Number of phonon modes
    h1p   = None # Phonon Hamiltonian
    h1e1p = None # Electron-phonon interaction Hamiltonian

    def __init__(self, h1p=None, h1e1p=None):
        if h1p is None:
            nmode = 0
        else:
            nmode = h1p.shape[0]
        assert h1p.shape == (nmode, nmode)
        assert h1e1p.shape[0] == nmode

        self.nmode = nmode
        self.h1p = h1p
        self.h1e1p = h1e1p

class FullConfigurationInteractionDirectSpin1(pygfn.fci.FullConfigurationInteractionDirectSpin1, WithPhonon):
    def __init__(self, hf_obj, h1p=None, h1e1p=None):
        super().__init__(hf_obj)
        WithPhonon.__init__(self, h1p=h1p, h1e1p=h1e1p)

        nsite = self.nsite
        nmode = self.nmode

        assert self.h1p.shape == (nmode, nmode)
        assert self.h1e1p.shape == (nmode, nsite, nsite)

    def get_rhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        vec0 = self.vec0 if vec0 is None else vec0
        na, nb = vec0.shape[:2]
        np = vec0.reshape(na * nb, -1).shape[1]

        gen_rhs_ip = lambda v: super().get_rhs_ip(vec0=v, orb_list=orb_list, verbose=verbose)
        rhs_ip = [gen_rhs_ip(vec0.reshape(na * nb, -1)[:, i]) for i in range(np)]
        rhs_ip = numpy.asarray(rhs_ip).transpose([1, 2, 3, 0]).reshape(len(orb_list), -1)

        return rhs_ip

    def get_lhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        vec0 = self.vec0 if vec0 is None else vec0
        na, nb = vec0.shape[:2]
        np = vec0.reshape(na * nb, -1).shape[1]

        gen_lhs_ip = lambda v: super().get_lhs_ip(vec0=v, orb_list=orb_list, verbose=verbose)
        lhs_ip = [gen_lhs_ip(vec0.reshape(na * nb, -1)[:, i]) for i in range(np)]
        lhs_ip = numpy.asarray(lhs_ip).transpose([1, 2, 3, 0]).reshape(len(orb_list), -1)

        return lhs_ip

    def get_rhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        vec0 = self.vec0 if vec0 is None else vec0
        na, nb = vec0.shape[:2]
        np = vec0.reshape(na * nb, -1).shape[1]

        gen_rhs_ea = lambda v: super().get_rhs_ea(vec0=v, orb_list=orb_list, verbose=verbose)
        rhs_ea = [gen_rhs_ea(vec0.reshape(na * nb, -1)[:, i]) for i in range(np)]
        rhs_ea = numpy.asarray(rhs_ea).transpose([1, 2, 3, 0]).reshape(len(orb_list), -1)

        return rhs_ea

    def get_lhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        vec0 = self.vec0 if vec0 is None else vec0
        na, nb = vec0.shape[:2]
        np = vec0.reshape(na * nb, -1).shape[1]

        gen_lhs_ea = lambda v: super().get_lhs_ea(vec0=v, orb_list=orb_list, verbose=verbose)
        lhs_ea = [gen_lhs_ea(vec0.reshape(na * nb, -1)[:, i]) for i in range(np)]
        lhs_ea = numpy.asarray(lhs_ea).transpose([1, 2, 3, 0]).reshape(len(orb_list), -1)

        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)


