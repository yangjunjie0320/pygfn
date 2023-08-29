import numpy
import scipy
from line_profiler import profile

import pyscf
from pyscf import fci, cc, lib
from pyscf.fci.direct_spin1 import contract_1e
from pyscf.fci.direct_spin1 import contract_2e
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ip
from pyscf.cc.eom_rccsd import amplitudes_to_vector_ea

from pyscf import __config__

from pygfn.lib import gmres
from pygfn.fci import GreensFunctionMixin

class CoupledClusterSingleDoubleSpin0Slow(GreensFunctionMixin):
    is_approx_lambda = True
    def __init__(self, hf_obj):
        self._base : cc.ccsd.RCCSD
        self._base = cc.CCSD(hf_obj)
        self._base.verbose = 0

    def build(self, vec0=None):
        coeff = self._base._scf.mo_coeff
        assert coeff is not None
        nao, nmo = coeff.shape
        occ  = self._base._scf.get_occ(mo_coeff=coeff)

        cc_obj = cc.CCSD(self._base._scf, mo_coeff=coeff, mo_occ=occ)
        nocc = cc_obj.nocc
        eris = cc_obj.ao2mo(mo_coeff=coeff)
        self._eris = eris

        if vec0 is None:
            t1, t2 = cc_obj.init_amps(eris)[1:]
        else:
            t1, t2 = cc_obj.vector_to_amplitudes(vec0, nmo, nocc)
        ene0, t1, t2 = self._base.kernel(eris=eris, t1=t1, t2=t2)
        assert self._base.converged
        vec0 = self._base.amplitudes_to_vector(t1, t2)
        amp0 = (t1, t2)

        nelec0 = self._base.mol.nelec
        assert nelec0[0] >= nelec0[1]
        nelec_ip = (nelec0[0] - 1, nelec0[1])
        nelec_ea = (nelec0[0], nelec0[1] + 1)
        self.nelec0 = nelec0
        self._nelec_ip = nelec_ip
        self._nelec_ea = nelec_ea

        self.norb = nmo
        self.ene0 = ene0 - self._base.mol.energy_nuc()
        self.vec0 = vec0
        self.amp0 = amp0

    def get_rhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0

        rhs_ip_list = []
        for p in orb_list:
            if p < nocc:
                rhs_ip = numpy.zeros((nocc + nocc * nocc * nvir,))
                rhs_ip[p] = 1.0

            else:
                rhs_ip = amplitudes_to_vector_ip(t1[:, p-nocc], t2[:, :, p-nocc, :])

            rhs_ip_list.append(rhs_ip)

        rhs_ip = numpy.array(rhs_ip_list)
        return rhs_ip

    def get_lhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        lhs_ip = numpy.asarray([fci.addons.des_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        lhs_ip = lhs_ip.reshape(len(orb_list), -1)

        return lhs_ip

    def get_rhs_ea(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0

        rhs_ea_list = []
        for q in orb_list:
            if q >= nocc:
                rhs_ea = numpy.zeros((nvir + nocc * nvir * nvir,))
                rhs_ea[q-nocc] = 1.0

            else:
                rhs_ea = amplitudes_to_vector_ea(-t1[q, :], -t2[q, :, :, :])

            rhs_ea_list.append(rhs_ea)

        rhs_ea = numpy.array(rhs_ea_list)
        return rhs_ea

    def get_lhs_ea(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        lhs_ea = numpy.asarray([fci.addons.cre_b(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        lhs_ea = lhs_ea.reshape(len(orb_list), -1)

        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        norb = self.norb
        nelec0 = self.nelec0
        nelec  = self._nelec_ip
        assert nelec[0] >= 0 and nelec[1] >= 0

        h1e = self._h1e
        eri = self._eri
        ene0 = self.ene0

        vec_hdiag = self._base.make_hdiag(h1e, eri, norb, nelec)
        vec_size = vec_hdiag.size

        hm = fci.direct_spin1.pspace(h1e, eri, norb, nelec, hdiag=vec_hdiag, np=vec_size)[1]
        assert hm.shape == (vec_size, vec_size)

        if vec_size * vec_size * 8 / 1024 ** 3 > self.max_memory:
            raise ValueError("Not enough memory for FCI Hamiltonian.")

        def hv0(omega, eta):
            hm0 = hm - ene0 * numpy.eye(vec_size)
            omega_eta = (omega - 1j * eta) * numpy.eye(vec_size)
            assert hm0.shape == (vec_size, vec_size)
            return hm0 + omega_eta

        return hv0, None

    def gen_hop_ea(self, verbose=None):
        norb = self.norb
        nelec0 = self.nelec0
        nelec  = self._nelec_ea
        assert nelec[0] <= norb and nelec[1] <= norb

        h1e = self._h1e
        eri = self._eri
        ene0 = self.ene0

        vec_hdiag = self._base.make_hdiag(h1e, eri, norb, nelec)
        vec_size = vec_hdiag.size

        hm = fci.direct_spin1.pspace(h1e, eri, norb, nelec, hdiag=vec_hdiag, np=vec_size)[1]
        assert hm.shape == (vec_size, vec_size)

        if vec_size * vec_size * 8 / 1024 ** 3 > self.max_memory:
            raise ValueError("Not enough memory for FCI Hamiltonian.")

        def gen_hv0(omega, eta):
            hm0 = hm - ene0 * numpy.eye(vec_size)
            omega_eta = (omega + 1j * eta) * numpy.eye(vec_size)
            assert hm0.shape == (vec_size, vec_size)
            return - hm0 + omega_eta

        return gen_hv0, None

def CCGF(hf_obj, method="slow"):
    if method.lower() == "slow":
        assert isinstance(hf_obj, pyscf.scf.hf.RHF)
        return CoupledClusterSingleDoubleSpin0Slow(hf_obj)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, scf

    m = gto.M(
        atom='H 0 0 0; Li 0 0 1.1',
        basis='sto3g',
        verbose=0,
    )
    rhf_obj = scf.RHF(m)
    rhf_obj.verbose = 0
    rhf_obj.kernel()

    cc_obj = cc.CCSD(rhf_obj)
    cc_obj.verbose = 4
    cc_obj.conv_tol = 1e-8
    cc_obj.conv_tol_normt = 1e-8
    ene_ccsd, t1, t2 = cc_obj.kernel()
    assert cc_obj.converged
    vec0 = cc_obj.amplitudes_to_vector(t1, t2)

    print("ene_ccsd = %12.8f" % ene_ccsd)

    eta = 0.01
    omega_list = numpy.linspace(-0.5, 0.5, 21)
    coeff = rhf_obj.mo_coeff
    nao, nmo = coeff.shape
    ps = [p for p in range(nmo)]
    qs = [q for q in range(nmo)]

    gfn_obj = CCGF(rhf_obj, method="slow")
    gfn_obj.conv_tol = 1e-8
    gfn_obj.build(vec0=vec0)

    rhs_ip_1 = gfn_obj.get_rhs_ip(ps)
    rhs_ea_1 = gfn_obj.get_rhs_ea(qs)

    try:
        from fcdmft.solver.ccgf import greens_b_vector_ip_rhf, greens_b_vector_ea_rhf
        rhs_ip_2 = [greens_b_vector_ip_rhf(cc_obj, p) for p in ps]
        rhs_ea_2 = [greens_b_vector_ea_rhf(cc_obj, q) for q in qs]
        rhs_ip_2 = numpy.array(rhs_ip_2)
        rhs_ea_2 = numpy.array(rhs_ea_2)

        err1 = numpy.linalg.norm(rhs_ip_1 - rhs_ip_2)
        err2 = numpy.linalg.norm(rhs_ea_1 - rhs_ea_2)

        # Not as small as I expected
        assert err1 < 1e-6, err1
        assert err2 < 1e-6, err2

    except ImportError:
        pass