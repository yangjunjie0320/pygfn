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

@profile
def greens_b_singles_ea_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        return -t1[p,:]
    else:
        p = p-nocc
        result = numpy.zeros((nvir,), dtype=ds_type)
        result[p] = 1.0
        return result

@profile
def greens_b_doubles_ea_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return -t2[p,:,:,:]
    else:
        return numpy.zeros((nocc,nvir,nvir), dtype=ds_type)

@profile
def greens_b_vector_ea_rhf(cc, p, vec0):
    return amplitudes_to_vector_ea(
        greens_b_singles_ea_rhf(cc.t1, p),
        greens_b_doubles_ea_rhf(cc.t2, p),
    )

@profile
def greens_b_singles_ip_rhf(t1, p):
    nocc, nvir = t1.shape
    ds_type = t1.dtype
    if p < nocc:
        result = numpy.zeros((nocc,), dtype=ds_type)
        result[p] = 1.0
        return result
    else:
        p = p-nocc
        return t1[:,p]

@profile
def greens_b_doubles_ip_rhf(t2, p):
    nocc, _, nvir, _ = t2.shape
    ds_type = t2.dtype
    if p < nocc:
        return numpy.zeros((nocc,nocc,nvir), dtype=ds_type)
    else:
        p = p-nocc
        return t2[:,:,p,:]

@profile
def greens_b_vector_ip_rhf(cc, p, vec0):
    return amplitudes_to_vector_ip(
        greens_b_singles_ip_rhf(cc.t1, p),
        greens_b_doubles_ip_rhf(cc.t2, p),
    )




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

    @profile
    def get_rhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0

        def gen_rhs_occ(i):
            rhs_ip_1 = numpy.zeros((nocc,))
            rhs_ip_2 = numpy.zeros((nocc, nocc, nvir))
            rhs_ip_1[i] = 1.0
            return amplitudes_to_vector_ip(rhs_ip_1, rhs_ip_2)

        def gen_rhs_vir(a):
            rhs_ip_1 = t1[:, a-nocc]
            rhs_ip_2 = t2[:, :, a-nocc, :]
            return amplitudes_to_vector_ip(rhs_ip_1, rhs_ip_2)

        mask_occ = orb_list < nocc
        mask_vir = ~mask_occ
        rhs_ip_occ = [gen_rhs_occ(i) for i in orb_list[mask_occ]]
        rhs_ip_vir = [gen_rhs_vir(a) for a in orb_list[mask_vir]]
        rhs_ip = numpy.array(rhs_ip_occ + rhs_ip_vir)
        return rhs_ip

    def get_lhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        lhs_ip = numpy.asarray([fci.addons.des_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        lhs_ip = lhs_ip.reshape(len(orb_list), -1)

        return lhs_ip

    @profile
    def get_rhs_ea(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0

        def gen_rhs_occ(i):
            rhs_ea_1 = -t1[i, :]
            rhs_ea_2 = -t2[i, :, :, :]
            return amplitudes_to_vector_ea(rhs_ea_1, rhs_ea_2)

        def gen_rhs_vir(a):
            rhs_ea_1 = numpy.zeros((nvir,))
            rhs_ea_2 = numpy.zeros((nocc, nvir, nvir))
            rhs_ea_1[a-nocc] = 1.0
            return amplitudes_to_vector_ea(rhs_ea_1, rhs_ea_2)

        mask_occ = orb_list < nocc
        mask_vir = ~mask_occ
        rhs_ea_occ = [gen_rhs_occ(i) for i in orb_list[mask_occ]]
        rhs_ea_vir = [gen_rhs_vir(a) for a in orb_list[mask_vir]]
        rhs_ea = numpy.array(rhs_ea_occ + rhs_ea_vir)
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
        atom='''
          H    2.1489399    1.2406910    0.0000000
          C    1.2116068    0.6995215    0.0000000
          C    1.2116068   -0.6995215    0.0000000
          H    2.1489399   -1.2406910    0.0000000
          C   -0.0000000   -1.3990430   -0.0000000
          H   -0.0000000   -2.4813820   -0.0000000
          C   -1.2116068   -0.6995215   -0.0000000
          H   -2.1489399   -1.2406910   -0.0000000
          C   -1.2116068    0.6995215   -0.0000000
          H   -2.1489399    1.2406910   -0.0000000
          C    0.0000000    1.3990430    0.0000000
          H    0.0000000    2.4813820    0.0000000
        ''', basis='ccpvdz', verbose=0,
    )

    rhf_obj = scf.RHF(m)
    rhf_obj.verbose = 0
    rhf_obj.kernel()

    cc_obj = cc.CCSD(rhf_obj)
    cc_obj.verbose = 4
    cc_obj.conv_tol = 1e-2
    cc_obj.conv_tol_normt = 1e-2
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

    import time

    time0 = time.time()
    rhs_ip_1 = gfn_obj.get_rhs_ip(ps)
    rhs_ea_1 = gfn_obj.get_rhs_ea(qs)
    print("Time for get_rhs_ip and get_rhs_ea: %6.4e" % (time.time() - time0))

    try:
        import fcdmft.solver.ccgf

        time0 = time.time()
        rhs_ip_2 = [greens_b_vector_ip_rhf(cc_obj, p, vec0) for p in ps]
        rhs_ea_2 = [greens_b_vector_ea_rhf(cc_obj, q, vec0) for q in qs]
        rhs_ip_2 = numpy.array(rhs_ip_2)
        rhs_ea_2 = numpy.array(rhs_ea_2)
        print("Time for fcdmft.solver.ccgf: %6.4e" % (time.time() - time0))

        err1 = numpy.linalg.norm(rhs_ip_1 - rhs_ip_2) / rhs_ip_1.size
        err2 = numpy.linalg.norm(rhs_ea_1 - rhs_ea_2) / rhs_ea_1.size

        # Not as small as I expected
        assert err1 < 1e-6
        assert err2 < 1e-6

    except ImportError:
        pass