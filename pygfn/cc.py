import numpy, scipy

import pyscf
from pyscf import cc, lib

from pygfn.fci import _pack
from pygfn.fci import GreensFunctionMixin

def amplitudes_to_vector(amp, lam=None, gfn_obj=None, comp=None):
    assert (comp is None) or (comp in ["ip", "ea"])

    if comp == "ip":
        func = gfn_obj._base.eomip_method().amplitudes_to_vector
    elif comp == "ea":
        func = gfn_obj._base.eomea_method().amplitudes_to_vector
    else:
        func = gfn_obj._base.amplitudes_to_vector

    vec_amp = func(*amp)
    if lam is not None:
        vec_lam = func(*lam)
        vec = lib.tag_array(vec_amp, vec_lam=vec_lam)
    else:
        vec = vec_amp

    return vec

def vector_to_amplitudes(vec, gfn_obj=None, comp=None):
    assert (comp is None) or (comp in ["ip", "ea"])

    if comp == "ip":
        func = gfn_obj._base.eomip_method().vector_to_amplitudes
    elif comp == "ea":
        func = gfn_obj._base.eomea_method().vector_to_amplitudes
    else:
        func = gfn_obj._base.vector_to_amplitudes

    vec_amp = vec
    vec_lam = getattr(vec, "vec_lam", None)

    amp = func(vec_amp)
    lam = func(vec_lam) if vec_lam is not None else None

    return amp, lam

def _gen_hop_direct(gfn_obj, comp="ip", verbose=None):
    assert comp in ["ip", "ea"]

    norb = gfn_obj.norb
    nelec  = gfn_obj._nelec_ip if comp == "ip" else gfn_obj._nelec_ea
    assert nelec[0] >= 0 and nelec[1] >= 0
    assert nelec[0] <= norb and nelec[1] <= norb

    eom_obj = gfn_obj._base.eomea_method() if comp == "ea" else gfn_obj._base.eomip_method()
    imds = eom_obj.make_imds(eris=gfn_obj._eris)

    vec_hdiag = eom_obj.get_diag(imds=imds)
    vec_size = vec_hdiag.size
    vec_ones = numpy.ones(vec_size)

    def gen_hv0(omega, eta):
        def hv0(v):
            c = v
            hc_real = eom_obj.matvec(c.real, imds=imds)
            hc_imag = eom_obj.matvec(c.imag, imds=imds)
            hc0 = hc_real + 1j * hc_imag  # - ene0 * c
            return _pack(hc0.reshape(-1), omega, eta, v=v, comp=comp)
        return hv0

    def gen_hd0(omega, eta):
        return _pack(vec_hdiag, omega, eta, v=vec_ones, comp=comp)

    return gen_hv0, gen_hd0


class DirectCoupledClusterSingleDouble(GreensFunctionMixin):
    is_approx_lambda = True

    _h1e = None
    _eri = None
    _eris = None

    max_space = 100
    def __init__(self, m=None):
        if isinstance(m, pyscf.scf.hf.SCF):
            self._base = cc.CCSD(m)

        elif isinstance(m, pyscf.cc.ccsd.CCSD):
            self._base = m

        else:
            self._base = None

    def _is_build(self):
        is_build = True
        is_build = is_build and (self._base is not None)
        is_build = is_build and (self._eris is not None)
        is_build = is_build and (self.ene0 is not None)
        is_build = is_build and (self.vec0 is not None)

        is_build = is_build and (self.norb is not None)
        is_build = is_build and (self._nelec is not None)

        if is_build:
            log = lib.logger.new_logger(self, self.verbose)
            log.warn("GF object is already built, skipping.")

        return is_build

    def _check_sanity(self):
        norb = self.norb
        nelec = self._nelec
        assert nelec[0] >= 0 and nelec[1] >= 0

        ene0 = self.ene0

        vec0 = self.vec0
        # vec0 = vec0.reshape(-1)
        amp, lam = vector_to_amplitudes(vec0, gfn_obj=self)
        t1, t2 = amp
        l1, l2 = lam

        if self._h1e is not None:
            h1e = numpy.asarray(self._h1e)
            if h1e.ndim == 2:
                assert h1e.shape == (norb, norb)
            elif h1e.ndim == 3:
                assert h1e.shape == (2, norb, norb)

            assert self._eri is not None

    def build(self, amp=None, lam=None, coeff=None, occ=None, verbose=None):
        if self._base is None:
            # TODO: Implement this. Given the h1e and eri,
            # TODO: build the CCSD object.
            assert self._nelec is not None
            assert self._h1e is not None
            assert self._eri is not None
            raise NotImplementedError

        m = self._base.mol
        nelec = m.nelec
        nelec = sorted(nelec, reverse=True)
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0], nelec[1] + 1)
        self._nelec = nelec
        self._nelec_ip = nelec_ip
        self._nelec_ea = nelec_ea

        coeff = self._base._scf.mo_coeff if coeff is None else coeff
        occ = self._base._scf.mo_occ if occ is None else occ
        assert coeff is not None
        assert occ is not None
        nao, nmo = coeff.shape[-2:]
        assert numpy.sum(occ) == nelec[0] + nelec[1]

        # Rebuid the CCSD object.
        cc_obj = cc.CCSD(self._base._scf, mo_coeff=coeff, mo_occ=occ)
        nocc = cc_obj.nocc
        eris = cc_obj.ao2mo(mo_coeff=coeff)
        self._eris = eris

        t1, t2 = amp if amp is not None else (None, None)
        l1, l2 = lam if lam is not None else (None, None)

        e_corr, t1, t2 = self._base.kernel(eris=eris, t1=t1, t2=t2)
        assert self._base.converged

        ene0 = self._base.e_tot - self._base._scf.energy_nuc()

        if self.is_approx_lambda:
            l1, l2 = t1, t2
        else:
            l1, l2 = self._base.solve_lambda(eris=eris, t1=t1, t2=t2)
        self._base.l1 = l1
        self._base.l2 = l2

        vec0 = amplitudes_to_vector((t1, t2), (l1, l2), gfn_obj=self)

        self.norb = nmo
        self.ene0 = ene0 - self._base.mol.energy_nuc()
        self.vec0 = vec0

    def get_rhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        vec0  = self.vec0 if vec0 is None else vec0
        amp, lam = vector_to_amplitudes(vec0, gfn_obj=self)
        t1, t2 = amp
        l1, l2 = lam
        nocc, nvir = t1.shape

        rhs_ip_list = []
        for p in orb_list:
            if p < nocc:
                rhs_ip = numpy.zeros((nocc + nocc * nocc * nvir,))
                rhs_ip[p] = 1.0

            else:
                rhs_ip = amplitudes_to_vector((t1[:, p - nocc], t2[:, :, p - nocc, :]), gfn_obj=self, comp="ip")

            rhs_ip_list.append(rhs_ip)

        rhs_ip = numpy.array(rhs_ip_list)
        return rhs_ip

    def get_lhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        vec0  = self.vec0 if vec0 is None else vec0
        amp, lam = vector_to_amplitudes(vec0, gfn_obj=self)
        t1, t2 = amp
        l1, l2 = lam
        nocc, nvir = t1.shape

        lhs_ip_list = []
        for p in orb_list:
            if p < nocc:
                lhs_ip_1 = numpy.zeros((nocc,))
                lhs_ip_1[p] = -1.0

                lhs_ip_1 += numpy.einsum("ia,a->i", l1, t1[p, :])
                lhs_ip_1 += numpy.einsum("ilcd,lcd->i", l2, t2[p, :, :, :]) * 2
                lhs_ip_1 -= numpy.einsum('ilcd,ldc->i', l2, t2[p, :, :, :])

                lhs_ip_2 = numpy.zeros((nocc, nocc, nvir))
                lhs_ip_2[p, :, :] += -2 * l1
                lhs_ip_2[:, p, :] += l1
                lhs_ip_2 += 2 * numpy.einsum('c,ijcb->ijb', t1[p, :], l2)
                lhs_ip_2 -= numpy.einsum('c,jicb->ijb', t1[p, :], l2)

            else:
                lhs_ip_1 = -l1[:, p - nocc]
                lhs_ip_2 = -2 * l2[:, :, p - nocc, :] + l2[:, :, :, p - nocc]

            lhs_ip = amplitudes_to_vector((lhs_ip_1, lhs_ip_2), gfn_obj=self, comp="ip")
            lhs_ip_list.append(lhs_ip)

        lhs_ip = numpy.array(lhs_ip_list)
        return lhs_ip

    def get_rhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        vec0  = self.vec0 if vec0 is None else vec0
        amp, lam = vector_to_amplitudes(vec0, gfn_obj=self)
        t1, t2 = amp
        l1, l2 = lam
        nocc, nvir = t1.shape

        rhs_ea_list = []
        for q in orb_list:
            if q >= nocc:
                rhs_ea = numpy.zeros((nvir + nocc * nvir * nvir,))
                rhs_ea[q - nocc] = 1.0

            else:
                rhs_ea = amplitudes_to_vector((-t1[q, :], -t2[q, :, :, :]), gfn_obj=self, comp="ea")

            rhs_ea_list.append(rhs_ea)

        rhs_ea = numpy.array(rhs_ea_list)
        return rhs_ea

    def get_lhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        vec0  = self.vec0 if vec0 is None else vec0
        amp, lam = vector_to_amplitudes(vec0, gfn_obj=self)
        t1, t2 = amp
        l1, l2 = lam
        nocc, nvir = t1.shape

        lhs_ea_list = []
        for p in orb_list:
            if p < nocc:
                lhs_ea_1 = l1[p, :]
                lhs_ea_2 = 2 * l2[p, :, :, :] - l2[:, p, :, :]

            else:
                lhs_ea_1 = numpy.zeros((nvir,))
                lhs_ea_1[p - nocc] = -1.0
                lhs_ea_1 += numpy.einsum("ia,i->a", l1, t1[:, p - nocc])
                lhs_ea_1 += 2*numpy.einsum('klca,klc->a', l2, t2[:,:,:,p - nocc])
                lhs_ea_1 -= numpy.einsum('klca,lkc->a', l2, t2[:,:,:,p - nocc])

                lhs_ea_2 = numpy.zeros((nocc, nvir, nvir))
                lhs_ea_2[:, p - nocc, :] += -2 * l1
                lhs_ea_2[:, :, p - nocc] += l1
                lhs_ea_2 += 2 * numpy.einsum('k,jkba->jab', t1[:,p - nocc], l2)
                lhs_ea_2 -= numpy.einsum('k,jkab->jab', t1[:,p - nocc], l2)

            lhs_ea = amplitudes_to_vector((lhs_ea_1, lhs_ea_2), gfn_obj=self, comp="ea")
            lhs_ea_list.append(lhs_ea)

        lhs_ea = numpy.array(lhs_ea_list)
        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)

    def gen_hop_ip(self, verbose=None):
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)

def CCGF(hf_obj, method="direct"):
    if method.lower() == "direct":
        assert isinstance(hf_obj, pyscf.scf.hf.RHF)
        return DirectCoupledClusterSingleDouble(hf_obj)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    from pyscf import gto, scf

    m = gto.M(
        atom='H 0 0 0; Li 0 0 1.1',
        basis='ccpvqz',
        verbose=0,
    )
    rhf_obj = scf.RHF(m)
    rhf_obj.verbose = 0
    rhf_obj.kernel()

    cc_obj = cc.CCSD(rhf_obj)
    cc_obj.verbose = 0
    cc_obj.conv_tol = 1e-8
    cc_obj.conv_tol_normt = 1e-8
    eris = cc_obj.ao2mo()
    ene_ccsd, t1, t2 = cc_obj.kernel(eris=eris)
    cc_obj.solve_lambda(eris=eris, t1=t1, t2=t2)
    assert cc_obj.converged
    print("ene_ccsd = %12.8f" % ene_ccsd)

    eta = 0.01
    omega_list = numpy.linspace(-0.5, 0.5, 21)
    coeff = rhf_obj.mo_coeff
    nao, nmo = coeff.shape
    ps = [q for q in range(nmo)]
    qs = [0, 1]

    gfn_obj = CCGF(rhf_obj, method="direct")
    gfn_obj.conv_tol = 1e-8
    gfn_obj.build()

    import time
    time0 = time.time()
    gfn1_ip, gfn1_ea = gfn_obj.kernel(omega_list, ps=ps, qs=qs, eta=eta)
    time1 = time.time()
    print("time = %12.8f" % (time1 - time0))

    try:
        import fcdmft.solver.ccgf
        from fcdmft.solver.ccgf import greens_b_vector_ip_rhf
        from fcdmft.solver.ccgf import greens_b_vector_ea_rhf

        gfn_obj = fcdmft.solver.ccgf.CCGF(gfn_obj._base)
        gfn_obj.tol = 1e-8

        time0 = time.time()
        gfn2_ip = -gfn_obj.ipccsd_mo(qs, ps, omega_list, eta)
        gfn2_ip = gfn2_ip.transpose(2, 1, 0)
        gfn2_ea = -gfn_obj.eaccsd_mo(qs, ps, omega_list, eta)
        gfn2_ea = gfn2_ea.transpose(2, 1, 0)
        assert numpy.linalg.norm(gfn1_ip - gfn2_ip) < 1e-6
        assert numpy.linalg.norm(gfn1_ea - gfn2_ea) < 1e-6
        time1 = time.time()
        print("time = %12.8f" % (time1 - time0))

        print("All tests passed!")

    except ImportError:
        pass
