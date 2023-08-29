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


class CoupledClusterSingleDoubleSpin0Direct(GreensFunctionMixin):
    is_approx_lambda = True

    def __init__(self, hf_obj):
        self._base: cc.ccsd.RCCSD
        self._base = cc.CCSD(hf_obj)
        self._base.verbose = 0

    def build(self, vec0=None):
        coeff = self._base._scf.mo_coeff
        assert coeff is not None
        nao, nmo = coeff.shape
        occ = self._base._scf.get_occ(mo_coeff=coeff)

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

        if self.is_approx_lambda:
            l1, l2 = t1, t2
        else:
            l1, l2 = self._base.solve_lambda(eris=eris, t1=t1, t2=t2)
        lam0 = (l1, l2)
        self._base.l1 = l1
        self._base.l2 = l2

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
        self.lam0 = lam0

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
                rhs_ip = amplitudes_to_vector_ip(t1[:, p - nocc], t2[:, :, p - nocc, :])

            rhs_ip_list.append(rhs_ip)

        rhs_ip = numpy.array(rhs_ip_list)
        return rhs_ip

    def get_lhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0
        l1, l2 = self.lam0

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

            lhs_ip = amplitudes_to_vector_ip(lhs_ip_1, lhs_ip_2)
            lhs_ip_list.append(lhs_ip)

        lhs_ip = numpy.array(lhs_ip_list)
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
                rhs_ea[q - nocc] = 1.0

            else:
                rhs_ea = amplitudes_to_vector_ea(-t1[q, :], -t2[q, :, :, :])

            rhs_ea_list.append(rhs_ea)

        rhs_ea = numpy.array(rhs_ea_list)
        return rhs_ea

    def get_lhs_ea(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = numpy.array(orb_list if orb_list is not None else range(norb))

        nocc = self._base.nocc
        nvir = norb - nocc
        t1, t2 = self.amp0
        l1, l2 = self.lam0

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

            lhs_ea = amplitudes_to_vector_ea(lhs_ea_1, lhs_ea_2)
            lhs_ea_list.append(lhs_ea)

        lhs_ea = numpy.array(lhs_ea_list)
        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        norb = self.norb
        nelec0 = self.nelec0
        nelec  = self._nelec_ip
        assert nelec[0] >= 0 and nelec[1] >= 0

        ene0 = self.ene0
        eom_obj = pyscf.cc.eom_rccsd.EOMIP(self._base)
        imds = eom_obj.make_imds(eris=self._eris)

        vec_hdiag = eom_obj.get_diag(imds=imds)
        vec_size = vec_hdiag.size

        def gen_hv0(omega, eta):
            def hv0(v):
                c = v
                hc_real = eom_obj.matvec(c.real)
                hc_imag = eom_obj.matvec(c.imag)

                hc0 = hc_real + 1j * hc_imag - ene0 * c
                omega_eta = (omega - 1j * eta) * c
                return (hc0 + omega_eta).reshape(-1)

            return hv0

        def gen_hd0(omega, eta):
            vec_h0 = vec_hdiag - ene0
            omega_eta = (omega - 1j * eta)
            assert vec_h0.shape == (vec_size,)
            return vec_h0 + omega_eta

        return gen_hv0, gen_hd0

    def gen_hop_ea(self, verbose=None):
        norb = self.norb
        nelec0 = self.nelec0
        nelec  = self._nelec_ea
        assert nelec[0] <= norb and nelec[1] <= norb

        ene0 = self.ene0
        eom_obj = pyscf.cc.eom_rccsd.EOMEA(self._base)
        imds = eom_obj.make_imds(eris=self._eris)

        vec_hdiag = eom_obj.get_diag(imds=imds)
        vec_size = vec_hdiag.size

        def gen_hv0(omega, eta):
            def hv0(v):
                c = v
                hc_real = eom_obj.matvec(c.real)
                hc_imag = eom_obj.matvec(c.imag)

                hc0 = hc_real + 1j * hc_imag - ene0 * c
                omega_eta = (omega + 1j * eta) * c
                return (- hc0 + omega_eta).reshape(-1)

            return hv0

        def gen_hd0(omega, eta):
            vec_h0 = vec_hdiag - ene0
            omega_eta = (omega + 1j * eta)
            assert vec_h0.shape == (vec_size,)
            return -vec_h0 + omega_eta

        return gen_hv0, gen_hd0


def CCGF(hf_obj, method="direct"):
    if method.lower() == "direct":
        assert isinstance(hf_obj, pyscf.scf.hf.RHF)
        return CoupledClusterSingleDoubleSpin0Direct(hf_obj)

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

    gfn_obj = CCGF(rhf_obj, method="direct")
    gfn_obj.conv_tol = 1e-8
    gfn_obj.build(vec0=vec0)

    rhs_ip_1 = gfn_obj.get_rhs_ip(ps)
    rhs_ea_1 = gfn_obj.get_rhs_ea(qs)
    lhs_ip_1 = gfn_obj.get_lhs_ip(ps)
    lhs_ea_1 = gfn_obj.get_lhs_ea(qs)
    gfn1_ip, gfn1_ea = gfn_obj.kernel(omega_list, ps=ps, qs=qs, eta=eta)

    try:
        import fcdmft.solver.ccgf
        from fcdmft.solver.ccgf import greens_b_vector_ip_rhf, greens_b_vector_ea_rhf
        from fcdmft.solver.ccgf import greens_e_vector_ip_rhf, greens_e_vector_ea_rhf

        rhs_ip_2 = [greens_b_vector_ip_rhf(gfn_obj._base, p) for p in ps]
        rhs_ea_2 = [greens_b_vector_ea_rhf(gfn_obj._base, p) for p in ps]
        rhs_ip_2 = numpy.array(rhs_ip_2)
        rhs_ea_2 = numpy.array(rhs_ea_2)

        err1 = numpy.linalg.norm(rhs_ip_1 - rhs_ip_2)
        err2 = numpy.linalg.norm(rhs_ea_1 - rhs_ea_2)

        # Not as small as I expected
        assert err1 < 1e-6, err1
        assert err2 < 1e-6, err2

        lhs_ip_2 = [greens_e_vector_ip_rhf(gfn_obj._base, p) for p in ps]
        lhs_ea_2 = [greens_e_vector_ea_rhf(gfn_obj._base, p) for p in ps]
        lhs_ip_2 = numpy.array(lhs_ip_2)
        lhs_ea_2 = numpy.array(lhs_ea_2)

        err1 = numpy.linalg.norm(lhs_ip_1 - lhs_ip_2)
        err2 = numpy.linalg.norm(lhs_ea_1 - lhs_ea_2)

        # Not as small as I expected
        assert err1 < 1e-6, err1
        assert err2 < 1e-6, err2

        gfn_obj = fcdmft.solver.ccgf.CCGF(gfn_obj._base)
        gfn_obj.conv_tol = 1e-8
        gfn2_ip, gfn2_ea = gfn_obj.get_gf(ps, qs, omega_list, broadening=eta)
        gfn2_ip = gfn2_ip.transpose(2, 0, 1)
        gfn2_ea = gfn2_ea.transpose(2, 0, 1)

        print("gfn1_ip = \n", gfn1_ip.shape)
        print("gfn2_ip = \n", gfn2_ip.shape)

        assert numpy.linalg.norm(gfn1_ip - gfn2_ip) < 1e-6, numpy.linalg.norm(gfn1_ip - gfn2_ip)
        assert numpy.linalg.norm(gfn1_ea - gfn2_ea) < 1e-6, numpy.linalg.norm(gfn1_ea - gfn2_ea)

        print("All tests passed!")

    except ImportError:
        pass
