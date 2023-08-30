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

from pygfn.fci import _pack
from pygfn.fci import GreensFunctionMixin
def _gen_hop_direct(gfn_obj, comp="ip", verbose=None):
    assert comp in ["ip", "ea"]

    norb = gfn_obj.norb
    nelec0 = gfn_obj.nelec0
    nelec  = gfn_obj._nelec_ip if comp == "ip" else gfn_obj._nelec_ea
    assert nelec[0] >= 0 and nelec[1] >= 0
    assert nelec[0] <= norb and nelec[1] <= norb

    from pyscf.cc.eom_rccsd import EOMIP, EOMEA
    eom_obj = EOMEA(gfn_obj._base) if comp == "ea" else EOMIP(gfn_obj._base)
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
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)


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
    cc_obj.verbose = 0
    cc_obj.conv_tol = 1e-8
    cc_obj.conv_tol_normt = 1e-8
    ene_ccsd, t1, t2 = cc_obj.kernel()
    assert cc_obj.converged
    vec0 = cc_obj.amplitudes_to_vector(t1, t2)

    print("ene_ccsd = %12.8f" % ene_ccsd)

    eta = 0.01
    omega_list = [0.0] # numpy.linspace(-0.5, 0.5, 21)
    coeff = rhf_obj.mo_coeff
    nao, nmo = coeff.shape
    ps = [p for p in range(nmo)]
    qs = [q for q in range(nmo)]

    gfn_obj = CCGF(rhf_obj, method="direct")
    gfn_obj.conv_tol = 1e-8
    gfn1_ip, gfn1_ea = gfn_obj.kernel(omega_list, ps=ps, qs=qs, eta=eta)

    try:
        import fcdmft.solver.ccgf
        gfn_obj = fcdmft.solver.ccgf.CCGF(gfn_obj._base)
        gfn_obj.tol = 1e-8
        gfn2_ip = -gfn_obj.ipccsd_mo(qs, ps, omega_list, eta)
        gfn2_ip = gfn2_ip.transpose(2, 1, 0)
        gfn2_ea = -gfn_obj.eaccsd_mo(qs, ps, omega_list, eta)
        gfn2_ea = gfn2_ea.transpose(2, 1, 0)
        assert numpy.linalg.norm(gfn1_ip - gfn2_ip) < 1e-6
        assert numpy.linalg.norm(gfn1_ea - gfn2_ea) < 1e-6

        print("All tests passed!")

    except ImportError:
        pass
