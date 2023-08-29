import numpy
import scipy

import pyscf
from pyscf import fci, lib
from pyscf import __config__

from pygfn.lib import gmres
from pyscf.fci.direct_spin1 import contract_1e
from pyscf.fci.direct_spin1 import contract_2e


def _unpack_pq(ps, qs, norb):
    if ps is None:
        ps = range(norb)
    if qs is None:
        qs = range(norb)

    ps = numpy.asarray(ps)
    qs = numpy.asarray(qs)
    np = ps.size
    nq = qs.size

    return ps, qs, np, nq


class GreensFunctionMixin(lib.StreamObject):
    """
    Green's function base class

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default equals to :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-9
        max_cycle : int
            max number of iterations.  If max_cycle <= 0, SCF iteration will
            be skiped and the kernel function will compute only the total
            energy based on the intial guess. Default value is 50.
        gmres_m : int
            m used in GMRES method.
    """
    verbose = getattr(__config__, 'gf_verbose', 4)
    max_memory = getattr(__config__, 'gf_max_memory', 40000)  # 2 GB
    conv_tol = getattr(__config__, 'gf_conv_tol', 1e-6)
    max_cycle = getattr(__config__, 'gf_max_cycle', 50)
    gmres_m = getattr(__config__, 'gf_gmres_m', 30)

    norb = None
    coeff = None

    nelec0 = None
    ene0 = None
    vec0 = None
    amp0 = None

    def __init__(self) -> None:
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def get_rhs_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_rhs_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lhs_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lhs_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def gen_hop_ip(self, verbose=None):
        raise NotImplementedError

    def gen_hop_ea(self, verbose=None):
        raise NotImplementedError

    def solve_gfn_ip(self, omegas, ps=None, qs=None, eta=0.01, verbose=None):
        res = self.solve_gfn(omegas, ps=ps, qs=qs, eta=eta, comp="ip", verbose=verbose)
        return res

    def solve_gfn_ea(self, omegas, ps=None, qs=None, eta=0.01, verbose=None):
        res = self.solve_gfn(omegas, ps=ps, qs=qs, eta=eta, comp="ea", verbose=verbose)
        return res

    def solve_gfn(self, omegas, ps=None, qs=None, eta=0.01, comp="ip", verbose=None):
        r"""Solves Green's function in the frequency domain, represented by the following mathematical expressions:

            .. math::
                G^{\mathrm{IP}}_{pq}(\omega) = \langle L | a^{\dagger}_q \left[\omega+(H - E_0)+\mathrm{i} \eta\right]^{-1} a_p |R\rangle

            .. math::
                G^{\mathrm{EA}}_{pq}(\omega) = \langle L | a_p \left[\omega-(H - E_0)+\mathrm{i} \eta\right]^{-1} a^{\dagger}_q |R\rangle

            .. math::
                G_{pq}(\omega) = G^{\mathrm{IP}}_{pq}(\omega) + G^{\mathrm{EA}}_{pq}(\omega)
                               =  X_{\mu p} (\omega) L_{\mu q} +  Y_{\nu q} (\omega) L_{\nu p}

            Parameters:
            ----------
            omega : list of float
                Frequencies at which to evaluate the Green's function.
            ps : list of int
                List of 'p' indices for the function.
            qs : list of int
                List of 'q' indices for the function.
            eta : float
                A constant used in the Green's function equations.
            comp : str, optional
                Specifies the component of the Green's function to compute. Must be one of 'ip' or 'ea'.
            method : str, optional
                Method used to solve the Green's function.
            verbose : int, optional
                The level of logging or printing during execution. Higher values indicate more detailed output.

            Returns:
            -------
            res : numpy.ndarray
                The computed Green's function values for the specified inputs.
        """
        log = lib.logger.new_logger(self, verbose)

        comp = comp.lower()
        assert comp.lower() in ["ip", "ea"]

        norb = self.norb
        ps, qs, np, nq = _unpack_pq(ps, qs, norb)

        nelec0 = self.nelec0
        assert nelec0[0] >= nelec0[1]

        if comp == "ip":  # IP
            vec_rhs = self.get_rhs_ip(qs, verbose=verbose)
            vec_lhs = self.get_lhs_ip(ps, verbose=verbose)
            gen_hv0, gen_hd0 = self.gen_hop_ip(verbose=verbose)

        else:  # EA
            vec_rhs = self.get_rhs_ea(ps, verbose=verbose)
            vec_lhs = self.get_lhs_ea(qs, verbose=verbose)
            gen_hv0, gen_hd0 = self.gen_hop_ea(verbose=verbose)

        vec_size = vec_rhs.shape[1]
        assert vec_rhs.shape[1] == vec_size
        assert vec_lhs.shape[1] == vec_size

        if gen_hd0 is None:
            def gen_gfn(omega):
                hv0 = gen_hv0(omega, eta)
                vec_x = numpy.linalg.solve(hv0, vec_rhs.T)
                gfn = numpy.dot(vec_lhs, vec_x)

                gfn = gfn if comp == "ip" else gfn.T
                assert gfn.shape == (np, nq)
                return gfn

        else:
            def gen_gfn(omega):
                hv0 = gen_hv0(omega, eta)
                hd0 = gen_hd0(omega, eta)

                vec_x = gmres(
                    hv0, vec_rhs, x0=(vec_rhs / hd0), d=hd0,
                    tol=self.conv_tol, max_cycle=self.max_cycle,
                    m=self.gmres_m, verbose=log
                )
                vec_x = vec_x.T
                gfn = numpy.dot(vec_lhs, vec_x)

                gfn = gfn if comp == "ip" else gfn.T
                assert gfn.shape == (np, nq)
                return gfn

        res = numpy.array([gen_gfn(omega) for omega in omegas])
        assert res.shape == (len(omegas), np, nq)
        return res

    def kernel(self, omegas, eta=1e-2, ps=None, qs=None, coeff=None, vec0=None, verbose=None):
        coeff = coeff if coeff is not None else self.coeff
        assert coeff is not None
        self.build(coeff=coeff, vec0=vec0)

        assert self.ene0 is not None
        assert self.vec0 is not None

        gfn_ip = self.solve_gfn_ip(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)
        gfn_ea = self.solve_gfn_ea(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)

        gfn = gfn_ip + gfn_ea
        return gfn


class FullConfigurationInteractionSlow(GreensFunctionMixin):
    def __init__(self, mol_or_mf_obj, coeff=None):
        self._base = fci.FCI(mol_or_mf_obj, mo=coeff)
        self.coeff = coeff

    def build(self, coeff=None, vec0=None):
        assert coeff is not None
        self._base = fci.FCI(self._base.mol, mo=coeff)
        ene0, vec0 = self._base.kernel(ci0=vec0)

        nelec0 = self._base.nelec
        assert nelec0[0] >= nelec0[1]
        nelec_ip = (nelec0[0] - 1, nelec0[1])
        nelec_ea = (nelec0[0], nelec0[1] + 1)
        self.nelec0 = nelec0
        self._nelec_ip = nelec_ip
        self._nelec_ea = nelec_ea

        import inspect
        kwargs = inspect.signature(self._base.kernel).parameters
        norb = kwargs["norb"].default
        norb2 = (norb + 1) * norb // 2
        h1e = kwargs["h1e"].default
        eri = kwargs["eri"].default

        assert h1e.shape == (norb, norb)
        assert eri.shape == (norb2, norb2)

        self.norb = norb
        self.ene0 = ene0 - self._base.mol.energy_nuc()
        self.vec0 = vec0
        self._h1e = h1e
        self._eri = eri

    def get_rhs_ip(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        rhs_ip = numpy.asarray([fci.addons.des_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        rhs_ip = rhs_ip.reshape(len(orb_list), -1)

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
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        rhs_ea = numpy.asarray([fci.addons.cre_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        rhs_ea = rhs_ea.reshape(len(orb_list), -1)

        return rhs_ea

    def get_lhs_ea(self, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self.nelec0
        vec0 = self.vec0

        lhs_ea = numpy.asarray([fci.addons.cre_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
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


class FullConfigurationInteractionDirectSpin1(FullConfigurationInteractionSlow):
    def gen_hop_ip(self, verbose=None):
        norb = self.norb
        nelec0 = self.nelec0
        nelec  = self._nelec_ip
        assert nelec[0] >= 0 and nelec[1] >= 0

        h1e = self._h1e
        eri = self._eri
        ene0 = self.ene0

        vec_hdiag = self._base.make_hdiag(h1e, eri, norb, nelec)
        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])

        vec_size = vec_hdiag.size
        assert na * nb == vec_size

        h2e = self._base.absorb_h1e(h1e, eri, norb, nelec, .5)

        def gen_hv0(omega, eta):
            def hv0(v):
                c = v.reshape(na, nb)
                hc_real = contract_2e(h2e, c.real, norb, nelec)
                hc_imag = contract_2e(h2e, c.imag, norb, nelec)

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

        h1e = self._h1e
        eri = self._eri
        ene0 = self.ene0

        vec_hdiag = self._base.make_hdiag(h1e, eri, norb, nelec)
        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])

        vec_size = vec_hdiag.size
        assert na * nb == vec_size

        h2e = self._base.absorb_h1e(h1e, eri, norb, nelec, .5)

        def gen_hv0(omega, eta):
            def hv0(v):
                c = v.reshape(na, nb)
                hc_real = contract_2e(h2e, c.real, norb, nelec)
                hc_imag = contract_2e(h2e, c.imag, norb, nelec)

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


def FCIGF(mol_or_mf_obj, coeff=None, method="slow"):
    if method.lower() == "slow":
        return FullConfigurationInteractionSlow(mol_or_mf_obj, coeff=coeff)

    elif method.lower() == "direct":
        return FullConfigurationInteractionDirectSpin1(mol_or_mf_obj, coeff=coeff)

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

    fci_obj = fci.FCI(rhf_obj)
    ene_fci, vec_fci = fci_obj.kernel()

    eta = 0.01
    omega_list = numpy.linspace(-0.5, 0.5, 21)
    coeff = rhf_obj.mo_coeff
    nao, nmo = coeff.shape
    ps = [0, 1]
    qs = [0, 1, 2, 3]

    gfn_obj = FCIGF(m, coeff=coeff, method="direct")
    gfn1 = gfn_obj.kernel(omega_list, eta=eta, ps=ps, qs=qs)
    gen_hv0_ip_1, gen_hd0_ip = gfn_obj.gen_hop_ip()
    gen_hv0_ea_1, gen_hd0_ea = gfn_obj.gen_hop_ea()

    gfn_obj = FCIGF(m, coeff=coeff, method="slow")
    gfn2 = gfn_obj.kernel(omega_list, eta=eta, ps=ps, qs=qs)
    gen_hv0_ip_2, _ = gfn_obj.gen_hop_ip()
    gen_hv0_ea_2, _ = gfn_obj.gen_hop_ea()

    assert numpy.linalg.norm(gfn1 - gfn2) < 1e-4

    for omega in omega_list:
        vec_hd0_ip = gen_hd0_ip(omega, eta)
        vec_hd0_ea = gen_hd0_ea(omega, eta)

        hv0_ip_1 = numpy.asarray([gen_hv0_ip_1(omega, eta)(x) for x in numpy.eye(vec_hd0_ip.size)])
        hv0_ea_1 = numpy.asarray([gen_hv0_ea_1(omega, eta)(x) for x in numpy.eye(vec_hd0_ea.size)])

        hv0_ip_2 = gen_hv0_ip_2(omega, eta)
        hv0_ea_2 = gen_hv0_ea_2(omega, eta)

        err1 = numpy.linalg.norm(hv0_ip_1 - hv0_ip_2)
        err2 = numpy.linalg.norm(hv0_ea_1 - hv0_ea_2)

        assert err1 < 1e-10
        assert err2 < 1e-10
