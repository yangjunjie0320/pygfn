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

def _pack(h, omega, eta, v=None, comp="ip"):
    assert h.shape == v.shape
    pm = 1.0 if comp == "ip" else -1.0
    omega_eta = (omega - pm * eta * 1.0j)
    return pm * h + omega_eta * v

def _gen_hop_slow(gfn_obj, comp="ip", verbose=None):
    norb = gfn_obj.norb
    nelec0 = gfn_obj.nelec0
    nelec  = gfn_obj._nelec_ip if comp == "ip" else gfn_obj._nelec_ea

    h1e = gfn_obj._h1e
    eri = gfn_obj._eri
    ene0 = gfn_obj.ene0

    vec_hdiag = gfn_obj._base.make_hdiag(h1e, eri, norb, nelec)
    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])

    vec_size = vec_hdiag.size
    assert na * nb == vec_size

    hm = fci.direct_spin1.pspace(h1e, eri, norb, nelec, hdiag=vec_hdiag, np=vec_size)[1]
    im = numpy.eye(vec_size)
    assert hm.shape == (vec_size, vec_size)

    if vec_size * vec_size * 8 / 1024 ** 3 > gfn_obj.max_memory:
        raise ValueError("Not enough memory for FCI Hamiltonian.")

    def hv0(omega, eta):
        hm0 = hm - ene0 * im
        return _pack(hm0, omega, eta, v=im, comp=comp)

    return hv0, None

def _gen_hop_direct(gfn_obj, comp="ip", verbose=None):
    assert comp in ["ip", "ea"]

    norb = gfn_obj.norb
    nelec = gfn_obj._nelec_ip if comp == "ip" else gfn_obj._nelec_ea
    assert nelec[0] >= 0 and nelec[1] >= 0
    assert nelec[0] <= norb and nelec[1] <= norb

    h1e = gfn_obj._h1e
    eri = gfn_obj._eri
    ene0 = gfn_obj.ene0

    vec_hdiag = gfn_obj._base.make_hdiag(h1e, eri, norb, nelec)
    vec_size = vec_hdiag.size
    vec_ones = numpy.ones(vec_size)

    na = fci.cistring.num_strings(norb, nelec[0])
    nb = fci.cistring.num_strings(norb, nelec[1])
    assert na * nb == vec_size

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

    nsite  = None
    norb = None

    _nelec0 = None
    _nelec_ip = None
    _nelec_ea = None

    ene0 = None
    vec0 = None
    amp0 = None

    def __init__(self) -> None:
        raise NotImplementedError

    def build(self, vec0=None):
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

        nelec0 = self._nelec0
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

        if gen_hd0 is None: # Slow: build the full Hamiltonian matrix
            def gen_gfn(omega):
                hv0 = gen_hv0(omega, eta)
                vec_x = numpy.linalg.solve(hv0, vec_rhs.T)
                gfn = numpy.dot(vec_lhs, vec_x)

                gfn = gfn if comp == "ip" else gfn.T
                assert gfn.shape == (np, nq)
                return gfn

        else: # Direct: use GMRES to solve the linear equations
            def gen_gfn(omega):
                hv0 = gen_hv0(omega, eta)
                hd0 = gen_hd0(omega, eta)

                vec_x = gmres(
                    hv0, vec_rhs, x0=(vec_rhs / hd0), d=hd0,
                    tol=self.conv_tol, max_cycle=self.max_cycle,
                    m=self.gmres_m, verbose=log
                )
                vec_x = vec_x.reshape(*vec_rhs.shape)

                gfn = numpy.einsum("iI,jI -> ij", vec_lhs, vec_x)
                if comp == "ip":
                    assert gfn.shape == (np, nq)
                else:
                    assert gfn.shape == (nq, np)
                    gfn = gfn.T
                return gfn

        res = numpy.array([gen_gfn(omega) for omega in omegas])
        assert res.shape == (len(omegas), np, nq)
        return res

    def kernel(self, omegas, eta=1e-2, ps=None, qs=None, vec0=None, verbose=None):
        self.build(vec0=vec0)

        assert self.ene0 is not None
        assert self.vec0 is not None

        gfn_ip = self.solve_gfn_ip(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)
        gfn_ea = self.solve_gfn_ea(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)

        return (gfn_ip, gfn_ea)

def is_build(gf_obj):
    is_build = True
    is_build = is_build and (gf_obj.ene0 is not None)
    is_build = is_build and (gf_obj.vec0 is not None)
    is_build = is_build and (gf_obj.norb is not None)
    is_build = is_build and (gf_obj.nsite is not None)
    is_build = is_build and (gf_obj._h1e is not None)
    is_build = is_build and (gf_obj._eri is not None)
    is_build = is_build and (gf_obj._nelec0 is not None)
    is_build = is_build and (gf_obj._nelec_ip is not None)
    is_build = is_build and (gf_obj._nelec_ea is not None)
    return is_build

class FullConfigurationInteractionSlow(GreensFunctionMixin):
    _h1e = None
    _eri = None
    def __init__(self, hf_obj=None, nelec=None, h1e=None, eri=None):
        self._base = fci.FCI(hf_obj, mo=None)
        self._base.mf = hf_obj

        self._nelec0 = nelec
        self._h1e = h1e
        self._eri = eri

    def build(self, vec0=None):
        if is_build(self):
            return None

        mf = self._base.mf
        assert mf is not None, "mf is not given"

        coeff = self._base.mf.mo_coeff
        assert coeff is not None

        self._base = fci.FCI(self._base.mol, mo=coeff)
        self._base.mf = mf
        ene0, vec0 = self._base.kernel(ci0=vec0)

        nelec0 = self._base.nelec
        assert nelec0[0] >= nelec0[1]
        nelec_ip = (nelec0[0] - 1, nelec0[1])
        nelec_ea = (nelec0[0], nelec0[1] + 1)
        self._nelec0 = nelec0
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

        self.nsite = norb

    def get_rhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self._nelec0
        vec0  = self.vec0 if vec0 is None else vec0

        print(vec0.shape)

        rhs_ip = numpy.asarray([fci.addons.des_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        rhs_ip = rhs_ip.reshape(len(orb_list), -1)

        return rhs_ip

    def get_lhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self._nelec0
        vec0 = self.vec0 if vec0 is None else vec0

        lhs_ip = numpy.asarray([fci.addons.des_a(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        lhs_ip = lhs_ip.reshape(len(orb_list), -1)

        return lhs_ip

    def get_rhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self._nelec0
        vec0 = self.vec0 if vec0 is None else vec0

        rhs_ea = numpy.asarray([fci.addons.cre_b(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        rhs_ea = rhs_ea.reshape(len(orb_list), -1)

        return rhs_ea

    def get_lhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)

        nelec0 = self._nelec0
        vec0 = self.vec0 if vec0 is None else vec0

        lhs_ea = numpy.asarray([fci.addons.cre_b(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        lhs_ea = lhs_ea.reshape(len(orb_list), -1)

        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        return _gen_hop_slow(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_slow(self, comp="ea", verbose=verbose)

class FullConfigurationInteractionDirectSpin1(FullConfigurationInteractionSlow):
    def gen_hop_ip(self, verbose=None):
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)

def FCIGF(hf_obj, method="slow"):
    if method.lower() == "slow":
        return FullConfigurationInteractionSlow(hf_obj)

    elif method.lower() == "direct":
        return FullConfigurationInteractionDirectSpin1(hf_obj)

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
    nao, nmo = rhf_obj.mo_coeff.shape
    ps = [0, 1]
    qs = [q for q in range(nmo)]

    gfn_obj = FCIGF(rhf_obj, method="direct")
    gfn_obj.conv_tol = 1e-8
    gfn_obj.build(vec0=vec_fci)

    import time
    time0 = time.time()
    gfn1_ip, gfn1_ea = gfn_obj.kernel(omega_list, eta=eta, ps=ps, qs=qs)
    time1 = time.time()
    print("time = %12.8f" % (time1 - time0))

    gfn_obj = FCIGF(rhf_obj, method="slow")
    gfn_obj.conv_tol = 1e-8
    gfn2_ip, gfn2_ea = gfn_obj.kernel(omega_list, eta=eta, ps=ps, qs=qs)

    assert numpy.linalg.norm(gfn1_ip - gfn2_ip) < 1e-6
    assert numpy.linalg.norm(gfn1_ea - gfn2_ea) < 1e-6

    try:
        import fcdmft.solver.fcigf
        gfn_obj = fcdmft.solver.fcigf.FCIGF(fci_obj, rhf_obj, tol=1e-8)

        import time
        time0 = time.time()
        gfn3_ip = gfn_obj.ipfci_mo(ps, qs, omega_list, eta).transpose(2, 0, 1)
        gfn3_ea = gfn_obj.eafci_mo(ps, qs, omega_list, eta).transpose(2, 0, 1)
        assert numpy.linalg.norm(gfn1_ip - gfn3_ip) < 1e-6
        assert numpy.linalg.norm(gfn1_ea - gfn3_ea) < 1e-6
        time1 = time.time()
        print("time = %12.8f" % (time1 - time0))

        print("All tests passed!")

    except ImportError:
        pass