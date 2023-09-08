import numpy
import scipy

import pyscf
from pyscf import fci, lib
from pyscf import __config__

from pygfn.lib import gmres

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
    nelec = gfn_obj._nelec
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
            hc_real = gfn_obj._base.contract_2e(h2e, c.real, norb, nelec)
            hc_imag = gfn_obj._base.contract_2e(h2e, c.imag, norb, nelec)
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

    norb = None

    _nelec = None
    _nelec_ip = None
    _nelec_ea = None
    is_ip = True
    is_ea = True

    ene0 = None
    vec0 = None
    amp   = None
    lam  = None

    def __init__(self) -> None:
        raise NotImplementedError

    @property
    def nsite(self):
        """
        Number of sites. Alias of `norb`.
        """
        assert self.norb is not None
        return self.norb

    def _is_build(self):
        raise NotImplementedError

    def _check_sanity(self):
        raise NotImplementedError

    def build(self):
        """
        Build the Green's function object. Will run
        scf
        :return:
        """
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

        nelec = self._nelec
        assert nelec[0] >= nelec[1]

        if comp == "ip":  # IP
            vec_rhs = self.get_rhs_ip(orb_list=qs, verbose=verbose)
            vec_lhs = self.get_lhs_ip(orb_list=ps, verbose=verbose)
            gen_hv0, gen_hd0 = self.gen_hop_ip(verbose=verbose)

        else:  # EA
            vec_rhs = self.get_rhs_ea(orb_list=ps, verbose=verbose)
            vec_lhs = self.get_lhs_ea(orb_list=qs, verbose=verbose)
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

    def kernel(self, omegas, eta=1e-2, ps=None, qs=None, verbose=None, is_ip=None, is_ea=None):
        if not self._is_build():
            self.build()

        self._check_sanity()

        is_ip = is_ip if is_ip is not None else self.is_ip
        is_ea = is_ea if is_ea is not None else self.is_ea

        gfn_ip = None
        if is_ip:
            nelec = self._nelec
            nelec_ip = self._nelec_ip
            assert nelec_ip is not None
            assert nelec_ip[0] + nelec_ip[1] == nelec[0] + nelec[1] - 1
            assert nelec_ip[0] >= 0 and nelec_ip[1] >= 0
            gfn_ip = self.solve_gfn_ip(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)

        gfn_ea = None
        if is_ea:
            nelec = self._nelec
            nelec_ea = self._nelec_ea
            assert nelec_ea is not None
            assert nelec_ea[0] + nelec_ea[1] == nelec[0] + nelec[1] + 1
            assert nelec_ea[0] >= 0 and nelec_ea[1] >= 0
            gfn_ea = self.solve_gfn_ea(omegas, ps=ps, qs=qs, eta=eta, verbose=verbose)

        return (gfn_ip, gfn_ea)
class SlowFullConfigurationInteraction(GreensFunctionMixin):
    _h1e = None
    _eri = None
    max_space = 100
    def __init__(self, m=None):
        self._base = fci.FCI(m, mo=None)
        self._base.m = m

    def _is_build(self):
        is_build = True
        is_build = is_build and (self._base is not None)
        is_build = is_build and (self.ene0 is not None)
        is_build = is_build and (self.vec0 is not None)

        is_build = is_build and (self.norb is not None)
        is_build = is_build and (self._nelec is not None)

        is_build = is_build and (self._h1e is not None)
        is_build = is_build and (self._eri is not None)

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
        vec0 = vec0.reshape(-1)
        ndet_alph = fci.cistring.num_strings(norb, nelec[0])
        ndet_beta = fci.cistring.num_strings(norb, nelec[1])
        assert self._base is not None
        assert ene0 is not None
        assert vec0.shape == (ndet_alph * ndet_beta, )
        self.vec0 = vec0.reshape(-1)

        h1e = numpy.asarray(self._h1e)
        if h1e.ndim == 2:
            assert h1e.shape == (norb, norb)
        elif h1e.ndim == 3:
            assert h1e.shape == (2, norb, norb)

        assert self._eri is not None

    def build(self, ci0=None, coeff=None, verbose=None):
        m = self._base.mol
        assert m is not None, "mf is not given"

        nelec = m.nelec if hasattr(m, "nelec") else m.mol.nelec
        nelec = sorted(nelec, reverse=True)
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0], nelec[1] + 1)
        self._nelec = nelec
        self._nelec_ip = nelec_ip
        self._nelec_ea = nelec_ea

        if coeff is None:
            if hasattr(m, "mo_coeff"):
                coeff = m.mo_coeff
        assert coeff is not None, "coeff is not given"

        fci_obj = fci.FCI(m.mol, mo=coeff)
        fci_obj.max_cycle = self.max_cycle
        fci_obj.conv_tol = self.conv_tol

        import inspect
        kwargs = inspect.signature(self._base.kernel).parameters
        norb = kwargs["norb"].default
        h1e = kwargs["h1e"].default
        eri = kwargs["eri"].default

        ene0, vec0 = fci_obj.kernel(
            norb=norb, nelec=nelec, h1e=h1e, eri=eri,
            ci0=ci0, verbose=verbose, tol=self.conv_tol,
            max_cycle=self.max_cycle,
            max_space=self.max_space,
            ecore=0.0
        )

        self._base = fci_obj
        self.norb = norb
        self.ene0 = ene0
        self.vec0 = vec0
        self._h1e = h1e
        self._eri = eri

    def get_rhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        nelec_ip = self._nelec_ip
        assert nelec_ip[0] + nelec_ip[1] == nelec[0] + nelec[1] - 1
        vec0  = self.vec0 if vec0 is None else vec0

        if nelec_ip[0] == nelec[0] - 1:
            des_op = fci.addons.des_a
        else:
            des_op = fci.addons.des_b

        rhs_ip = numpy.asarray([des_op(vec0, norb, nelec, p).reshape(-1) for p in orb_list])
        rhs_ip = rhs_ip.reshape(len(orb_list), -1)

        return rhs_ip

    def get_lhs_ip(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        nelec_ip = self._nelec_ip
        assert nelec_ip[0] + nelec_ip[1] == nelec[0] + nelec[1] - 1
        vec0  = self.vec0 if vec0 is None else vec0

        if nelec_ip[0] == nelec[0] - 1:
            des_op = fci.addons.des_a
        else:
            des_op = fci.addons.des_b

        lhs_ip = numpy.asarray([des_op(vec0, norb, nelec, p).reshape(-1) for p in orb_list])
        lhs_ip = lhs_ip.reshape(len(orb_list), -1)

        return lhs_ip

    def get_rhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        nelec_ea = self._nelec_ea
        assert nelec_ea[0] + nelec_ea[1] == nelec[0] + nelec[1] + 1
        vec0  = self.vec0 if vec0 is None else vec0

        if nelec_ea[0] == nelec[0] + 1:
            cre_op = fci.addons.cre_a
        else:
            cre_op = fci.addons.cre_b

        rhs_ea = numpy.asarray([cre_op(vec0, norb, nelec, p).reshape(-1) for p in orb_list])
        rhs_ea = rhs_ea.reshape(len(orb_list), -1)

        return rhs_ea

    def get_lhs_ea(self, vec0=None, orb_list=None, verbose=None):
        norb = self.norb
        orb_list = orb_list if orb_list is not None else range(norb)
        orb_list = numpy.asarray(orb_list)

        nelec = self._nelec
        nelec_ea = self._nelec_ea
        assert nelec_ea[0] + nelec_ea[1] == nelec[0] + nelec[1] + 1
        vec0  = self.vec0 if vec0 is None else vec0

        if nelec_ea[0] == nelec[0] + 1:
            cre_op = fci.addons.cre_a
        else:
            cre_op = fci.addons.cre_b

        lhs_ea = numpy.asarray([cre_op(vec0, norb, nelec, p).reshape(-1) for p in orb_list])
        lhs_ea = lhs_ea.reshape(len(orb_list), -1)

        return lhs_ea

    def gen_hop_ip(self, verbose=None):
        return _gen_hop_slow(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_slow(self, comp="ea", verbose=verbose)

class DirectFullConfigurationInteraction(SlowFullConfigurationInteraction):
    def gen_hop_ip(self, verbose=None):
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)

def FCIGF(hf_obj, method="slow"):
    if method.lower() == "slow":
        return SlowFullConfigurationInteraction(hf_obj)

    elif method.lower() == "direct":
        return DirectFullConfigurationInteraction(hf_obj)

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
    gfn_obj.build()

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