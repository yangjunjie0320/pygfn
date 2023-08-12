import pyscf
from pyscf import lib
from pyscf import __config__

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
    verbose    = getattr(__config__, 'gf_verbose',  4)
    max_memory = getattr(__config__, 'gf_max_memory', 40000)  # 2 GB
    conv_tol   = getattr(__config__, 'gf_conv_tol', 1e-6)
    max_cycle  = getattr(__config__, 'gf_max_cycle', 50)
    gmres_m    = getattr(__config__, 'gf_gmres_m', 30)

    nelec = None
    norb  = None

    def __init__(self, m, tol=1e-8, verbose=0) -> None:
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def solve_gf_ip(self, omegas, ps=None, qs=None, eta=0.01, verbose=None):
        norb = self.norb
        ps, qs, np, nq = _get_pq(ps, qs, norb=norb)
        pass
    
    def solve_gf_ea(self):
        raise NotImplementedError

    def solve_gf(self, omegas, ps=None, qs=None, eta=0.01, name="ip", verbose=None):
        name = name.lower()
        assert name.lower() in ["ip", "ea"]

        norb = self.norb
        ps, qs, np, nq = _unpack_pq(ps, qs, norb=norb)

        nelec = self.nelec
        assert nelec[0] >= nelec[1]
        nelec_n = (nelec[0], nelec[1] + 1) if name == "ip" else (nelec[0] - 1, nelec[1])
        assert nelec_n[0] >= 0 and nelec_n[1] >= 0

        vec_size = 0
        amp_size = 0

        if name == "ip":
            rhs_n   = self.get_rhs_ip(qs, verbose=verbose)
            lag_n   = self.get_lag_ip(ps, verbose=verbose)
            hop_n, hdiag_n = self.gen_hop_ip(verbose=verbose)

            amp_n_size = 0
            vec_n_size = 0

            assert rhs_n.shape   == (nq, vec_n_size)
            assert lag_n.shape   == (np, vec_n_size)
            assert hdiag_n.shape == (vec_n_size,)

        elif name == "ea":
            rhs_n   = self.get_rhs_ea(qs, verbose=verbose)
            lag_n   = self.get_lag_ea(ps, verbose=verbose)
            hop_n, hdiag_n = self.gen_hop_ea(verbose=verbose)

            amp_n_size = 0
            vec_n_size = 0

            assert rhs_n.shape   == (nq, vec_n_size)
            assert lag_n.shape   == (np, vec_n_size)
            assert hdiag_n.shape == (vec_n_size,)

        else:
            raise ValueError

        def gen_gf(omega):
            omega_e0_eta = omega + 1j * eta
            hdiag_omega  = hdiag_ + omega_e0_eta

            def h_omega(x):
                return hop.dot(x) - omega_e0_eta * x

    def get_rhs_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_rhs_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lag_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lag_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def gen_hop_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def gen_hop_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def kernel(self):
        raise NotImplementedError

GF = GreensFunctionMixin