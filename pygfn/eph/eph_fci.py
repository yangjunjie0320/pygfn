import pygfn
from pyscf import lib

def

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
        pygfn.fci.FullConfigurationInteractionDirectSpin1.__init__(self, hf_obj)
        WithPhonon.__init__(self, h1p=h1p, h1e1p=h1e1p)

        nsite = self.nsite
        nmode = self.nmode

        assert self.h1p.shape == (nmode, nmode)
        assert self.h1e1p.shape == (nmode, nsite, nsite)

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

        rhs_ea = numpy.asarray([fci.addons.cre_b(vec0, norb, nelec0, p).reshape(-1) for p in orb_list])
        rhs_ea = rhs_ea.reshape(len(orb_list), -1)

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
        return _gen_hop_direct(self, comp="ip", verbose=verbose)

    def gen_hop_ea(self, verbose=None):
        return _gen_hop_direct(self, comp="ea", verbose=verbose)


