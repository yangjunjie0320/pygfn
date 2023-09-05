import pygfn
from pygfn.eph.fci import FullConfigurationInteractionDirectSpin1

def EPH(gf_obj, h1p=None, h1e1p=None):
    assert isinstance(gf_obj, pygfn.fci.GreensFunctionMixin)
    assert not isinstance(gf_obj, pygfn.eph.fci.WithPhonon)

    if isinstance(gf_obj, pygfn.fci.FullConfigurationInteractionDirectSpin1):
        return FullConfigurationInteractionDirectSpin1(gf_obj._base, h1p=h1p, h1e1p=h1e1p)

    raise NotImplementedError