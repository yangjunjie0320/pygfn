import numpy

import pyscf
from pyscf import fci, lib, ao2mo

from pyscf.fci import direct_ep
from pyscf.fci.cistring import num_strings
from pyscf.fci.cistring import gen_linkstr_index
from pyscf.fci.direct_spin1 import _unpack_nelec
fci_direct = pyscf.fci.direct_spin1

import pygfn

def make_shape(nsite, nelec, nmode, nph_max):
    nelec_alph, nelec_beta = _unpack_nelec(nelec)
    na = num_strings(nsite, nelec_alph)
    nb = num_strings(nsite, nelec_beta)
    return (na, nb) + (nph_max + 1,) * nmode

def contract_h1e(h1e, v, nsite, nelec, nmode, nph_max):
    shape = make_shape(nsite, nelec, nmode, nph_max)
    c = v.reshape(shape)
    na, nb = shape[:2]
    c = c.reshape(na * nb, -1)
    np = c.shape[1]

    gen_hc = lambda i: fci_direct.contract_1e(h1e, c[:, i], nsite, nelec).reshape(na, nb)
    hc = [gen_hc(i) for i in range(np)]
    hc = numpy.asarray(hc).transpose(1, 2, 0)

    return hc.reshape(shape)

def contract_h2e(h2e, v, nsite, nelec, nmode, nph_max):
    shape = make_shape(nsite, nelec, nmode, nph_max)
    c = v.reshape(shape)
    na, nb = shape[:2]
    c = c.reshape(na * nb, -1)
    np = c.shape[1]

    gen_hc = lambda i: fci_direct.contract_2e(h2e, c[:, i], nsite, nelec).reshape(na, nb)
    hc = [gen_hc(i) for i in range(np)]
    hc = numpy.asarray(hc).transpose(1, 2, 0)

    return hc.reshape(shape)

def contract_h1e1p(h1e1p, v, nsite, nelec, nmode, nph_max):
    h1e1p = numpy.asarray(h1e1p)
    shape = make_shape(nsite, nelec, nmode, nph_max)
    c = v.reshape(shape)
    na, nb = shape[:2]
    hc = numpy.zeros_like(c)

    # c = c.reshape(na * nb, -1)
    # np = c.shape[1]

    from pyscf.fci import cistring
    link_index_alph = cistring.gen_linkstr_index(range(nsite), nelec[0])
    link_index_beta = cistring.gen_linkstr_index(range(nsite), nelec[1])
    factors = numpy.sqrt(numpy.arange(1, nph_max + 1))

    for alph in range(nmode):
        # gen_hc_e = lambda i: fci_direct.contract_1e(h1e1p[..., alph], c[:, i], nsite, nelec).reshape(na, nb)
        # hc_e = [gen_hc_e(i) for i in range(np)]
        # hc_e = numpy.asarray(hc_e).transpose(1, 2, 0)
        # hc_e = hc_e.reshape(shape)
        hc_e = numpy.zeros(shape)

        for str0, tab in enumerate(link_index_alph):
            for a, j, str1, sign in tab:
                hc_e[str1] += sign * c[str0] * h1e1p[0, a, j, alph]

        for str0, tab in enumerate(link_index_beta):
            for a, j, str1, sign in tab:
                hc_e[:, str1] += sign * c[:, str0] * h1e1p[0, a, j, alph]

        # hc_e -= float(nelec[0] + nelec[1]) / nsite
        from pyscf.fci.direct_ep import slices_for, slices_for_cre, slices_for_des
        for nph, f in enumerate(factors):
            s0 = slices_for(alph, nmode, nph)
            s1 = slices_for_cre(alph, nmode, nph)

            hc[s1] += hc_e[s0] * f
            hc[s0] += hc_e[s1] * f

    return hc.reshape(shape)

def contract_h1p(h1p, v, nsite, nelec, nmode, nph_max):
    # Phonon-phonon coupling
    cishape = make_shape(nsite, nelec, nmode, nph_max)
    c = v.reshape(cishape)
    hc = numpy.zeros_like(c)

    factors = numpy.sqrt(numpy.arange(1, nph_max + 1))
    t1 = numpy.zeros((nmode,) + cishape)

    for alph in range(nmode):
        for nph, f in enumerate(factors):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = nph
            s0 = (alph,) + tuple(s0)

            s1 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s1[2 + alph] = nph + 1
            s1 = tuple(s1)

            t1[s0] += c[s1] * f

    t1 = lib.dot(h1p, t1.reshape(nsite, -1)).reshape(t1.shape)

    for alph in range(nmode):
        for nph, f in enumerate(factors):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = nph
            s0 = (alph,) + tuple(s0)

            s1 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s1[2 + alph] = nph + 1
            s1 = tuple(s1)

            hc[s1] += t1[s0] * f

    return hc.reshape(cishape)


def make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max):
    shape = make_shape(nsite, nelec, nmode, nph_max)
    na, nb = shape[:2]

    hdiag_e = fci_direct.make_hdiag(h1e, eri, nsite, nelec).reshape(na, nb)
    hdiag = numpy.hstack([hdiag_e] * (nph_max + 1) ** nmode).reshape(shape)

    for alph in range(nmode):
        for nph in range(nph_max+1):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = nph
            s0 = tuple(s0)

            hdiag[s0] += nph + 1

    return hdiag.ravel()


def kernel(h1e, eri, h1e1p, h1p, nsite, nmode, nelec, nph_max,
           tol=1e-9, max_cycle=100, verbose=0, ci0=None, h0=0.0,
           noise=1e-6, **kwargs):
    shape = make_shape(nsite, nelec, nmode, nph_max)

    if ci0 is None:
        # Add noise for initial guess, remove it if problematic
        ci0 = numpy.zeros(shape)
        ci0.__setitem__((0, 0) + (0,) * nsite, 1.0)

        if noise is not None:
            ci0[0, :] += numpy.random.random(ci0[0, :].shape) * noise
            ci0[:, 0] += numpy.random.random(ci0[:, 0].shape) * noise

    else:
        ci0 = ci0.reshape(shape)

    # h2e = fci_direct.absorb_h1e(h1e, eri, nsite, nelec, .5)
    def hop(v):
        c = v.reshape(shape)
        hc  = contract_h1e(h1e, c, nsite, nelec, nmode, nph_max)
        hc += contract_h1e1p(h1e1p, c, nsite, nelec, nmode, nph_max)
        hc += contract_h2e(eri, c, nsite, nelec, nmode, nph_max)
        hc += contract_h1p(h1p, c, nsite, nelec, nmode, nph_max)
        hv  = hc.reshape(-1)
        return hv

    hdiag = make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max)
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    e, c = lib.davidson(
        hop, ci0.reshape(-1), precond,
        tol=tol, max_cycle=max_cycle,
        verbose=verbose,
        **kwargs
    )
    return e + h0, c

if __name__ == '__main__':
    nsite = 4
    nmode = 4
    nph_max = 2

    u = 1.5
    g = 0.5

    h1e = numpy.zeros((nsite, nsite))
    idx_site = numpy.arange(nsite - 1)
    h1e[idx_site + 1, idx_site] = h1e[idx_site, idx_site + 1] = -1.0

    idx_site = numpy.arange(nsite)
    idx_mode = numpy.arange(nmode)
    eri = numpy.zeros((nsite, nsite, nsite, nsite))
    eri[idx_site, idx_site, idx_site, idx_site] = u

    h1e1p = numpy.zeros((nsite, nsite, nmode))
    h1e1p[idx_site, idx_site, idx_mode] = g

    idx_mode = numpy.arange(nmode - 1)
    h1p = numpy.eye(nmode) * 1.1
    h1p[idx_mode + 1, idx_mode] = h1p[idx_mode, idx_mode + 1] = 0.1

    fci_direct = pyscf.fci.direct_uhf

    nelecs = [(ia, ib) for ia in range(nsite + 1) for ib in range(ia + 1)]
    for nelec in nelecs:
        print("\n nelec = ", (nelec))
        c = numpy.random.random(make_shape(nsite, nelec, nmode, nph_max))
        hc1 = fci.direct_ep.contract_1e(h1e, c, nsite, nelec, nph_max)
        hc  = contract_h1e((h1e, h1e), c, nsite, nelec, nmode, nph_max)
        print("err = %6.4e" % numpy.linalg.norm(hc1 - hc))
        err = numpy.linalg.norm(hc1 - hc)
        assert err < 1e-10

        hc1 = fci.direct_ep.contract_2e((0.0 * eri, 0.5 * eri, 0.0 * eri), c, nsite, nelec, nph_max)
        hc2 = fci.direct_ep.contract_2e_hubbard(u, c, nsite, nelec, nph_max)
        hc  = contract_h2e((0.0 * eri, 0.5 * eri, 0.0 * eri), c, nsite, nelec, nmode, nph_max)
        print("err = %6.4e" % numpy.linalg.norm(hc1 - hc))
        err = numpy.linalg.norm(hc1 - hc)
        assert err < 1e-10

        hc1 = fci.direct_ep.contract_ep(g, c, nsite, nelec, nph_max)
        hc  = contract_h1e1p((h1e1p, h1e1p), c, nsite, nelec, nmode, nph_max)
        err = numpy.linalg.norm(hc1 - hc)
        print("err = %6.4e" % err)
        assert err < 1e-10

        hc1 = fci.direct_ep.contract_pp(h1p, c, nsite, nelec, nph_max)
        hc  = contract_h1p(h1p, c, nsite, nelec, nmode, nph_max)
        err = numpy.linalg.norm(hc1 - hc)
        print("err = %6.4e" % err)
        assert err < 1e-10

        ene1, c1 = fci.direct_ep.kernel(h1e, u, g, h1p, nsite, nelec, nph_max, tol=1e-10, max_cycle=1000, verbose=0)
        ene, c = kernel((h1e, h1e), (0.0 * eri, 0.5 * eri, 0.0 * eri), (h1e1p, h1e1p), h1p, nsite, nmode, nelec, nph_max,
                         tol=1e-10, max_cycle=1000, verbose=0, ci0=c1)

        print(ene, ene1)
        err = numpy.linalg.norm(ene - ene1)
        print("err = %6.4e" % err)
        assert err < 1e-10

        # hdiag  = make_hdiag((h1e, h1e), (0.0 * eri, 0.5 * eri, 0.0 * eri), h1e1p, h1p, nsite, nelec, nmode, nph_max)
        # hdiag1 = fci.direct_ep.make_hdiag(h1e, u, g, h1p, nsite, nelec, nph_max)
        # err = numpy.linalg.norm(hdiag - hdiag1)
        # print("err = %6.4e" % err)
        # assert err < 1e-10



