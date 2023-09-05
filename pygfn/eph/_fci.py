import numpy

import pyscf
from pyscf import fci, lib, ao2mo

from pyscf.fci import direct_ep
from pyscf.fci.cistring import num_strings
from pyscf.fci.cistring import gen_linkstr_index
from pyscf.fci.direct_spin1 import _unpack_nelec

fci_direct = pyscf.fci.direct_uhf

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
    # Ensure that the shape of h1e1p matches the expected dimensions
    assert h1e1p.shape == (nsite, nsite, nmode)

    # Determine the shape of the reshaped vector based on given parameters
    shape = make_shape(nsite, nelec, nmode, nph_max)
    # Reshape the vector v to the desired shape
    c = v.reshape(shape)
    # Initialize an array of zeros with the same shape as c for storing the results
    hc = numpy.zeros_like(c)

    # Calculate factors by combining phonon creation operators and h1e1p values
    factors = h1e1p[:, :, :nmode, None] * (numpy.sqrt(numpy.arange(1, nph_max + 1)))[None, None, None, :]

    # Loop over spin states (alpha and beta)
    for s, link_index in enumerate(map(lambda n: gen_linkstr_index(range(nsite), n), _unpack_nelec(nelec))):
        # Iterate over each link index
        for str0, tab in enumerate(link_index):
            # Iterate over elements in the link table
            for a, i, str1, sign in tab:
                # Loop over modes
                for alph in range(nmode):
                    # Get the relevant factors based on mode and link table indices
                    fs = factors[a, i, alph]
                    # Iterate over the factors
                    for np, f in enumerate(fs):
                        # Define a slice for initial configuration
                        s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
                        s0[2 + alph] = np
                        s0[s] = str0  # Set spin state
                        s0 = tuple(s0)

                        # Define a slice for the resultant configuration (with phonon number incremented by 1)
                        s1 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
                        s1[2 + alph] = np + 1
                        s1[s] = str1
                        s1 = tuple(s1)

                        # Update the resultant array using the calculated factors and the initial configuration
                        hc[s1] += sign * f * c[s0]

    # Return the reshaped result
    return hc.reshape(shape)


def contract_h1p(h1p, v, nsite, nelec, nmode, nph_max):
    # Phonon-phonon coupling
    cishape = make_shape(nsite, nelec, nmode, nph_max)
    c = v.reshape(cishape)
    hc = numpy.zeros_like(c)

    factors = numpy.sqrt(numpy.arange(1, nph_max + 1))
    t1 = numpy.zeros((nmode,) + cishape)

    for alph in range(nmode):
        for np, f in enumerate(factors):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = np
            s0 = (alph,) + tuple(s0)

            s1 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s1[2 + alph] = np + 1
            s1 = tuple(s1)

            t1[s0] += c[s1] * f

    t1 = lib.dot(h1p, t1.reshape(nsite, -1)).reshape(t1.shape)

    for alph in range(nmode):
        for np, f in enumerate(factors):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = np
            s0 = (alph,) + tuple(s0)

            s1 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s1[2 + alph] = np + 1
            s1 = tuple(s1)

            hc[s1] += t1[s0] * f

    return hc.reshape(cishape)


def make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max):
    shape = make_shape(nsite, nelec, nmode, nph_max)
    na, nb = shape[:2]

    hdiag_e = fci_direct.make_hdiag(h1e, eri, nsite, nelec).reshape(na, nb)

    hdiag = numpy.zeros(shape)
    for alph in range(nmode):
        for np in range(nph_max+1):
            s0 = [slice(None, None, None)] * (2 + nmode)  # +2 for electron indices
            s0[2 + alph] = np
            s0 = tuple(s0)

            hdiag[:, :, alph, np] = hdiag_e
            hdiag[s0] += np + 1

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
        assert ci0.shape == shape

    h2e = fci_direct.absorb_h1e(h1e, eri, nsite, nelec, .5)
    def hop(v):
        c = v.reshape(shape)
        hc  = contract_h1e(h1e, c, nsite, nelec, nmode, nph_max)
        hc += contract_h1e1p(h1e1p, c, nsite, nelec, nmode, nph_max)
        hc += contract_h1p(h1p, c, nsite, nelec, nmode, nph_max)
        hv = hc.reshape(-1)
        return hv

    hdiag = make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max)
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
    e, c = lib.davidson(
        hop, ci0.reshape(-1), precond,
        tol=tol, max_cycle=max_cycle, verbose=verbose,
        **kwargs
    )
    return e + h0, c


if __name__ == '__main__':
    nsite = 2
    nmode = 2
    nelec = 2
    nphonon = 1

    t = numpy.zeros((nsite, nsite))
    idx = numpy.arange(nsite - 1)
    t[idx + 1, idx] = t[idx, idx + 1] = -1

    u = 0.0
    g = 0.0

    hpp = numpy.eye(nmode) * 1.1
    idx = numpy.arange(nmode - 1)
    hpp[idx + 1, idx] = hpp[idx, idx + 1] = .1

    eri = numpy.zeros((nsite, nsite, nsite, nsite))
    eri[idx, idx, idx, idx] = u


    nelecs = [(ia, ib) for ia in range(nsite + 1) for ib in range(ia + 1)]
    for nelec in nelecs:
        c = numpy.random.random(make_shape(nsite, nelec, nmode, nphonon))
        hc1 = fci.direct_ep.contract_1e(t, c, nsite, nelec, nphonon)
        hc  = contract_h1e((t, t), c, nsite, nelec, nmode, nphonon)

        err = numpy.linalg.norm(hc1 - hc)
        assert err < 1e-10

        hc1 = fci.direct_ep.contract_2e((0.0 * eri, 0.5 * eri, 0.0 * eri), c, nsite, nelec, nphonon)
        hc2 = fci.direct_ep.contract_2e_hubbard(u, c, nsite, nelec, nphonon)
        hc  = contract_h2e((0.0 * eri, 0.5 * eri, 0.0 * eri), c, nsite, nelec, nmode, nphonon)

        err = numpy.linalg.norm(hc1 - hc)
        assert err < 1e-10

        err = numpy.linalg.norm(hc2 - hc)
        assert err < 1e-10


