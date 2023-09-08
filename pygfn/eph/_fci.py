import numpy

import pyscf
from pyscf import fci, lib, ao2mo

from pyscf.fci import direct_ep
from pyscf.fci.cistring import num_strings
from pyscf.fci.cistring import gen_linkstr_index
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.direct_ep import slices_for, slices_for_cre, slices_for_des

# Electron-phonon FCI ground state solver
# Note: this implementation is an improved version of pyscf.fci.direct_ep

def make_shape(nsite, nelec, nmode, nph_max):
    """
    Construct the shape tuple for the array that represents the quantum states.

    Parameters:
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.

    Returns:
    - tuple: The shape of the array representing quantum states.
    """
    # Split the number of electrons into alpha and beta electrons
    nelec_alph, nelec_beta = _unpack_nelec(nelec)
    # Determine the number of ways to place alpha electrons among available sites
    na = num_strings(nsite, nelec_alph)
    # Determine the number of ways to place beta electrons among available sites
    nb = num_strings(nsite, nelec_beta)
    # Return the array shape
    return (na, nb) + (nph_max + 1,) * nmode


def contract_h1e(h1e, v, nsite, nelec, nmode, nph_max, fci_obj=None):
    """
    Perform the contraction of a one-electron operator with FCI vector.

    Parameters:
    - h1e: One-electron Hamiltonian.
    - v: FCI vector.
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.
    - fci_obj (optional): The FCI object used for contraction. Uses pyscf.fci.direct_spin1 by default.

    Returns:
    - numpy.ndarray: Result of the contraction.
    """
    # Use default FCI object if none is provided
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    # Ensure h1e is a numpy array
    h1e = numpy.asarray(h1e)
    # Calculate the shape for the wave function array
    shape = make_shape(nsite, nelec, nmode, nph_max)
    # Reshape the input array according to the calculated shape
    c = v.reshape(shape)
    na, nb = shape[:2]
    c = c.reshape(na * nb, -1)
    np = c.shape[1]

    gen_hc = lambda i: fci_obj.contract_1e(h1e, c[:, i], nsite, nelec).reshape(na, nb)
    hc = [gen_hc(i) for i in range(np)]
    hc = numpy.asarray(hc).transpose(1, 2, 0).reshape(shape)

    return hc.reshape(shape)


def contract_h2e(h2e, v, nsite, nelec, nmode, nph_max, fci_obj=None):
    """
    Perform the contraction of a two-electron operator with a wave function array.

    Parameters:
    - h2e: Two-electron Hamiltonian. Note: this is not ERI.
    - v: FCI vector.
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.
    - fci_obj (optional): The FCI object used for contraction. Uses pyscf.fci.direct_spin1 by default.

    Returns:
    - numpy.ndarray: Result of the contraction.
    """
    # Use default FCI object if none is provided
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    # Ensure h2e is a numpy array
    h2e = numpy.asarray(h2e)
    # Calculate the shape for the wave function array
    shape = make_shape(nsite, nelec, nmode, nph_max)
    # Reshape the input array according to the calculated shape
    c = v.reshape(shape)
    na, nb = shape[:2]
    c = c.reshape(na * nb, -1)
    np = c.shape[1]

    gen_hc = lambda i: fci_obj.contract_2e(h2e, c[:, i], nsite, nelec).reshape(na, nb)
    hc = [gen_hc(i) for i in range(np)]
    hc = numpy.asarray(hc).transpose(1, 2, 0).reshape(shape)

    return hc.reshape(shape)


def contract_h1e1p(h1e1p, v, nsite, nelec, nmode, nph_max, fci_obj=None):
    """
    Perform the contraction of a electron-phonon Hamiltonian with FCI vector.

    Parameters:
    - h1e1p: Electron-phonon Hamiltonian.
    - v: FCI vector.
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.
    - fci_obj (optional): The FCI object used for contraction. Uses pyscf.fci.direct_spin1 by default.

    Returns:
    - numpy.ndarray: Result of the contraction.
    """
    # Use default FCI object if none is provided
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    # Ensure h1e1p is a numpy array
    h1e1p = numpy.asarray(h1e1p)
    # Calculate the shape for the wave function array
    shape = make_shape(nsite, nelec, nmode, nph_max)
    # Reshape the input array according to the calculated shape
    c = v.reshape(shape)

    # Initialize the result with zeros having the same shape as input array
    hc = numpy.zeros_like(c)
    na, nb = shape[:2]

    # Further reshape the array for contraction
    c = c.reshape(na * nb, -1)
    np = c.shape[1]

    # Factors for phonon contributions
    factors = numpy.sqrt(numpy.arange(1, nph_max + 1))

    # Loop through the phonon modes
    for alph in range(nmode):
        gen_hc_e = lambda i: fci_obj.contract_1e(h1e1p[..., alph], c[:, i], nsite, nelec).reshape(na, nb)
        hc_e = [gen_hc_e(i) for i in range(np)]
        hc_e = numpy.asarray(hc_e).transpose(1, 2, 0)
        hc_e = hc_e.reshape(shape)

        # Apply phonon factors to the result
        for nph, f in enumerate(factors):
            s0 = slices_for(alph, nmode, nph)
            s1 = slices_for_cre(alph, nmode, nph)

            hc[s1] += hc_e[s0] * f
            hc[s0] += hc_e[s1] * f

    return hc.reshape(shape)


def contract_h1p(h1p, v, nsite, nelec, nmode, nph_max, fci_obj=None):
    """
    Perform the contraction of a one-phonon operator with a wave function array.

    Parameters:
    - h1p: One-phonon operator.
    - v: Wave function array.
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.
    - fci_obj (optional): The FCI object. It's mentioned in the parameter list but not used in this function.
      It might be kept for consistency with other functions or future implementations.

    Returns:
    - numpy.ndarray: Result of the contraction.
    """
    # Phonon-phonon coupling
    # Determine the shape of the CI array
    shape = make_shape(nsite, nelec, nmode, nph_max)
    # Reshape the input array according to the determined shape
    c = v.reshape(shape)
    # Initialize the result with zeros having the same shape as input array
    hc = numpy.zeros_like(c)

    # Factors for phonon contributions
    factors = numpy.sqrt(numpy.arange(1, nph_max + 1))
    t1 = numpy.zeros((nmode,) + shape)

    # Calculate t1 array using the input wave function array
    for alph in range(nmode):
        for nph, f in enumerate(factors):
            s0 = slices_for(alph, nmode, nph)
            s1 = slices_for_cre(alph, nmode, nph)
            t1[(alph,) + s0] += c[s1] * f

    # Dot product of h1p with reshaped t1 to update the t1 array
    t1 = lib.dot(h1p, t1.reshape(nmode, -1)).reshape(t1.shape)

    # Use the updated t1 array to calculate the result hc array
    for alph in range(nmode):
        for nph, f in enumerate(factors):
            s0 = slices_for(alph, nmode, nph)
            s1 = slices_for_cre(alph, nmode, nph)
            hc[s1] += t1[(alph,) + s0] * f

    return hc.reshape(shape)


def make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max, fci_obj=None):
    """
    Compute the diagonal of the Hamiltonian for electron and phonon systems.
    Note: the implementation in pyscf.fci.direct_ep is not correct.

    Parameters:
    - h1e: One-electron integral array.
    - eri: Two-electron integral array.
    - h1e1p: One-electron one-phonon operator.
    - h1p: One-phonon operator.
    - nsite: Number of molecular orbitals (or sites).
    - nelec: Tuple of the number of alpha and beta electrons.
    - nmode: Number of phonon modes.
    - nph_max: Maximum number of phonons.
    - fci_obj (optional): The FCI object used for contraction. Uses pyscf.fci.direct_spin1 by default.

    Returns:
    - numpy.ndarray: Diagonal of the Hamiltonian.
    """
    # Use default FCI object if none is provided
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    # Determine the shape of the CI array
    shape = make_shape(nsite, nelec, nmode, nph_max)
    na, nb = shape[:2]

    # Calculate the electron-part of the Hamiltonian diagonal
    hdiag_e = fci_obj.make_hdiag(h1e, eri, nsite, nelec).reshape(na, nb)

    # Expand the electron-only diagonal to the full phonon space
    hdiag = numpy.asarray([hdiag_e] * (nph_max + 1) ** nmode)
    hdiag = hdiag.transpose(1, 2, 0).reshape(shape)

    # Add phonon contributions to the Hamiltonian diagonal
    for alph in range(nmode):
        for nph in range(nph_max + 1):
            s0 = slices_for(alph, nmode, nph)
            hdiag[s0] += nph * h1p[alph, alph]

    # Flatten and return the diagonal
    return hdiag.ravel()

def gen_hop(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max, fci_obj=None):
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    shape = make_shape(nsite, nelec, nmode, nph_max)

    if eri is not None:
        h2e = fci_obj.absorb_h1e(h1e, eri, nsite, nelec, .5)
        def hop(v):
            c = v.reshape(shape)
            hc = contract_h2e(h2e, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            hc += contract_h1e1p(h1e1p, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            hc += contract_h1p(h1p, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            return hc.ravel()

    else:
        def hop(v):
            c = v.reshape(shape)
            hc = contract_h1e(h1e, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            hc += contract_h1e1p(h1e1p, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            hc += contract_h1p(h1p, c, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)
            return hc.ravel()

    return hop

def kernel(h1e, eri, h1e1p, h1p, nsite, nmode, nelec, nph_max,
           tol=1e-9, max_cycle=100, verbose=0, ci0=None, h0=0.0,
           noise=1e-6, fci_obj=None, **kwargs):
    """
    Compute the ground state energy and wavefunction of the electron-phonon system using the Davidson algorithm.

    Parameters:
    - h1e, eri, h1e1p, h1p: Electronic and phononic integrals.
    - nsite: Number of molecular orbitals (or sites).
    - nmode: Number of phonon modes.
    - nelec: Tuple of the number of alpha and beta electrons.
    - nph_max: Maximum number of phonons.
    - tol: Convergence tolerance.
    - max_cycle: Maximum number of Davidson iterations.
    - verbose: Verbosity level.
    - ci0: Initial guess for the CI vector.
    - h0: Constant energy shift.
    - noise: Magnitude of random noise added for initial guess.
    - fci_obj: FCI object used for contractions. Uses pyscf.fci.direct_spin1 by default.
    - **kwargs: Additional arguments for the Davidson solver.

    Returns:
    - e: Ground state energy.
    - c: Ground state wavefunction (CI coefficients).
    """
    # Initialize FCI object if none provided
    if fci_obj is None:
        fci_obj = pyscf.fci.direct_spin1

    # Determine the shape of the CI array
    shape = make_shape(nsite, nelec, nmode, nph_max)

    # Construct initial guess for CI coefficients
    if ci0 is None:
        ci0 = numpy.zeros(shape)
        ci0.__setitem__((0, 0) + (0,) * nmode, 1.0)

        # Add noise to the initial guess if specified
        if noise is not None:
            ci0[0, :] += numpy.random.random(ci0[0, :].shape) * noise
            ci0[:, 0] += numpy.random.random(ci0[:, 0].shape) * noise
    else:
        ci0 = ci0.reshape(shape)

    # Convert input arrays to numpy arrays
    h1e = numpy.asarray(h1e)
    eri = numpy.asarray(eri)
    h1e1p = numpy.asarray(h1e1p)
    h1p = numpy.asarray(h1p)

    # Compute the diagonal of the Hamiltonian
    hdiag = make_hdiag(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max, fci_obj=fci_obj)

    # Preconditioner for Davidson
    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

    # Use Davidson algorithm to find the lowest eigenvalue and eigenvector
    e, c = lib.davidson(
        gen_hop(h1e, eri, h1e1p, h1p, nsite, nelec, nmode, nph_max, fci_obj=fci_obj),
        ci0.reshape(-1), precond, tol=tol, max_cycle=max_cycle, verbose=verbose, **kwargs
    )

    return e + h0, c

if __name__ == '__main__':
    nsite = 2
    nmode = 2
    nph_max = 4
    nroots = 5

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

    nelecs = [(ia, ib) for ia in range(nsite + 1) for ib in range(ia + 1)]

    for nelec in nelecs:
        shape = make_shape(nsite, nelec, nmode, nph_max)
        size  = numpy.prod(shape)

        ene_0, c_0 = fci.direct_ep.kernel(h1e, u, g, h1p, nsite, nelec, nph_max,
                                          tol=1e-10, max_cycle=1000, verbose=0, nroots=nroots)
        ene_1, c_1 = kernel(h1e, eri, h1e1p, h1p, nmode, nsite, nelec, nph_max=nph_max, nroots=nroots)

        err = numpy.linalg.norm(ene_1[0] - ene_0[0]) / ene_1.size
        assert err < 1e-8, "error in energy: %6.4e" % err

        # Note: the implementation in pyscf.fci.direct_ep is not correct.
        # Directly comparing the hdiag with the one from pyscf.fci.direct_ep will fail.
        hdiag_0 = []
        hop = gen_hop(h1e, eri, h1e1p, 0.0 * h1p, nsite, nelec, nmode, nph_max)
        for i in range(size):
            v = numpy.zeros(size)
            v[i] = 1.0
            hdiag_0.append(hop(v)[i])
        hdiag_0 = numpy.asarray(hdiag_0)

        hdiag_1 = make_hdiag(h1e, eri, h1e1p, 0.0 * h1p, nsite, nelec, nmode, nph_max)

        err = numpy.linalg.norm(hdiag_1 - hdiag_0) / hdiag_0.size
        assert err < 1e-8, "error in hdiag: %6.4e" % err

        print("Passed: ", nelec)