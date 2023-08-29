import sys, os, numpy
from scipy.sparse.linalg import LinearOperator, gcrotmk

from pyscf.lib import logger

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

def gmres(h, b, x0=None, d=None, tol=1e-5, max_cycle=100, m=30,
          verbose=logger.NOTE):
    """Solve a matrix equation using flexible GCROT(m,k) algorithm."""

    log = logger.new_logger(None, verbose)

    assert b is not None
    if b.ndim == 1:
        b = b.reshape(1, -1)
    nb, n = b.shape
    nnb = nb * n

    if callable(h):
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        hop = LinearOperator((nnb, nnb), matvec=matvec)
    else:
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h.dot(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        assert h.shape == (n, n)
        hop = LinearOperator((nnb, nnb), matvec=matvec)

    mop = None
    if d is not None:
        assert d.shape == (n,)
        d = d.reshape(-1)

        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([x / d for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        mop = LinearOperator((nnb, nnb), matvec=matvec)

    num_iter = 0

    def callback(rk):
        nonlocal num_iter
        num_iter += 1
        log.debug(f"GMRES: iter = {num_iter:4d}, residual = {numpy.linalg.norm(rk) / nb:6.4e}")

    log.debug(f"\nGMRES Start")
    log.debug(f"GMRES: nb  = {nb:4d}, n = {n:4d},  m = {m:4d}")
    log.debug(f"GMRES: tol = {tol:4.2e}, max_cycle = {max_cycle:4d}")

    if x0 is not None:
        x0 = x0.reshape(-1)

    xs, info = gcrotmk(
        hop, b.reshape(-1), x0=x0, M=mop,
        maxiter=max_cycle, callback=callback, m=m,
        tol=tol / nb, atol=tol / nb
    )

    if info > 0:
        raise ValueError(f"Convergence to tolerance not achieved in {info} iterations")

    if nb == 1:
        xs = xs.reshape(n, )
    else:
        xs = xs.reshape(nb, n)

    return xs

if __name__ == "__main__":
    tol = 1e-10

    aa  = 100 * numpy.diag(numpy.random.rand(10))
    aa += numpy.random.rand(10, 10)
    aa += aa.T
    dd = numpy.diag(aa)

    bb = numpy.random.rand(10)
    xx = gmres(aa, bb, tol=1e-10, verbose=5)
    assert numpy.linalg.norm(aa.dot(xx) - bb) < tol

    bb = numpy.random.rand(5, 10)
    xx = gmres(aa, bb, d=dd, tol=1e-10, verbose=5)
    assert numpy.linalg.norm(numpy.einsum("ij,kj->ki", aa, xx) - bb) < tol

    bb = numpy.random.rand(1, 10)
    xx = gmres(lambda x: aa.dot(x), bb, tol=1e-10, verbose=5)
    assert numpy.linalg.norm(numpy.einsum("ij,kj->ki", aa, xx) - bb) < tol
