import numpy

import pyscf
from pyscf import gto, fci
from pyscf.lo.orth import orth_ao
from pygfn import eph

m = pyscf.gto.Mole()
m.verbose = 0
m.atom = 'H 0 0 0; Li 0 0 5.0'
m.unit = 'B'
m.basis = '631g'
m.build()

nelec = m.nelec
coeff_lo = orth_ao(m, 'meta_lowdin')
nao, norb = coeff_lo.shape

d_ao = m.intor('int1e_r', comp=3).reshape(3, nao, nao)
d_lo = numpy.einsum('xmn,mp,nq->xpq', d_ao, coeff_lo, coeff_lo)

fci_obj = fci.FCI(m, mo=coeff_lo, singlet=False)
fci_obj.nroots = 10
fci_obj.max_cycle = 1000
fci_obj.conv_tol = 1e-10
fci_obj.verbose = 0
e0, c0 = fci_obj.kernel()
for istate in range(10):
    print("e0[%d] = %12.8f" % (istate, e0[istate]))
    print("s2 = %12.8f" % fci.spin_square(c0[istate], norb, nelec)[1])

nmode = 1
nph_max = 3
state_1 = 0
state_2 = 2

omega = e0[state_2] - e0[state_1]
h1p = numpy.zeros((nmode, nmode))
h1p[0, 0] = omega

c0_1 = c0[state_1]
c0_2 = c0[state_2]
tdm_lo = fci_obj.trans_rdm1(c0_1, c0_2, norb, nelec)
dd = numpy.ones(3) # numpy.einsum('pq,xpq->x', tdm_lo, d_lo)

for alph in [0.005]:
    vv = dd * alph / numpy.linalg.norm(dd)
    vv = vv.reshape(nmode, 3)
    h1e1p_ao = numpy.einsum('Ix,xpq->pqI', vv, d_ao)

    eph_obj = eph.FCI(m, mo=coeff_lo, h1p=h1p, h1e1p=h1e1p_ao, nph_max=3, singlet=False)
    eph_obj.max_cycle = 1000
    eph_obj.conv_tol = 1e-10
    e1, c1 = eph_obj.kernel(nroots=10)

    for istate in range(10):
        print("e1[%d] = %12.8f" % (istate, e1[istate]))
