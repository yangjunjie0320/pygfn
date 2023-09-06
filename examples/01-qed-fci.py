import numpy

import pyscf
from pyscf import gto, fci
from pyscf.lo.orth import orth_ao
from pygfn import eph

m = pyscf.gto.Mole()
m.verbose = 0
m.atom = 'H 0 0 0; Li 0 0 1.1'
m.basis = '631g'
m.build()

nelec = m.nelec
coeff_lo = orth_ao(m, 'meta_lowdin')
nao, norb = coeff_lo.shape

d_ao = m.intor('int1e_r', comp=3).reshape(3, nao, nao)
d_lo = numpy.einsum('xmn,mp,nq->xpq', d_ao, coeff_lo, coeff_lo)

fci_obj = fci.FCI(m, mo=coeff_lo)
fci_obj.nroots = 20
fci_obj.max_cycle = 1000
fci_obj.conv_tol = 1e-10
e0, c0 = fci_obj.kernel()

nmode = 1
state_1 = 0
state_2 = 2

omega = e0[state_2] - e0[state_1]
h1p = numpy.zeros((nmode, nmode))
h1p[0, 0] = omega

c0_1 = c0[state_1]
c0_2 = c0[state_2]
tdm_lo = fci_obj.trans_rdm1(c0_1, c0_2, norb, nelec)
dd = numpy.einsum('pq,xpq->x', tdm_lo, d_lo)

from pyscf.fci.direct_ep import des_phonon, cre_phonon
for alph in [1e-4, 2e-4, 4e-4, 8e-4, 16e-4]:
    vv = dd / numpy.linalg.norm(dd) * alph
    vv = vv.reshape(nmode, 3)
    h1e1p = numpy.einsum('Ix,xpq->pqI', vv, d_lo)

    eph_obj = eph.FCI(m, mo=coeff_lo, h1p=h1p, h1e1p=h1e1p)
    eph_obj.nroots = 20
    eph_obj.max_cycle = 1000
    eph_obj.conv_tol = 1e-10
    e1, c1 = eph_obj.kernel()

    for c in c1:
        cc = des_phonon(c, 1, nelec, nphonon, j)
        cc = cre_phonon(cc, 1, nelec, nphonon, i)
                dm1a[i, j] = numpy.dot(c.ravel(), c1.ravel())
        print('check phonon DM', numpy.allclose(dm1, dm1a))
