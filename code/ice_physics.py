# flake8: noqa ignore=F405
from fenics import *
import numpy as np


def grad2D(u):
    return as_matrix(((u[0].dx(0), u[0].dx(1), 0), (u[1].dx(0), u[1].dx(1), 0), (u[2].dx(0), u[2].dx(1), 0),))


def div2D(u):
    return u[0].dx(0) + u[1].dx(1)


def strainrate2D(u):
    return sym(grad2D(u))


def strainrate(u):
    return sym(grad(u))


def tau(eps, A, n):
    I2 = np.power(inner(eps, eps), (1.0 - n) / (2.0 * n))
    nu = A ** (-1.0 / n) * I2
    return nu * eps


### Orthotropic flow law (Rathmann and Lilien, 2022)
def tau_orthotropic(eps_3D, A, n, Exx, Eyy, Ezz, Exy, Exz, Eyz):

    # eps_3D:   strain-rate tensor in 3D
    # A:        Rate factor
    # n:        flow exponent
    # Eij:      directional enhancement factors

    # Eij = 1 => **Isotropy** (Glen's law)
    # Eij = 1 except Exx ~= 1e-2               => ice stream hard for extension (if x is along the ice stream)
    # Eij = 1 except Exy ~= 10 in shear margin => ice-stream shear margins are soft for x--y (horizontal plane) shear as if fabric was a strong single max. pointing into the ice stream

    # Assumes fabric eigen frame is the cartesian frame, mi = ei
    #   => Eij are the enhancement factors w.r.t. cartesian (model) coordinate system, (e1,e3,e3) = (x,y,z)
    m1 = Constant((1, 0, 0))
    m2 = Constant((0, 1, 0))
    m3 = Constant((0, 0, 1))

    # Tensors
    Id = Identity(3)
    P11 = (Id - 3 * outer(m1, m1)) / 2
    P22 = (Id - 3 * outer(m2, m2)) / 2
    P33 = (Id - 3 * outer(m3, m3)) / 2
    M23 = sym(outer(m2, m3))
    M31 = sym(outer(m3, m1))
    M12 = sym(outer(m1, m2))

    # Invariants
    J1 = -3 / 2 * inner(eps_3D, outer(m1, m1))
    J2 = -3 / 2 * inner(eps_3D, outer(m2, m2))
    J3 = -3 / 2 * inner(eps_3D, outer(m3, m3))
    J4 = inner(eps_3D, M23)
    J5 = inner(eps_3D, M31)
    J6 = inner(eps_3D, M12)

    # Weights
    nn = 2 / (n + 1)
    gamma = 2 * ((E22 * E33) ** nn + (E33 * E11) ** nn + (E11 * E22) ** nn) - E11 ** (2 * nn) - E22 ** (2 * nn) - E33 ** (2 * nn)
    lam1 = 4 / 3 * (E22 ** nn + E33 ** nn - E11 ** nn)
    lam2 = 4 / 3 * (E33 ** nn + E11 ** nn - E22 ** nn)
    lam3 = 4 / 3 * (E11 ** nn + E22 ** nn - E33 ** nn)
    lam4 = 2 * E23 ** nn
    lam5 = 2 * E31 ** nn
    lam6 = 2 * E12 ** nn

    # Construct flow law
    tensorialpart = 1 / gamma * (lam1 * J1 * P11 + lam2 * J2 * P22 + lam3 * J3 * P33) + 4 * (
        J4 * M23 / lam4 + J5 * M31 / lam5 + J6 * M12 / lam6
    )  # tensorial part

    viscosity = (
        1
        / 2
        * A ** (-1 / n)
        * (1 / gamma * (lam1 * J1 ** 2 + lam2 * J2 ** 2 + lam3 * J3 ** 2) + 4 * (J4 ** 2 / lam4 + J5 ** 2 / lam5 + J6 ** 2 / lam6))
        ** ((1 - n) / (2 * n))
    )

    tau_3D = 2 * viscosity * tensorialpart

    return tau_3D
