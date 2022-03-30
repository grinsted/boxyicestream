# flake8: noqa ignore=F405
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

import settings
import solution_io


def my_grad(u):
    return as_matrix(((u[0].dx(0), u[0].dx(1), 0), (u[1].dx(0), u[1].dx(1), 0), (u[2].dx(0), u[2].dx(1), 0),))


def my_div(u):
    return u[0].dx(0) + u[1].dx(1)


def strainrate(u):
    return sym(my_grad(u))


def I2(eps, p):
    return np.power(inner(eps, eps), p)


def nu(eps, A, n):
    return A ** (-1.0 / n) * I2(eps, (1.0 - n) / (2.0 * n))


def tau(u, A, n):
    eps = strainrate(u)
    return 2 * nu(eps, A, u) * eps


def a(n, A, u, v, p, q):
    eps = strainrate(u)
    return (nu(eps, A, n) * inner(sym(my_grad(v)), eps) - my_div(v) * p + q * my_div(u)) * dx


def cross_section(experiment=settings.control_experiment, **kwargs):

    experiment.update(kwargs)

    domain_w = experiment["domain_size"][1]
    domain_h = experiment["domain_size"][2]
    resolution_w = experiment["resolution"][1]
    resolution_h = experiment["resolution"][2]
    icestream_width = experiment["icestream_width"]
    shearmargin_enhancement = experiment["shearmargin_enhancement"]
    A = experiment["A"]
    rho = experiment["rho"]
    n = experiment["n"]
    gmag = experiment["gmag"]
    alpha = experiment["alpha"]

    Alin = A * 2.2e10  # Enhancement factor for linear viscious problem

    if experiment["model_half"]:
        mesh = RectangleMesh(Point(0, 0), Point(+domain_w / 2, domain_h), resolution_w, resolution_h)
    else:
        mesh = RectangleMesh(Point(-domain_w / 2, 0), Point(+domain_w / 2, domain_h), resolution_w, resolution_h)

    Uele = VectorElement("CG", mesh.ufl_cell(), degree=2, dim=3)
    Pele = FiniteElement("CG", mesh.ufl_cell(), 1)

    MixedEle = MixedElement([Uele, Pele])

    U = VectorFunctionSpace(mesh, "CG", 2, dim=3)
    P = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, MixedEle)

    (u, p) = TrialFunctions(W)  # the unknowns
    (v, q) = TestFunctions(W)  # the weighting funcs
    w = Function(W)

    # Define BCs
    bottom_noslip = lambda x, on_boundary: on_boundary and near(x[1], 0) and (abs(x[0]) >= icestream_width / 2)
    bottom = lambda x, on_boundary: on_boundary and near(x[1], 0)
    side = lambda x, on_boundary: on_boundary and near(abs(x[0]), domain_w / 2)
    top = lambda x, on_boundary: on_boundary and near(x[1], domain_h)
    centerline = lambda x, on_boundary: on_boundary and near(x[0], 0)
    anyboundary = lambda x, on_boundary: on_boundary

    hydrostatic_pressure = Expression("dpdz*(H-x[1])", H=domain_h, dpdz=rho * gmag / np.cos(alpha), degree=2)

    bc = [DirichletBC(W.sub(1), Constant(0), top)]

    bc += [DirichletBC(W.sub(0).sub(0), Constant(0), bottom_noslip)]
    bc += [DirichletBC(W.sub(0).sub(2), Constant(0), bottom_noslip)]
    bc += [DirichletBC(W.sub(0).sub(1), Constant(0), bottom)]
    bc += [DirichletBC(W.sub(0).sub(0), Constant(0), side)]
    if experiment["model_half"]:
        bc += [DirichletBC(W.sub(0).sub(0), Constant(0), centerline)]

    g = Constant((0, -cos(alpha) * gmag * rho, sin(alpha) * gmag * rho))  # grav vec
    L = inner(v, g) * dx

    E_spatial = Expression("1+E*exp(-0.5*pow((pos-abs(x[0]))/sigma,2))", pos=icestream_width, sigma=1e3, E=shearmargin_enhancement, degree=2)

    # https://bitbucket.org/fenics-project/dolfin/issues/252/function-assignment-failing-with-mixed
    p0 = interpolate(hydrostatic_pressure, P)
    assign(w.sub(1), p0)

    solver_parameters = {"linear_solver": "mumps", "preconditioner": "petsc_amg"}
    solve(a(1, Alin * E_spatial, u, v, p, q) == L, w, bc, solver_parameters=solver_parameters)
    (usol_lin, psol_lin) = w.split(deepcopy=True)

    if n != 1:  # NLIN
        print("NON-LINEAR SOLVE!")
        F = a(n, A * E_spatial, u, v, p, q) - L
        R = action(F, w)
        DR = derivative(R, w)  # Gateaux derivative
        problem = NonlinearVariationalProblem(R, w, bc, DR)
        solver = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        prm["newton_solver"]["linear_solver"] = "mumps"  # ?
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["absolute_tolerance"] = 1e-6
        prm["newton_solver"]["relaxation_parameter"] = 0.41
        prm["newton_solver"]["maximum_iterations"] = 100  # 100?
        prm["newton_solver"]["convergence_criterion"] = "incremental"
        # -------------------
        prm["newton_solver"]["report"] = True
        prm["newton_solver"]["error_on_nonconvergence"] = True
        prm["newton_solver"]["krylov_solver"]["report"] = True
        prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = True
        prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
        prm["newton_solver"]["krylov_solver"]["error_on_nonconvergence"] = False
        # set_log_level(LogLevel.PROGRESS)
        solver.solve()
        # set_log_level(LogLevel.ERROR)
    else:
        print(f"n=1 - so skipping non-linear solver.")

    (usol, psol) = w.split(deepcopy=True)

    fname = f"{settings.outputfolder}/2d_{experiment['name']}.h5"
    solution_io.save_solution(fname, mesh, usol, psol, experiment)


if __name__ == "__main__":
    cross_section()
    cross_section(name="wider_icestream", icestream_width=22e3)

