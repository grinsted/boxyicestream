"""

This is the 3d boxy icestream code.

"""
# flake8: noqa ignore=F405
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

import settings
import solution_io


def run_model3d(experiment):
    print("Running 3d experiment: ", experiment["name"])
    fname = settings.filename3d(experiment)
    domain_w = experiment["domain_w"]
    domain_h = experiment["domain_h"]
    domain_l = experiment["domain_l"]
    resolution_w = experiment["resolution_w"]
    resolution_h = experiment["resolution_h"]
    resolution_l = experiment["resolution_l"]
    icestream_width = experiment["icestream_width"]
    shearmargin_enhancement = experiment["shearmargin_enhancement"]
    shearmargin_enhancement_pos = experiment["shearmargin_enhancement_pos"]
    A = experiment["A"]
    rho = experiment["rho"]
    n = experiment["n"]
    gmag = experiment["gmag"]
    alpha = experiment["alpha"]

    Alin = A * 2.2e10  # Enhancement factor for linear viscious problem

    if experiment["model_half"]:
        mesh = BoxMesh(Point(0, 0, 0), Point(domain_l, +domain_w / 2, domain_h), resolution_l, resolution_w, resolution_h)
    else:
        mesh = BoxMesh(Point(0, -domain_w / 2, 0), Point(domain_l, +domain_w / 2, domain_h), resolution_l, resolution_w, resolution_h)

    Uele = VectorElement("CG", mesh.ufl_cell(), 2)
    Pele = FiniteElement("CG", mesh.ufl_cell(), 1)

    MixedEle = MixedElement([Uele, Pele])

    U = VectorFunctionSpace(mesh, "CG", 2)
    P = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, MixedEle)

    (u, p) = TrialFunctions(W)  # the unknowns
    (v, q) = TestFunctions(W)  # the weighting funcs
    w = Function(W)

    e_x = Expression(("1.0", "0.0", "0.0"), element=Uele)
    e_y = Expression(("0.0", "1.0", "0.0"), element=Uele)
    e_z = Expression(("0.0", "0.0", "1.0"), element=Uele)

    # %%
    # SETUP BCs, L, and a

    bottom_noslip = lambda x, on_boundary: on_boundary and near(x[2], 0) and (abs(x[1]) >= icestream_w / 2)
    bottom = lambda x, on_boundary: on_boundary and near(x[2], 0)
    side = lambda x, on_boundary: on_boundary and near(abs(x[1]), domain_w / 2)
    icedivide = lambda x, on_boundary: on_boundary and near(abs(x[0]), 0)
    front = lambda x, on_boundary: on_boundary and near(x[0], domain_l)
    top = lambda x, on_boundary: on_boundary and near(x[2], domain_h)
    centerline = lambda x, on_boundary: on_boundary and near(x[1], 0)
    anyboundary = lambda x, on_boundary: on_boundary

    hydrostatic_pressure = Expression("dpdz*(H-x[2])", H=domain_h, dpdz=rho * gmag / np.cos(alpha), degree=2)

    bc = [DirichletBC(W.sub(1), Constant(0), top)]

    bc += [DirichletBC(W.sub(0).sub(0), Constant(0), bottom_noslip)]
    bc += [DirichletBC(W.sub(0).sub(1), Constant(0), bottom_noslip)]
    bc += [DirichletBC(W.sub(0).sub(2), Constant(0), bottom)]
    bc += [DirichletBC(W.sub(0).sub(1), Constant(0), side)]
    bc += [DirichletBC(W.sub(0).sub(0), Constant(0), icedivide)]
    if model_half:
        bc += [DirichletBC(W.sub(0).sub(1), Constant(0), centerline)]

    # bc += [DirichletBC(W.sub(0).sub(1), Constant(0), front)]
    # bc += [DirichletBC(W.sub(0).sub(2), Constant(0), front)]
    # bc += [DirichletBC(W.sub(1), hydrostatic_pressure, icedivide)]
    # bc += [DirichletBC(W.sub(1), hydrostatic_pressure, front)]

    # set pressure at front boundary to 2d solution
    # fname2d = settings.filename2d(experiment)
    # result2d = solution_io.load_solution(fname2d)
    # p2d = result2d["p2d"]
    # HOW: https://fenicsproject.discourse.group/t/how-to-use-2d-solution-result-as-initial-value-of-3d-mesh/4867/4

    def strainrate(u):
        return sym(grad(u))

    def I2(eps, p):
        return np.power(inner(eps, eps), p)

    def nu(eps, A, n):
        return A ** (-1.0 / n) * I2(eps, (1.0 - n) / (2.0 * n))

    def tau(u, A, n):
        eps = strainrate(u)
        return 2 * nu(eps, A, u) * eps

    def a(n, A):
        eps = strainrate(u)
        return (nu(eps, A, n) * inner(sym(grad(v)), eps) - div(v) * p + q * div(u)) * dx

    g = Constant((sin(alpha) * gmag * rho, 0, -cos(alpha) * gmag * rho))  # grav vec
    L = inner(v, g) * dx

    # https://bitbucket.org/fenics-project/dolfin/issues/252/function-assignment-failing-with-mixed
    p0 = interpolate(hydrostatic_pressure, P)
    assign(w.sub(1), p0)

    solver_parameters = {"linear_solver": "mumps", "preconditioner": "petsc_amg"}
    solve(a(1, A * Elin) == L, w, bc, solver_parameters=solver_parameters)

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
        prm["newton_solver"]["report"] = False
        prm["newton_solver"]["error_on_nonconvergence"] = True
        prm["newton_solver"]["krylov_solver"]["report"] = False
        prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
        prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
        prm["newton_solver"]["krylov_solver"]["error_on_nonconvergence"] = False
        # set_log_level(LogLevel.PROGRESS)
        solver.solve()
        # set_log_level(LogLevel.ERROR)
    else:
        print(f"n=1 - so skipping non-linear solver.")

    (usol, psol) = w.split(deepcopy=True)

    print("saving to", fname)
    solution_io.save_solution(fname, mesh, usol, psol, experiment)

