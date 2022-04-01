# flake8: noqa ignore=F405
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

import settings
import solution_io
import ice_physics


def run_experiment(experiment):
    print("\nRunning 2d experiment: ", experiment["name"])
    fname = settings.filename2d(experiment)
    domain_w = experiment["domain_w"]
    domain_h = experiment["domain_h"]
    resolution_w = experiment["resolution_w"]
    resolution_h = experiment["resolution_h"]
    icestream_width = experiment["icestream_width"]
    shearmargin_enhancement = experiment["shearmargin_enhancement"]
    shearmargin_enhancement_pos = experiment["shearmargin_enhancement_pos"]
    A = experiment["A"]
    rho = experiment["rho"]
    n = experiment["n"]
    gmag = experiment["gmag"]
    alpha = experiment["alpha"]
    beta = experiment["weertman_beta"]

    settings.print_experiment_highlights(experiment)

    Alin = A * 2.2e10  # Enhancement factor for linear viscious problem

    if experiment["model_half"]:
        mesh = RectangleMesh(Point(0, 0), Point(+domain_w / 2, domain_h), resolution_w, resolution_h)
    else:
        mesh = RectangleMesh(Point(-domain_w / 2, 0), Point(+domain_w / 2, domain_h), resolution_w, resolution_h)

    # mesh refinement.
    for x in mesh.coordinates():
        x[1] = (x[1] / domain_h) ** 1.8 * domain_h
        if abs(x[0]) < (icestream_width):
            x[0] += np.sin((x[0] / icestream_width - 1) * 2 * np.pi) * (icestream_width / 10)

    # plot(mesh)
    # plt.axis("auto")

    Uele = VectorElement("CG", mesh.ufl_cell(), degree=2, dim=3)
    Pele = FiniteElement("CG", mesh.ufl_cell(), 1)

    MixedEle = MixedElement([Uele, Pele])

    U = VectorFunctionSpace(mesh, "CG", 2, dim=3)
    P = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, MixedEle)

    (u, p) = TrialFunctions(W)  # the unknowns
    (v, q) = TestFunctions(W)  # the weighting funcs
    w = Function(W)

    near = lambda a, b: abs(a - b) < 0.1

    # Define BCs
    bottom_noslip = lambda x, on_boundary: on_boundary and near(x[1], 0) and (abs(x[0]) >= icestream_width / 2)
    bottom = lambda x, on_boundary: on_boundary and near(x[1], 0)

    side = lambda x, on_boundary: on_boundary and near(abs(x[0]), domain_w / 2)
    top = lambda x, on_boundary: on_boundary and near(x[1], domain_h)
    centerline = lambda x, on_boundary: on_boundary and near(x[0], 0)

    class bottom_weertman(SubDomain):
        def inside(self, x, on_boundary):
            return bottom(x, on_boundary) and not bottom_noslip(x, on_boundary)

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    bottom_weertman().mark(boundaries, 1)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

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

    E_spatial = Expression(
        "1+E*exp(-0.5*pow((pos-abs(x[0]))/sigma,2))", pos=shearmargin_enhancement_pos, sigma=1e3, E=shearmargin_enhancement, degree=2,
    )

    def a_fun(n):
        if n == 1:
            AE = Alin * E_spatial
        else:
            AE = A * E_spatial
        eps = ice_physics.strainrate2D(u)
        tau = ice_physics.tau(eps, AE, n)
        a = (inner(sym(ice_physics.grad2D(v)), tau) - ice_physics.div2D(v) * p + q * ice_physics.div2D(u)) * dx
        a += beta * dot(v, u) * ds(1)
        return a

    # Weertman - linear in v

    # https://bitbucket.org/fenics-project/dolfin/issues/252/function-assignment-failing-with-mixed
    p0 = interpolate(hydrostatic_pressure, P)
    assign(w.sub(1), p0)

    solver_parameters = {"linear_solver": "mumps", "preconditioner": "petsc_amg"}
    solve(a_fun(n=1) == L, w, bc, solver_parameters=solver_parameters)
    (usol_lin, psol_lin) = w.split(deepcopy=True)

    if n != 1:  # NLIN
        F = a_fun(n) - L
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

    print("saving to ", fname)
    solution_io.save_solution(fname, mesh, usol, psol, experiment)
    return {"mesh": mesh, "u": usol, "p": psol, "experiment": experiment}


if __name__ == "__main__":

    results = run_experiment(settings.experiment())
    print(results["u"](1000, 500))
