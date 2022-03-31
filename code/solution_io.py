from fenics import *
from ast import literal_eval


def save_solution(fname, mesh, usol, psol, experiment={}):
    with HDF5File(mesh.mpi_comm(), fname, "w") as hdf:
        usol.rename("u", "velocity")
        psol.rename("p", "pressure")
        hdf.write(mesh, "/mesh")
        hdf.write(usol, "/usol")
        hdf.write(psol, "/psol")
        att = hdf.attributes("/mesh")
        att["experiment"] = str(experiment)


def load_solution(fname):
    with HDF5File(MPI.comm_self, fname, "r") as hdf:
        mesh = Mesh()
        hdf.read(mesh, "/mesh", False)
        fs_u = VectorFunctionSpace(mesh, "CG", degree=2, dim=3)
        fs_p = FunctionSpace(mesh, "CG", 1)
        usol = Function(fs_u)
        psol = Function(fs_p)
        hdf.read(usol, "/usol")
        hdf.read(psol, "/psol")
        att = hdf.attributes("/mesh")
        experiment = literal_eval(att["experiment"])
    return {"mesh": mesh, "u": usol, "p": psol, "experiment": experiment}


def save_xdmf(fname, mesh, usol, psol, metadata_dict):
    with XDMFFile(fname) as xf:
        # xf.parameters.update(
        # {
        #    "functions_share_mesh": True,
        #    "rewrite_function_mesh": False
        # })
        usol.rename("u", "velocity")
        psol.rename("p", "pressure")
        xf.write(usol, 0)
        xf.write(psol, 1)
        xf.write_checkpoint(usol, "usol", 0, XDMFFile.Encoding.HDF5, True)
        xf.write_checkpoint(psol, "psol", 0, XDMFFile.Encoding.HDF5, True)
        # att = hdf.attributes("/mesh")
        # att["metadata_dict"] = str(metadata_dict)


def load_xdmf(fname):
    with XDMFFile(fname) as xf:
        mesh = Mesh()
        xf.read(mesh)
        fs_u = VectorFunctionSpace(mesh, "CG", degree=2, dim=3)
        fs_p = FunctionSpace(mesh, "CG", 1)
        usol = Function(fs_u)
        psol = Function(fs_p)
        xf.read_checkpoint(usol, "usol", 0)
        xf.read_checkpoint(psol, "psol", 0)
        # att = hdf.attributes("/mesh")
        # metadata_dict= literal_eval(att["metadata_dict"])
    return (mesh, usol, psol, metadata_dict)
