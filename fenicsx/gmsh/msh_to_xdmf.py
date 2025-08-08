import pyvista

from dolfinx import io, plot
from mpi4py import MPI

mesh_path = "gmsh/plate_with_corner_hole_5e-2_triangle"
domain, markers, facets = io.gmshio.read_from_msh(mesh_path + ".msh", MPI.COMM_WORLD)

xdmf_file = io.XDMFFile(MPI.COMM_WORLD, mesh_path + ".xdmf", "w")
xdmf_file.write_mesh(domain)