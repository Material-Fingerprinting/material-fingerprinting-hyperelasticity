import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl

from dolfinx import fem, io, log, mesh, nls, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

np.random.seed(0)

"""
general settings
"""

VERBOSE = True
PLOT = True
SAVE = True
SAVE_NORMALIZED = True

# set noise
# the standard deviation of the noise is set to NOISE times the absolute maximum reaction force or displacement 
NOISE = 0.0
NOISE = 0.001
NOISE = 0.01
NOISE = 0.05

"""
models
"""

NPARA = 11
IDX_HOMOGENEITY = np.arange(8)
IDX_NONHOMOGENEITY = np.arange(8, NPARA)

def strain_energy_density(I1_bar,I2_bar,J,p):
    # p is a set of 11 material parameters (8 homogeneity parameters and 3 non-homogeneity parameters)
    W = p[0] * (J - 1)**(2.0)
    W += p[1] * (I1_bar - 3)
    W += p[2] * (I1_bar - 3)**(2.0)
    W += p[3] * (I1_bar - 3)**(3.0)
    W += p[4] * (I2_bar - 3)
    W += p[5] * (ufl.exp(p[8] * (I1_bar - 3)) - 1.0) # Demiray
    W += p[6] * (ufl.exp(p[9] * (I1_bar - 3)**(2.0)) - 1.0) # Holzapfel
    W += - p[7] * (ufl.ln(1.0 - p[10] * (I1_bar - 3))) # Gent
    return W

# compressible Blatz-Ko
p_Blatz = np.zeros((NPARA))
p_Blatz[0] = 5.0
p_Blatz[4] = 50.0

# compressible Demiray
p_Demiray = np.zeros((NPARA))
p_Demiray[0] = 5.0
p_Demiray[5] = 10.0
p_Demiray[8] = 8.0

# compressible Mooney-Rivlin
p_Mooney = np.zeros((NPARA))
p_Mooney[0] = 20.0
p_Mooney[1] = 10.0
p_Mooney[4] = 40.0

# compressible Neo-Hooke
p_Neo = np.zeros((NPARA))
p_Neo[0] = 20.0
p_Neo[1] = 10.0

parameters = [
    p_Neo,
    p_Blatz,
    p_Mooney,
    p_Demiray,
]

names = [
    "NeoHooke",
    "Blatz",
    "MooneyRivlin",
    "Demiray",
]

NSIM = len(parameters)

"""
simulation settings
"""

UMAX = -0.3
NSTEP = 10
NREACTION = 2
TOL = 1e-8

"""
mesh
"""

mesh_name = "gmsh/plate_with_corner_hole_5e-2_triangle"
L = 1 # side length (must match with the given geometry)
radius = 0.3 # radius of the arc (must match with the given geometry)
with io.XDMFFile(MPI.COMM_WORLD, mesh_name + ".xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(name="mesh")
V = fem.functionspace(domain, ("Lagrange", 1, (domain.topology.dim, )))
metadata = {"quadrature_degree": 4}

"""
boundaries
"""

def boundary_left(x): return np.isclose(x[0], 0.0)
def boundary_right(x): return np.isclose(x[0], 1.0)
def boundary_bottom(x): return np.isclose(x[1], 0.0)
def boundary_top(x): return np.isclose(x[1], 1.0)
def boundary_arc(x): return np.isclose((x[0]-L)**2 + (x[1]-L)**2, radius**2)

fdim = domain.topology.dim - 1
facets_left = mesh.locate_entities_boundary(domain, fdim, boundary_left)
facets_right = mesh.locate_entities_boundary(domain, fdim, boundary_right)
facets_bottom = mesh.locate_entities_boundary(domain, fdim, boundary_bottom)
facets_top = mesh.locate_entities_boundary(domain, fdim, boundary_top)
facets_arc = mesh.locate_entities_boundary(domain, fdim, boundary_arc)

# Concatenate and sort the arrays based on facet indices.
facets_marked = np.hstack([facets_left, facets_right, facets_bottom, facets_top, facets_arc])
values_marked = np.hstack([
    np.full_like(facets_left, 1),
    np.full_like(facets_right, 2),
    np.full_like(facets_bottom, 3),
    np.full_like(facets_top, 4),
    np.full_like(facets_arc, 5),
    ])
facets_sorted = np.argsort(facets_marked)
facets_meshtags = mesh.meshtags(domain, fdim, facets_marked[facets_sorted], values_marked[facets_sorted])

dofs_left = fem.locate_dofs_topological(V, facets_meshtags.dim, facets_meshtags.find(1)) # left: tag 1
dofs_right = fem.locate_dofs_topological(V, facets_meshtags.dim, facets_meshtags.find(2)) # right: tag 2
dofs_bottom = fem.locate_dofs_topological(V, facets_meshtags.dim, facets_meshtags.find(3)) # bottom: tag 3
dofs_top = fem.locate_dofs_topological(V, facets_meshtags.dim, facets_meshtags.find(4)) # top: tag 4
dofs_arc = fem.locate_dofs_topological(V, facets_meshtags.dim, facets_meshtags.find(5)) # arc: tag 5

dofs_left_x = fem.locate_dofs_topological(V.sub(0), facets_meshtags.dim, facets_meshtags.find(1))
dofs_right_x = fem.locate_dofs_topological(V.sub(0), facets_meshtags.dim, facets_meshtags.find(2))
dofs_bottom_y = fem.locate_dofs_topological(V.sub(1), facets_meshtags.dim, facets_meshtags.find(3))
dofs_top_y = fem.locate_dofs_topological(V.sub(1), facets_meshtags.dim, facets_meshtags.find(4))
dofs_arc_x = fem.locate_dofs_topological(V.sub(0), facets_meshtags.dim, facets_meshtags.find(5))
dofs_arc_y = fem.locate_dofs_topological(V.sub(1), facets_meshtags.dim, facets_meshtags.find(5))

NDOFSARC = len(dofs_arc_x)

"""
test and solution functions
"""

u = fem.Function(V)
u.name = "u"
v = ufl.TestFunction(V)
v_left_x = fem.Function(V)
v_left_x.x.array[:] = 0.0
v_left_x.x.array[dofs_left_x] = 1.0
v_bottom_y = fem.Function(V)
v_bottom_y.x.array[:] = 0.0
v_bottom_y.x.array[dofs_bottom_y] = 1.0

"""
kinematics
"""

def Grad(u):
    return ufl.as_tensor([[u[0].dx(0), u[0].dx(1), 0.0],
                        [u[1].dx(0), u[1].dx(1), 0.0],
                        [0.0, 0.0, 0.0]])
Grad_u = Grad(u)
F = ufl.variable(ufl.Identity(3) + Grad_u)
J = ufl.det(F)
C = F.T * F
I1 = ufl.tr(C)
I2 = (1.0/2.0) * (ufl.tr(C) ** 2 - ufl.tr(C * C))
I3 = J**2
I1_bar = J**(-2.0/3.0) * I1
I2_bar = J**(-4.0/3.0) * I2

"""
strain energy density & first Piola-Kirchhoff stress
"""

parameters_constant = fem.Constant(domain, np.zeros(NPARA)) # fem.Constant() is important when the parameters are modified later
W = strain_energy_density(I1_bar,I2_bar,J,parameters_constant)
P = ufl.diff(W, F)

"""
variational form (no body force and no traction)
"""

dx = ufl.Measure("dx", domain=domain, metadata=metadata)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facets_meshtags, metadata=metadata)
W_int = ufl.inner(P,Grad(v)) * dx
W_int_left_x = fem.form(ufl.action(W_int,v_left_x))
W_int_bottom_y = fem.form(ufl.action(W_int,v_bottom_y))

"""
simulation
"""

def simulate(_parameters):

    parameters_constant.value = _parameters

    u.x.array[:] = 0.0

    f_reaction_x = np.zeros((NSTEP+1))
    f_reaction_y = np.zeros((NSTEP+1))
    f_u_x = np.zeros((NDOFSARC,NSTEP+1))
    f_u_y = np.zeros((NDOFSARC,NSTEP+1))

    for step in range(NSTEP + 1):
        
        bcs = [
            fem.dirichletbc(0.0, dofs_right_x, V.sub(0)),
            fem.dirichletbc(0.0, dofs_top_y, V.sub(1)),
            fem.dirichletbc(0.5 * step * UMAX / NSTEP, dofs_left_x, V.sub(0)),
            fem.dirichletbc(step * UMAX / NSTEP, dofs_bottom_y, V.sub(1)),
            ]
        problem = NonlinearProblem(W_int, u, bcs)
        solver = NewtonSolver(domain.comm, problem)
        
        solver.atol = solver.rtol = TOL
        solver.convergence_criterion = "incremental"

        num_its, converged = solver.solve(u)
        assert (converged)
        u.x.scatter_forward()

        """
        measure fingerprints
        """
        
        f_reaction_x[step] = fem.assemble_scalar(W_int_left_x)
        f_reaction_y[step] = fem.assemble_scalar(W_int_bottom_y)
        f_u_x[:,step] = u.x.array[dofs_arc_x]
        f_u_y[:,step] = u.x.array[dofs_arc_y]

    f_reaction = np.concatenate([f_reaction_x, f_reaction_y])
    # norm = np.linalg.norm(f_reaction)
    # f_reaction_normalized = f_reaction / norm
    f_u = np.concatenate([f_u_x.reshape(-1), f_u_y.reshape(-1)])

    return f_reaction, f_u

for sim in range(NSIM):
    print("Run simulation for " + names[sim] + " model.")
    f_reaction, f_u = simulate(parameters[sim])
    if NOISE > 0.0:
        f_reaction += np.random.normal(loc=0.0, scale=NOISE * np.max(np.abs(f_reaction)), size=len(f_reaction))
        f_u += np.random.normal(loc=0.0, scale=NOISE * np.max(np.abs(f_u)), size=len(f_u))

    path = "data/experiment_unsupervised11_" + names[sim] + "_noise" + str(NOISE) + ".npz"
    np.savez(path,
            parameters=parameters[sim],
            f_reaction=f_reaction,
            f_u=f_u,
            NOISE = NOISE,
            IDX_HOMOGENEITY=IDX_HOMOGENEITY,
            IDX_NONHOMOGENEITY=IDX_NONHOMOGENEITY,
            NSTEP=NSTEP,
            NREACTION=NREACTION,
            NDOFSARC=NDOFSARC
            )






