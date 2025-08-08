import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl

from dolfinx import fem, io, log, mesh, nls, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI

"""
general settings
"""

VERBOSE = True
PLOT = True
SAVE = True
SAVE_NORMALIZED = True

"""
models
"""

NPARA = 11
NDISCRETIZE = 100
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
    W += p[6] * (ufl.exp(p[9] * (I1_bar - 3)**2) - 1.0) # Holzapfel
    W += - p[7] * (ufl.ln(1.0 - p[10] * (I1_bar - 3))) # Gent
    return W

vary_positive = np.linspace(0.1, 10.0, NDISCRETIZE)
vary_smaller_one = np.linspace(0.1, 1.0, NDISCRETIZE)

vary_theta_power2_a, vary_theta_power2_b = np.meshgrid(vary_positive, vary_positive, indexing='ij')
vary_theta_power2_a = vary_theta_power2_a.ravel()
vary_theta_power2_b = vary_theta_power2_b.ravel()
vary_theta_power3_a, vary_theta_power3_b, vary_theta_power3_c = np.meshgrid(vary_positive, vary_positive, vary_positive, indexing='ij')
vary_theta_power3_a = vary_theta_power3_a.ravel()
vary_theta_power3_b = vary_theta_power3_b.ravel()
vary_theta_power3_c = vary_theta_power3_c.ravel()

vary_Gent_power2_a, vary_Gent_power2_b = np.meshgrid(vary_positive, vary_smaller_one, indexing='ij')
vary_Gent_power2_a = vary_Gent_power2_a.ravel()
vary_Gent_power2_b = vary_Gent_power2_b.ravel()

# compressible Neo-Hooke
p_Neo = np.zeros((NDISCRETIZE,NPARA))
p_Neo[:, 0] = 1.0
p_Neo[:, 1] = vary_positive

# compressible Blatz-Ko
p_Blatz = np.zeros((NDISCRETIZE,NPARA))
p_Blatz[:, 0] = 1.0
p_Blatz[:, 4] = vary_positive

# compressible Mooney-Rivlin
p_Mooney = np.zeros((NDISCRETIZE**2,NPARA))
p_Mooney[:, 0] = 1.0
p_Mooney[:, 1] = vary_theta_power2_a
p_Mooney[:, 4] = vary_theta_power2_b

# compressible Demiray
p_Demiray = np.zeros((NDISCRETIZE**2,NPARA))
p_Demiray[:, 0] = 1.0
p_Demiray[:, 5] = vary_theta_power2_a
p_Demiray[:, 8] = vary_theta_power2_b

# compressible Holzapfel (has convergence issues)
p_Holzapfel = np.zeros((NDISCRETIZE**2,NPARA))
p_Holzapfel[:, 0] = 1.0
p_Holzapfel[:, 6] = vary_theta_power2_a
p_Holzapfel[:, 9] = vary_theta_power2_b

# compressible Gent
p_Gent = np.zeros((NDISCRETIZE**2,NPARA))
p_Gent[:, 0] = 1.0
p_Gent[:, 7] = vary_Gent_power2_a
p_Gent[:, 10] = vary_Gent_power2_b

parameters = np.concatenate([
    p_Neo,
    p_Blatz,
    p_Mooney,
    p_Demiray,
    # p_Holzapfel,
    p_Gent,
])
NSIM = parameters.shape[0]

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
solve
"""

f_reaction = np.zeros((NSIM,NREACTION*(NSTEP+1)))
f_reaction_normalized = np.zeros((NSIM,NREACTION*(NSTEP+1)))
parameters_normalized = np.zeros_like(parameters)
f_u = np.zeros((NSIM,domain.topology.dim*NDOFSARC*(NSTEP+1)))

NFAIL = 0
for sim in range(NSIM):

    parameters_constant.value = parameters[sim,:]

    u.x.array[:] = 0.0

    f_reaction_x = np.zeros((NSTEP+1))
    f_reaction_y = np.zeros((NSTEP+1))
    f_u_x = np.zeros((NDOFSARC,NSTEP+1))
    f_u_y = np.zeros((NDOFSARC,NSTEP+1))

    try:

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
            u.x.scatter_forward()

            """
            measure fingerprints
            """
            
            f_reaction_x[step] = fem.assemble_scalar(W_int_left_x)
            f_reaction_y[step] = fem.assemble_scalar(W_int_bottom_y)
            f_u_x[:,step] = u.x.array[dofs_arc_x]
            f_u_y[:,step] = u.x.array[dofs_arc_y]

        f_reaction[sim,:] = np.concatenate([f_reaction_x, f_reaction_y])
        norm = np.linalg.norm(f_reaction[sim,:])
        f_reaction_normalized[sim,:] = f_reaction[sim,:] / norm
        parameters_normalized[sim,:] = np.concatenate([parameters[sim,IDX_HOMOGENEITY] / norm, parameters[sim,IDX_NONHOMOGENEITY]]) 
        f_u[sim,:] = np.concatenate([f_u_x.reshape(-1), f_u_y.reshape(-1)])
        # f = np.concatenate([f_reaction, f_u])
        # f_normalized = np.concatenate([f_reaction_normalized, f_u])

    except:
        NFAIL += 1
        print("Simulation failed. Moving on.")
        # print("Parameters: ", parameters[sim,:])

    if VERBOSE and (sim % 10 == 0): print("Completed simulations: (" + str(sim) + "/" +  str(NSIM) + ")")

print("Number of failed simulations: ", NFAIL)

"""
We generate a database by saving all fingerprints and parameters. We either save the non-normalized or the normalized fingerprints
and parameters. Here, note that when saving the normalized fingerprints and parameters, only the parameters and the reaction forces
f_reaction are normalized, while the non-normalized displacements f_u are saved. The reason is that the absolute values of f_u
may be needed during the pattern recognition. Normalizing the parameters and the reaction forces does not yield a loss of information.
It can be considered as a change in the force unit. However, normalizing the parameters, the reaction forces and the displacements yields
a loss of information.
"""

if SAVE: 
    np.savez("data/database_unsupervised11_" + str(NSIM) + ".npz",
            parameters=parameters,
            f_reaction=f_reaction,
            f_u=f_u,
            IDX_HOMOGENEITY=IDX_HOMOGENEITY,
            IDX_NONHOMOGENEITY=IDX_NONHOMOGENEITY,
            NDISCRETIZE=NDISCRETIZE,
            NSTEP=NSTEP,
            NREACTION=NREACTION,
            NDOFSARC=NDOFSARC,
            NSIM=NSIM,
            )
    
if SAVE_NORMALIZED:
    np.savez("data/database_unsupervised11_" + str(NSIM) + "_normalized.npz",
            parameters=parameters_normalized,
            f_reaction=f_reaction_normalized,
            f_u=f_u,
            IDX_HOMOGENEITY=IDX_HOMOGENEITY,
            IDX_NONHOMOGENEITY=IDX_NONHOMOGENEITY,
            NDISCRETIZE=NDISCRETIZE,
            NSTEP=NSTEP,
            NREACTION=NREACTION,
            NDOFSARC=NDOFSARC,
            NSIM=NSIM,
            )




