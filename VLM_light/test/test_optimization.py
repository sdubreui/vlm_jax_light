# Example of optimization script using jax auto-diff capabibilities
import os
# os.environ["JAX_LOG_COMPILES"] = "1"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

from VLM_light.VLM import VlmStudyOptimized
from extract_cli_distribution import parse_lod_file
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
from scipy.optimize import minimize, OptimizeResult
import time as t

# Optimal twist distribution under CL constraint
# initial configuration
initial_mesh = 'meshes/rectangular_wing_20_40.msh'
# create a VLM study with this mesh 
alpha_0 = 2.0
v_inf = 100.0
rho = 0.0023770
S_ref = 10.0
my_study = VlmStudyOptimized(initial_mesh,alpha_0,v_inf = v_inf,rho = rho,symmetry=True,x_wake = 1e6)
surfaces = my_study.compute_topology()
# sections where to define the twist control points 
nodes = jnp.array(my_study.nodes)
n_cp = 20
cp_span = jnp.linspace(nodes[:,2].min(),nodes[:,2].max(),n_cp+1)

def obj_fun(X):
    # X is the vector of the twist distribution at control points
    # get the initial nodes coordinates and sort them according to y coordinate
    #sort according to y coordinate
    coords = nodes[:,1:]
    node_tags = nodes[:,0]
    ind = coords[:,1].argsort()
    sorted_coords = coords[ind]
    n_sections = 41
    #linear interpolation between control points
    y_sections = sorted_coords[::21,1]
    Theta = jnp.interp(y_sections,cp_span,X)
    for i in range(n_sections) :
        y_section = sorted_coords[i*21:(i+1)*21,1]
        x_section = sorted_coords[i*21:(i+1)*21,0]
        z_section = sorted_coords[i*21:(i+1)*21,2]
        #Twist of the section
        theta = Theta[i]
        #apply deformation to the x coordinate (assuming rotation around 25% of te chord)
        x_origin = x_section.min() + 0.25*(x_section.max()-x_section.min()) 
        delta_x = x_section - x_origin
        x_section_deformed = x_origin + delta_x*jnp.cos(theta) - z_section*jnp.sin(theta)
        z_section_deformed = z_section*jnp.cos(theta) + delta_x*jnp.sin(theta)
        #update the coordinates
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,0].set(x_section_deformed)
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,2].set(z_section_deformed)
    # Rebuild full coordinate array in original order
    coords_deformed = jnp.zeros_like(coords)
    coords_deformed = coords_deformed.at[ind].set(sorted_coords)  # undo sorting
    segments, segments_ids, control_points, normals, control_point_quart,ring_pts = my_study.compute_geometry(coords_deformed, alpha_0)
    gamma = my_study.compute_circulation_parametrized(segments,segments_ids, control_points,normals,alpha_0,my_study.v_inf)
    CL,CD,forces_panel,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma,alpha_0,S_ref,segments,control_point_quart,segments_ids, ring_pts)
    return CD

#constraints on CL
def h(X):
    # X is the vector of the twist distribution at control points
    # get the initial nodes coordinates and sort them according to y coordinate
    #sort according to y coordinate
    coords = nodes[:,1:]
    node_tags = nodes[:,0]
    ind = coords[:,1].argsort()
    sorted_coords = coords[ind]
    n_sections = 41
    #linear interpolation between control points
    y_sections = sorted_coords[::21,1]
    Theta = jnp.interp(y_sections,cp_span,X)
    for i in range(n_sections) :
        y_section = sorted_coords[i*21:(i+1)*21,1]
        x_section = sorted_coords[i*21:(i+1)*21,0]
        z_section = sorted_coords[i*21:(i+1)*21,2]
        #Twist of the section
        theta = Theta[i]
        #apply deformation to the x coordinate (assuming rotation around 25% of te chord)
        x_origin = x_section.min() + 0.25*(x_section.max()-x_section.min()) 
        delta_x = x_section - x_origin
        x_section_deformed = x_origin + delta_x*jnp.cos(theta) - z_section*jnp.sin(theta)
        z_section_deformed = z_section*jnp.cos(theta) + delta_x*jnp.sin(theta)
        #update the coordinates
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,0].set(x_section_deformed)
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,2].set(z_section_deformed)
    # Rebuild full coordinate array in original order
    coords_deformed = jnp.zeros_like(coords)
    coords_deformed = coords_deformed.at[ind].set(sorted_coords)  # undo sorting
    segments, segments_ids, control_points, normals, control_point_quart,ring_pts = my_study.compute_geometry(coords_deformed, alpha_0)
    gamma = my_study.compute_circulation_parametrized(segments,segments_ids, control_points,normals,alpha_0,my_study.v_inf)
    CL,CD,forces_panel,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma,alpha_0,S_ref,segments,control_point_quart,segments_ids, ring_pts)
    return CL-0.5

X = jnp.ones(n_cp+1)*(2.0*jnp.pi/180.0) # initial guess for the twist distribution
CD = obj_fun(X)
print("Initial CD=", CD)
grad_f = jax.grad(obj_fun)
print("Gradient at initial guess=", grad_f(X))
print("Constraint value at initial guess=", h(X))
grad_h = jax.grad(h)
print("Gradient of constraint at initial guess=", grad_h(X))    

# Compiler les fonctions avec JIT
f_jit = jax.jit(obj_fun)
grad_f_jit = jax.jit(grad_f)
h_jit = jax.jit(h)
grad_h_jit = jax.jit(grad_h)

#conversion to scipy to use scipy optimize SLSQP 
def f_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(f_jit(jnp.array(x, dtype=jnp.float64)))

def grad_f_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.array(grad_f_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

def h_numpy(x):
    # Ensure input to JAX is float64 and output is float64
    return np.float64(h_jit(jnp.array(x, dtype=jnp.float64)))

def grad_h_numpy(x):
    # Ensure input to JAX is float64 and output numpy array is float64
    return np.array(grad_h_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

# Point initial
x0 = np.array(X, dtype=np.float64) # Explicitly set x0 to float64

t1 =t.time()
CD = f_jit(x0)
t2 = t.time()
print((t2-t1)*1000.0, "ms for first evaluation of f_jit")
CD = f_jit(x0).block_until_ready()
t3 = t.time()
print((t3-t2)*1000.0, "ms for second evaluation of f_jit")



# Définir la contrainte au format scipy
constraint = {
    'type': 'eq',
    'fun': h_numpy,
    'jac': grad_h_numpy
}

# callback function to monitor optimization progress
# Historique
history = []
def callback(intermediate_result: OptimizeResult):
    xk = intermediate_result.x
    fk = intermediate_result.fun  

    print(f"Iteration: x = {xk}, f(x) = {fk}")
    history.append((xk.copy(), fk))

# Optimisation avec SLSQP

result = minimize(
    fun=f_numpy,
    x0=x0,
    method='SLSQP',
    jac=grad_f_numpy,
    constraints=constraint,
    bounds = [(-10.0*jnp.pi/180.0,10.0*jnp.pi/180.0)]*(n_cp+1), # bounds on the twist distribution
    options={'disp': True,'maxiter': 15},
    callback=callback
)

#plot the evolution of the twist distribution and the CL,CD during the optimization
CL_history = []
CD_history = []
CL_distribution_history = []
for xk, fk in history:
    print("xk=", xk, "fk=", fk)
    coords = nodes[:,1:]
    node_tags = nodes[:,0]
    ind = coords[:,1].argsort()
    sorted_coords = coords[ind]
    n_sections = 41
    #linear interpolation between control points
    y_sections = sorted_coords[::21,1]
    Theta = jnp.interp(y_sections,cp_span,xk)
    for i in range(n_sections) :
        y_section = sorted_coords[i*21:(i+1)*21,1]
        x_section = sorted_coords[i*21:(i+1)*21,0]
        z_section = sorted_coords[i*21:(i+1)*21,2]
        #Twist of the section
        theta = Theta[i]
        #apply deformation to the x coordinate (assuming rotation around 25% of te chord)
        x_origin = x_section.min() + 0.25*(x_section.max()-x_section.min()) 
        delta_x = x_section - x_origin
        x_section_deformed = x_origin + delta_x*jnp.cos(theta) - z_section*jnp.sin(theta)
        z_section_deformed = z_section*jnp.cos(theta) + delta_x*jnp.sin(theta)
        #update the coordinates
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,0].set(x_section_deformed)
        sorted_coords = sorted_coords.at[i*21:(i+1)*21,2].set(z_section_deformed)
    # Rebuild full coordinate array in original order
    coords_deformed = jnp.zeros_like(coords)
    coords_deformed = coords_deformed.at[ind].set(sorted_coords)  # undo sorting
    segments, segments_ids, control_points, normals, control_point_quart,ring_pts = my_study.compute_geometry(coords_deformed,  alpha_0)
    gamma = my_study.compute_circulation_parametrized(segments,segments_ids, control_points,normals,alpha_0,my_study.v_inf)
    CL,CD,forces_panel,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma,alpha_0,S_ref,segments,control_point_quart,segments_ids, ring_pts)
    y_span, cl_local, cl_distribution = my_study.compute_cl_distribution_span(delta_L, ring_pts)
    CL_history.append(CL)
    CD_history.append(CD)
    CL_distribution_history.append(cl_distribution)

import matplotlib.pyplot as plt
plt.figure()
i = 0
for CL_dist in CL_distribution_history:
    plt.plot(CL_dist['y_column'],CL_dist['cl_integrated']/20.0,label=f'Iter {i}')
    i+=1
#eliptic distribution
AR = 10.0
y_sections = CL_distribution_history[0]['y_column']
dist_elliptic = 4*0.5/(np.pi*AR)*np.sqrt(1-(y_sections/10)**2)    
plt.plot(y_sections,dist_elliptic,label='Elliptic distribution',color='k',linestyle='--') 
plt.xlabel('Spanwise coordinate')
plt.ylabel('Integrated CL')
plt.title('Evolution of CL distribution during optimization')   
plt.legend(loc=0)
plt.grid(True)
plt.show()