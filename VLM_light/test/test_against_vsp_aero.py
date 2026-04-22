from VLM_light.VLM import VlmStudyOptimized
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

mesh_file = 'meshes/rectangular_wing_20_40.msh'
Alpha = jnp.linspace(0,10,11)
v_inf = 232.84
rho = 0.40207
S_ref = 10.0
CL_alpha = []
CD_alpha = []
# creation of the study 
my_study = VlmStudyOptimized(mesh_file,Alpha[0],v_inf = v_inf,rho = rho,symmetry=True,x_wake = 1e6)
for alpha in Alpha :
    print("alpha=", alpha)
    alpha = alpha*np.pi/180.0
    surfaces = my_study.compute_topology()
    nodes_coord = my_study.nodes[:,1:]
    segments, segments_ids, control_points, normals, control_point_quart,ring_pts = my_study.compute_geometry(nodes_coord, surfaces, alpha)
    gamma = my_study.compute_circulation_parametrized(segments,segments_ids, control_points,normals,alpha,my_study.v_inf)
    CL,CD,forces_panel,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma,alpha,S_ref,segments,control_point_quart,segments_ids, ring_pts)
    CL,CD,forces,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma)
    CL_alpha.append(CL)
    CD_alpha.append(CD)

#vspaero data 
vsp_data = np.loadtxt('ref_results/rectangular_wing_20_40.polar',skiprows=3)
vsp_Alpha = vsp_data[:,2]
vsp_CL = vsp_data[:,5]
vsp_CDi = vsp_data[:,8]


plt.figure()
plt.plot(Alpha,CL_alpha,'r',label='In_house')
plt.plot(vsp_Alpha,vsp_CL,'b--',label='vspaero')
plt.legend(loc=0)
plt.grid(True)
plt.savefig('figures/CL_alpha_rectangular_wing_20_40.png',dpi=300)
plt.figure()
plt.plot(Alpha,CD_alpha,'r',label='In_house')
plt.plot(vsp_Alpha,vsp_CDi,'b--',label='vspaero')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('figures/CD_alpha_rectangular_wing_20_40.png',dpi=300)
plt.figure()
plt.plot(CD_alpha,CL_alpha,'r',label='In_house')
plt.plot(vsp_CDi,vsp_CL,'b--',label='vspaero')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('figures/CD_CL_rectangular_wing_20_40.png',dpi=300)




