from VLM_light.VLM import VlmStudyOptimized
from extract_cli_distribution import parse_lod_file
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

# test_case = "rectangular_wing_20_40"
# test_case = "rectangular_wing_20_40_sweep_15"
# test_case = "rectangular_wing_20_40_sweep_15_twist_15"
# test_case = "rectangular_wing_20_40_HT"
test_case = "asg_29"

mesh_file = 'meshes/' + test_case + '.msh'
Alpha = jnp.linspace(0,10,11)
v_inf = 100.0
rho = 0.0023770
if test_case == "asg_29" :
    S_ref = 5.25
else :    
    S_ref = 10.0
CL_alpha = []
CD_alpha = []
CL_distribution = []

# creation of the study 
my_study = VlmStudyOptimized(mesh_file,Alpha[0],v_inf = v_inf,rho = rho,symmetry=True,x_wake = 1e6)
for alpha in Alpha :
    print("alpha=", alpha)
    surfaces = my_study.compute_topology()
    nodes_coord = my_study.nodes[:,1:]
    segments, segments_ids, control_points, normals, control_point_quart,ring_pts = my_study.compute_geometry(nodes_coord, alpha)
    gamma = my_study.compute_circulation_parametrized(segments,segments_ids, control_points,normals,alpha,my_study.v_inf)
    CL,CD,forces_panel,delta_L,delta_D = my_study.compute_CL_CD_forces(gamma,alpha,S_ref,segments,control_point_quart,segments_ids, ring_pts)
    y_span, cl_local, cl_distribution = my_study.compute_cl_distribution_span(delta_L, ring_pts)
    CL_alpha.append(CL)
    CD_alpha.append(CD)
    CL_distribution.append(cl_distribution)



#vspaero data 
vsp_data = np.loadtxt('ref_results/' + test_case + '.polar',skiprows=3)
vsp_Alpha = vsp_data[:,2]
vsp_CL = vsp_data[:,5]
vsp_CDi = vsp_data[:,8]

print("max relative squared error CL=", jnp.max(((jnp.array(CL_alpha)-vsp_CL)**2)/(vsp_CL+1e-12)**2))
print("max relative squared error CD=", jnp.max(((jnp.array(CD_alpha)-vsp_CDi)**2)/(vsp_CDi+1e-12)**2))

plt.figure()
plt.plot(Alpha,CL_alpha,'r',label='In_house')
plt.plot(vsp_Alpha,vsp_CL,'b--',label='vspaero')
plt.legend(loc=0)
plt.grid(True)
plt.savefig('figures/CL_alpha_' + test_case + '.png',dpi=300)
plt.figure()
plt.plot(Alpha,CD_alpha,'r',label='In_house')
plt.plot(vsp_Alpha,vsp_CDi,'b--',label='vspaero')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('figures/CD_alpha_' + test_case + '.png',dpi=300)
plt.figure()
plt.plot(CD_alpha,CL_alpha,'r',label='In_house')
plt.plot(vsp_CDi,vsp_CL,'b--',label='vspaero')
plt.grid(True)
plt.legend(loc=0)
plt.savefig('figures/CD_CL_' + test_case + '.png',dpi=300)

#load distribution comparison 
vsp_load = parse_lod_file('ref_results/' + test_case + '.lod') 
#plotting the load distribution for the different angles of attack
n_surfaces = 1
plt.figure()
k = 0
colors = plt.get_cmap('jet')
for alpha in Alpha :
    #we have to divide by 2 for the symmetry   
    if test_case == "asg_29" :
        plt.plot(CL_distribution[k]['y_column'],CL_distribution[k]['cl_integrated']/(2),color=colors(k/len(Alpha)),label='In_house alpha='+str(alpha))
        plt.plot(vsp_load[k][1]['yavg'],vsp_load[k][1]['cli'],color='k',linestyle='--',label='vspaero alpha='+str(alpha))
    else : 
        for s in range(n_surfaces):
            plt.plot(CL_distribution[k]["surfaces"][s+1]['y_column'],CL_distribution[k]["surfaces"][s+1]['cl_integrated']/(2),color=colors(k/len(Alpha)),label='In_house alpha='+str(alpha))
            plt.plot(vsp_load[k][s+1]['yavg'],vsp_load[k][s+1]['cli'],color='k',linestyle='--',label='vspaero alpha='+str(alpha))    
    k += 1
plt.legend(loc=0)
plt.grid(True)
plt.savefig('figures/CL_alpha_' + test_case + '_distribution.png',dpi=300)



