#Simple mesh deformation for test against vsp aero geometry
import numpy as np 
import gmsh 


#initial mesh 
gmsh.initialize()
gmsh.open('meshes/rectangular_wing_20_40_sweep_15.msh')
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_tags = np.asarray(node_tags)          
coords = np.array(node_coords, dtype=float).reshape(node_tags.shape[0], 3)
gmsh.finalize()

#sort according to y coordinate
ind = coords[:,1].argsort()
sorted_coords = coords[ind]
sorted_tags = node_tags[ind]

#twist deformation 
twist_tip = -15.0*np.pi/180.0
#loop over the section 
n_sections = 41
for i in range(n_sections) :
    y_section = sorted_coords[i*21:(i+1)*21,1]
    x_section = sorted_coords[i*21:(i+1)*21,0]
    z_section = sorted_coords[i*21:(i+1)*21,2]
    #compute the twist coorsponding to this section (assuming linear twist distribution in this simple example)
    theta = y_section*twist_tip/sorted_coords[:,1].max()
    #apply deformation to the x coordinate (assuming rotation around 25% of te chord)
    x_origin = x_section.min() + 0.25*(x_section.max()-x_section.min()) 
    delta_x = x_section - x_origin
    x_section_deformed = x_origin + delta_x*np.cos(theta) - z_section*np.sin(theta)
    z_section_deformed = z_section*np.cos(theta) + delta_x*np.sin(theta)
    #update the coordinates
    sorted_coords[i*21:(i+1)*21,0] = x_section_deformed
    sorted_coords[i*21:(i+1)*21,2] = z_section_deformed
# Rebuild full coordinate array in original order
coords_deformed = np.zeros_like(coords)
coords_deformed[ind] = sorted_coords  # undo sorting

# Reopen gmsh
gmsh.initialize()
gmsh.open('meshes/rectangular_wing_20_40_sweep_15.msh')

# Update nodes one by one
for i, tag in enumerate(node_tags):
    gmsh.model.mesh.setNode(int(tag), coords_deformed[i].tolist(), [])

# Write deformed mesh
gmsh.write('meshes/rectangular_wing_20_40_sweep_15_twist_15.msh')

gmsh.finalize()
