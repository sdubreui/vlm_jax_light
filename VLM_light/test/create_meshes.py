import numpy as np
import gmsh


# #aile rectangulaire 

gmsh.initialize()
# Next we add a new model named "wing" (if gmsh.model.add() is not called a new
# unnamed model will be created on the fly, if necessary):
gmsh.model.add("rectangular_wing")
#We create the rectangle using 4 points, 4 lines and 1 surface
#definition of the local mesh size around the points
# gmsh.model.geo.addPoint():
# - the first 3 arguments are the point coordinates (x, y, z)
# - the next (optional) argument is the target mesh size close to the point
# - the last (optional) argument is the point tag (a stricly positive integer
#   that uniquely identifies the point)
lc = 0.1
#chord and semi span
c = 1.0
b = 10.0
sweep = 0.0*np.pi/180.0

gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc, 1)
gmsh.model.geo.addPoint(c, 0.0, 0.0, lc, 2)
gmsh.model.geo.addPoint(np.tan(sweep)*b+c, b, 0, lc, 3)
gmsh.model.geo.addPoint(np.tan(sweep)*b, b, 0.0, lc, 4)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
gmsh.model.geo.addPlaneSurface([1], 1)
gmsh.model.geo.synchronize()
le1 = gmsh.model.addPhysicalGroup(1, [4], 1)
gmsh.model.setPhysicalName(1,le1,'leading_edge_1')
te1 = gmsh.model.addPhysicalGroup(1, [2], 2)
gmsh.model.setPhysicalName(1,te1,'trailing_edge_1')
wing1 = gmsh.model.addPhysicalGroup(2, [1], 3)
gmsh.model.setPhysicalName(2,wing1,'surface_1')
#mesh definition
nx = 21
ny1 = 41
# gmsh.model.mesh.setTransfiniteCurve(1,nx,meshType="Bump",coef=0.2)
# gmsh.model.mesh.setTransfiniteCurve(3,nx,meshType="Bump",coef=0.2)
gmsh.model.mesh.setTransfiniteCurve(1,nx)
gmsh.model.mesh.setTransfiniteCurve(3,nx)
gmsh.model.mesh.setTransfiniteCurve(2,ny1)
gmsh.model.mesh.setTransfiniteCurve(4,ny1)
gmsh.model.mesh.setTransfiniteSurface(1)
gmsh.model.mesh.setRecombine(2,1)
gmsh.model.mesh.generate(2)
gmsh.write("meshes/rectangular_wing_20_40.msh")



#aile rectangulaire 2 surfaces

# gmsh.initialize()
# # Next we add a new model named "wing" (if gmsh.model.add() is not called a new
# # unnamed model will be created on the fly, if necessary):
# gmsh.model.add("recatngular_wing")
# #We create the rectangle using 4 points, 4 lines and 1 surface
# #definition of the local mesh size around the points
# # gmsh.model.geo.addPoint():
# # - the first 3 arguments are the point coordinates (x, y, z)
# # - the next (optional) argument is the target mesh size close to the point
# # - the last (optional) argument is the point tag (a stricly positive integer
# #   that uniquely identifies the point)
# lc = 0.1
# #chord and semi span
# c = 1.0
# b = 10.0
# gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
# gmsh.model.geo.addPoint(c, 0, 0, lc, 2)
# gmsh.model.geo.addPoint(c, 0.5*b, 0, lc, 3)
# gmsh.model.geo.addPoint(0, 0.5*b, 0, lc, 4)
# gmsh.model.geo.addPoint(0, b, 0, lc, 5)
# gmsh.model.geo.addPoint(c, b, 0, lc, 6)
# gmsh.model.geo.addLine(1, 2, 1)
# gmsh.model.geo.addLine(2, 3, 2)
# gmsh.model.geo.addLine(3, 4, 3)
# gmsh.model.geo.addLine(4, 1, 4)
# gmsh.model.geo.addLine(3, 6, 5)
# gmsh.model.geo.addLine(6, 5, 6)
# gmsh.model.geo.addLine(5, 4, 7)

# #Creation of curveloop that closes the surfaces
# gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
# gmsh.model.geo.addCurveLoop([-3, 5, 6, 7], 2)
# #creation of the surfaces
# gmsh.model.geo.addPlaneSurface([1], 1)
# gmsh.model.geo.addPlaneSurface([2], 2)

# gmsh.model.geo.synchronize()

# #physical tags to define the leading edge, the trailing edge and the wing surface
# le1 = gmsh.model.addPhysicalGroup(1, [4], 1)
# gmsh.model.setPhysicalName(1,le1,'leading_edge_1')
# te1 = gmsh.model.addPhysicalGroup(1, [2], 2)
# gmsh.model.setPhysicalName(1,te1,'trailing_edge_1')
# wing1 = gmsh.model.addPhysicalGroup(2, [1], 3)
# gmsh.model.setPhysicalName(2,wing1,'surface_1')

# le2 = gmsh.model.addPhysicalGroup(1, [7], 4)
# gmsh.model.setPhysicalName(1,le2,'leading_edge_2')
# te2 = gmsh.model.addPhysicalGroup(1, [5], 5)
# gmsh.model.setPhysicalName(1,te2,'trailing_edge_2')
# wing2 = gmsh.model.addPhysicalGroup(2, [2], 6)
# gmsh.model.setPhysicalName(2,wing2,'surface_2')

# #definition of the mesh with the number of nodes per lines

# nx = 20
# ny1 = 20
# ny2 = 20

# gmsh.model.mesh.setTransfiniteCurve(1,nx,meshType="Bump",coef=0.2)
# gmsh.model.mesh.setTransfiniteCurve(3,nx,meshType="Bump",coef=0.2)
# gmsh.model.mesh.setTransfiniteCurve(6,nx,meshType="Bump",coef=0.2)
# # gmsh.model.mesh.setTransfiniteCurve(1,nx)
# # gmsh.model.mesh.setTransfiniteCurve(3,nx)
# # gmsh.model.mesh.setTransfiniteCurve(6,nx)
# gmsh.model.mesh.setTransfiniteCurve(2,ny1)
# gmsh.model.mesh.setTransfiniteCurve(4,ny1)
# gmsh.model.mesh.setTransfiniteCurve(5,ny2)
# gmsh.model.mesh.setTransfiniteCurve(7,ny2)
# gmsh.model.mesh.setTransfiniteSurface(1)
# gmsh.model.mesh.setTransfiniteSurface(2)

# gmsh.model.mesh.setRecombine(2,1)
# gmsh.model.mesh.setRecombine(2,2)
# # We can then generate a 2D mesh...
# gmsh.model.mesh.generate(2)
# # ... and save it to disk
# gmsh.write("rectangular_wing_20_40_2surfaces_bump.msh")

