"""
Optimized version of VLM in JAX.
Main optimizations:
1. Pre-calculation of normals (outside JIT)
2. Complete vectorization of calc_vorticity
3. Use of segment_sum instead of reduceat
4. Elimination of Python loops in JIT functions
5. Direct construction of arrays (no .at[].set())
"""

import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy.linalg as jlinalg
from jax.ops import segment_sum
from typing import Dict, List, Tuple
import gmsh
import time


class VlmStudyOptimized():
    
    def __init__(self, mesh_file: str, alpha: float = 0.0, v_inf: float = 0.84*295.4, 
                 rho: float = 0.38, x_wake: float = 1e3, symmetry: bool = True):       
        self.mesh_file = mesh_file
        self.v_inf = v_inf
        self.alpha = alpha
        self.rho = rho
        self.symmetry = symmetry
        self.x_wake = x_wake
        self.cosa = np.cos(self.alpha * np.pi / 180.)
        self.sina = np.sin(self.alpha * np.pi / 180.)
        self.tana = self.sina/self.cosa
        
        # Mesh reading (identical)
        self.surfaces = self.read_gmsh_mesh()
        
        # PRE-COMPUTE: Concatenate all normals into a single array
        # This avoids Python loops in JIT functions
        self._precompute_normals()
        
        # PRE-COMPUTE: Pre-compute geometry data for force calculations
        self._precompute_geometry_data()
        
        # PRE-COMPUTE: Indices for segment_sum
        self._precompute_segment_indices()
    
    def _precompute_normals(self):
        """Pre-computes a concatenated array of all normals."""
        all_normals = []
        for surface in self.surfaces:
            all_normals.append(surface['normals'])
        self.all_normals = jnp.array(np.vstack(all_normals))
        print(f"Pre-computed normals shape: {self.all_normals.shape}")
    
    def _precompute_geometry_data(self):
        """Pre-compute all geometry data needed for force calculations."""
        # Concatenate ring_points and panel_pairs from all surfaces
        all_ring_points = []
        all_leading_edge_bool = []
        all_trailing_edge_bool = []
        all_areas = []
        surface_ids_list = []  # Track which surface each panel belongs to
        panel_offset = 0
        
        # First pass: collect all data and build panel_pair mapping
        panel_pair_indices_list = [-1] * sum(s['n_panels'] for s in self.surfaces)
        panel_pair_type_list = [0] * sum(s['n_panels'] for s in self.surfaces)
        
        for surface_idx, surface in enumerate(self.surfaces):
            n_panels = surface['n_panels']
            all_ring_points.append(surface['ring_points'])
            all_areas.append(surface['areas'].flatten())
            
            # Track surface ID for each panel (1-indexed for consistency with VortexSheet)
            surface_ids_list.extend([surface_idx + 1] * n_panels)
            
            # Convert leading_edge (list of indices) to boolean array
            leading_edge_bool = np.zeros(n_panels, dtype=bool)
            if isinstance(surface['leading_edge'], list):
                leading_edge_bool[surface['leading_edge']] = True
            else:
                # already a boolean array
                leading_edge_bool = surface['leading_edge']
            all_leading_edge_bool.append(leading_edge_bool)
            
            # Convert trailing_edge (list of indices) to boolean array
            trailing_edge_bool = np.zeros(n_panels, dtype=bool)
            if isinstance(surface['trailing_edge'], list):
                trailing_edge_bool[surface['trailing_edge']] = True
            else:
                # already a boolean array
                trailing_edge_bool = surface['trailing_edge']
            all_trailing_edge_bool.append(trailing_edge_bool)
            
            # Build panel pair mapping
            if len(surface['panel_pairs']) > 0:
                for pair in surface['panel_pairs']:
                    idx1 = pair[0] + panel_offset
                    idx2 = pair[1] + panel_offset
                    panel_pair_indices_list[idx2] = idx1
                    panel_pair_type_list[idx2] = 1
            
            panel_offset += n_panels
        
        # Convert to numpy arrays (will convert to JAX on-demand)
        self.all_ring_points_np = np.vstack(all_ring_points)  # Shape: (n_panels, 4, 3)
        self.all_leading_edge_np = np.concatenate(all_leading_edge_bool)  # Shape: (n_panels,) boolean
        self.all_trailing_edge_np = np.concatenate(all_trailing_edge_bool)  # Shape: (n_panels,) boolean
        self.all_areas_np = np.concatenate(all_areas)  # Shape: (n_panels,)
        self.surface_ids_np = np.array(surface_ids_list, dtype=np.int32)  # Shape: (n_panels,) surface ID for each panel
        self.S_ref = float(np.sum(self.all_areas_np))
        
        # Store panel pair mapping as numpy for now
        self.panel_pair_indices_np = np.array(panel_pair_indices_list, dtype=np.int32)
        self.panel_pair_type_np = np.array(panel_pair_type_list, dtype=np.int32)
        self.ind_panel_pairs = jnp.where(self.panel_pair_indices_np >= 0)

        print(f"Pre-computed ring_points shape: {self.all_ring_points_np.shape}")
        print(f"Pre-computed leading_edge shape: {self.all_leading_edge_np.shape}")
        print(f"Pre-computed areas: total S_ref = {self.S_ref}")
        print(f"Pre-computed surface IDs shape: {self.surface_ids_np.shape}")
    
    def _precompute_segment_indices(self):
        """Pre-computes indices for segment_sum."""
        # Calculate segment indices for each vortex
        n_panels = len(self.global_sum_index)
        segment_ids = []
        
        for panel_idx, n_segments in enumerate(self.global_sum_index):
            segment_ids.extend([panel_idx] * n_segments)
        
        self.segment_ids = jnp.array(segment_ids, dtype=jnp.int32)
        print(f"Segment IDs shape: {self.segment_ids.shape}")
        print(f"Total segments: {len(self.segment_ids)}, Total panels: {n_panels}")

    def read_gmsh_mesh(self) -> List[Dict]:
        """Identical to the original version."""
        gmsh.initialize()
        gmsh.open(self.mesh_file) 
        nodes_tag, nodes_coord, par_coord = gmsh.model.mesh.getNodes() 
        n_nodes = int(len(nodes_tag))
        nodes_coord = nodes_coord.reshape((n_nodes, 3))
        nodes = np.zeros((n_nodes, 4))
        nodes[:, 0] = nodes_tag
        nodes[:, 1:] = nodes_coord
        self.nodes = nodes
        
        entities = np.array(gmsh.model.getEntities())
        ind_surf = entities[:, 0] == 2
        surf_entities = entities[ind_surf]
        n_surfaces = len(surf_entities)
        
        ind_edges = entities[:, 0] == 1
        edge_entities = entities[ind_edges]
        surfaces = []
        
        self.global_segment_bounds = []
        self.global_control_points = []
        self.global_control_points_quart = []
        self.global_sum_index = []
        
        for i in range(n_surfaces):
            pg = gmsh.model.getPhysicalGroupsForEntity(surf_entities[i][0], surf_entities[i][1])
            name = gmsh.model.getPhysicalName(2, pg[0])
            surf_index = name.split('_')[1]

            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, surf_entities[i][1])
            n_elements_surf = len(elemTags[0])
            Q4_elements = np.zeros((n_elements_surf, 5))
            Q4_elements[:, 0] = elemTags[0]
            Q4_elements[:, 1:] = elemNodeTags[0].reshape((n_elements_surf, 4))

            surfaces.append({'name': i+1, 'mesh': Q4_elements, 'n_panels': n_elements_surf})
            normals = np.zeros((surfaces[i]['n_panels'], 3))
            control_points = np.zeros((surfaces[i]['n_panels'], 3))
            control_points_quart = np.zeros((surfaces[i]['n_panels'], 3))
            ring_points = np.zeros((surfaces[i]['n_panels'], 4, 3))
            points = np.zeros((surfaces[i]['n_panels'], 4, 3))
            areas = np.zeros((surfaces[i]['n_panels'], 1))
            
            for ee in edge_entities:
                pg = gmsh.model.getPhysicalGroupsForEntity(ee[0], ee[1])
                if pg.shape[0] > 0:
                    if gmsh.model.getPhysicalName(1, pg[0]) == 'leading_edge_'+surf_index:
                        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, ee[1])
                        nodes_le = elemNodeTags[0].reshape((len(elemTags[0]), 2))
                    elif gmsh.model.getPhysicalName(1, pg[0]) == 'trailing_edge_'+surf_index:    
                        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, ee[1])
                        nodes_te = elemNodeTags[0].reshape((len(elemTags[0]), 2))
            
            leading_edge = []
            trailing_edge = []
            
            for j in range(surfaces[i]['n_panels']):
                panel = surfaces[i]['mesh'][j, :]
                panel_nodes = np.array([
                    nodes[np.argwhere(nodes[:, 0] == panel[1])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[2])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[3])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[4])[0, 0], 1:]
                ])
                
                if not np.array_equal(panel_nodes[:, 1], np.ones((4,)) * panel_nodes[0, 1]):
                    partition = np.argpartition(panel_nodes[:, 1], 2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2], 0]), :]
                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2], 0]), :]
                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:], 0]), :]
                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:], 0]), :]
                    normals[j, :] = np.cross(A-C, D-B)
                    norm = np.linalg.norm(normals[j, :])
                    normals[j, :] = normals[j, :] / norm
                else:
                    partition = np.argpartition(panel_nodes[:, 2], 2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2], 0]), :]
                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2], 0]), :]
                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:], 0]), :]
                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:], 0]), :]
                    normals[j, :] = np.cross(A-C, B-D)
                    norm = np.linalg.norm(normals[j, :])
                    normals[j, :] = normals[j, :] / norm
                
                E = A + 3.0/4.0 * (D - A)
                F = B + 3.0/4.0 * (C - B)
                control_points[j, :] = E + 0.5 * (F - E)
                ring_points[j, 0, :] = A + 1.0/4.0 * (D - A)
                ring_points[j, 1, :] = B + 1.0/4.0 * (C - B)
                ring_points[j, 2, :] = B + 5.0/4.0 * (C - B)
                ring_points[j, 3, :] = A + 5.0/4.0 * (D - A)
                control_points_quart[j, :] = ring_points[j, 0, :] + 0.5 * (ring_points[j, 1, :] - ring_points[j, 0, :])
                areas[j, 0] = 0.5 * np.linalg.norm(np.cross(C-A, D-B))
                points[j, 0, :] = A
                points[j, 1, :] = B
                points[j, 2, :] = C
                points[j, 3, :] = D
                
                if np.isin(nodes_le, panel[1:]).any():
                    leading_edge.append(j)
                if np.isin(nodes_te, panel[1:]).any():
                    trailing_edge.append(j)
                
                # Vortex segments
                self.global_segment_bounds.append([ring_points[j, 0, :], ring_points[j, 1, :]])
                self.global_segment_bounds.append([ring_points[j, 1, :], ring_points[j, 2, :]])
                self.global_segment_bounds.append([ring_points[j, 2, :], ring_points[j, 3, :]])
                self.global_segment_bounds.append([ring_points[j, 3, :], ring_points[j, 0, :]])

                if self.symmetry:
                    ring_A_sym = ring_points[j, 0, :].copy()
                    ring_A_sym[1] = -ring_A_sym[1]
                    ring_B_sym = ring_points[j, 1, :].copy()
                    ring_B_sym[1] = -ring_B_sym[1]
                    ring_C_sym = ring_points[j, 2, :].copy()
                    ring_C_sym[1] = -ring_C_sym[1]
                    ring_D_sym = ring_points[j, 3, :].copy()
                    ring_D_sym[1] = -ring_D_sym[1]
                    
                    self.global_segment_bounds.append([ring_B_sym, ring_A_sym])
                    self.global_segment_bounds.append([ring_A_sym, ring_D_sym])
                    self.global_segment_bounds.append([ring_D_sym, ring_C_sym])
                    self.global_segment_bounds.append([ring_C_sym, ring_B_sym])
                    
                    if np.isin(nodes_te, panel[1:]).any():
                        C = ring_points[j, 2, :].copy()
                        C[0] = self.x_wake
                        C[2] = C[0] * self.tana
                        D = ring_points[j, 3, :].copy()
                        D[0] = self.x_wake
                        D[2] = D[0] * self.tana
                        
                        self.global_segment_bounds.append([ring_points[j, 3, :], ring_points[j, 2, :]])
                        self.global_segment_bounds.append([ring_points[j, 2, :], C])
                        self.global_segment_bounds.append([C, D])
                        self.global_segment_bounds.append([D, ring_points[j, 3, :]])
                        
                        ring_B_sym = ring_points[j, 2, :].copy()
                        ring_A_sym = ring_points[j, 3, :].copy()
                        ring_C_sym = C.copy()
                        ring_D_sym = D.copy()
                        ring_A_sym[1] = -ring_A_sym[1]
                        ring_B_sym[1] = -ring_B_sym[1]
                        ring_C_sym[1] = -ring_C_sym[1]
                        ring_D_sym[1] = -ring_D_sym[1]
                        
                        self.global_segment_bounds.append([ring_B_sym, ring_A_sym])
                        self.global_segment_bounds.append([ring_A_sym, ring_D_sym])
                        self.global_segment_bounds.append([ring_D_sym, ring_C_sym])
                        self.global_segment_bounds.append([ring_C_sym, ring_B_sym])
                        
                        self.global_sum_index.append(16)
                    else:
                        self.global_sum_index.append(8)
                else:
                    if np.isin(nodes_te, panel[1:]).any():
                        C = ring_points[j, 2, :].copy()
                        C[0] = self.x_wake
                        C[2] = C[0] * self.tana
                        D = ring_points[j, 3, :].copy()
                        D[0] = self.x_wake
                        D[2] = D[0] * self.tana
                        
                        self.global_segment_bounds.append([ring_points[j, 3, :], ring_points[j, 2, :]])
                        self.global_segment_bounds.append([ring_points[j, 2, :], C])
                        self.global_segment_bounds.append([C, D])
                        self.global_segment_bounds.append([D, ring_points[j, 3, :]])
                        self.global_sum_index.append(8)
                    else:
                        self.global_sum_index.append(4)
                
                self.global_control_points.append(control_points[j, :])
                self.global_control_points_quart.append(control_points_quart[j, :])
            
            surfaces[i]['normals'] = normals
            surfaces[i]['control_points'] = control_points
            surfaces[i]['control_points_quart'] = control_points_quart
            surfaces[i]['ring_points'] = ring_points
            surfaces[i]['areas'] = areas
            surfaces[i]['points'] = points
            surfaces[i]['leading_edge'] = leading_edge
            surfaces[i]['trailing_edge'] = trailing_edge
            
            k = 0
            panel_candidate = np.arange(0, surfaces[i]['n_panels'], 1)
            panel_pairs = []
            for panel in surfaces[i]['points']:
                if k not in surfaces[i]['leading_edge']:
                    A = panel[0, :]
                    B = panel[1, :]
                    ind_1 = np.isclose(surfaces[i]['points'][:, 3, :], A).all(axis=1)
                    ind_2 = np.isclose(surfaces[i]['points'][:, 2, :], B).all(axis=1)
                    pair = panel_candidate[ind_1 & ind_2]
                    panel_pairs.append([k, pair[0]])
                k = k + 1
            
            surfaces[i]['panel_pairs'] = np.array(panel_pairs)
            
            for pairs in surfaces[i]['panel_pairs']:
                surfaces[i]['ring_points'][pairs[1]][2] = surfaces[i]['ring_points'][pairs[0]][1]
                surfaces[i]['ring_points'][pairs[1]][3] = surfaces[i]['ring_points'][pairs[0]][0]
                
        
        self.global_control_points = np.array(self.global_control_points)
        self.global_control_points_quart = np.array(self.global_control_points_quart)
        self.global_segment_bounds = np.array(self.global_segment_bounds)
        self.global_sum_index = np.array(self.global_sum_index)
        
        gmsh.finalize()
        return surfaces

    def compute_topology(self) -> List[Dict]:
        """Computes the topology from the mesh"""
        gmsh.initialize()
        gmsh.open(self.mesh_file) 
        #We still need the nodes to find the panel pairs
        nodes_tag, nodes_coord, par_coord = gmsh.model.mesh.getNodes() 
        n_nodes = int(len(nodes_tag))
        nodes_coord = nodes_coord.reshape((n_nodes, 3))
        nodes = np.zeros((n_nodes, 4))
        nodes[:, 0] = nodes_tag
        nodes[:, 1:] = nodes_coord
        self.nodes = nodes

        entities = np.array(gmsh.model.getEntities())
        ind_surf = entities[:, 0] == 2
        surf_entities = entities[ind_surf]
        n_surfaces = len(surf_entities)
        
        ind_edges = entities[:, 0] == 1
        edge_entities = entities[ind_edges]
        surfaces = []
        
        
        for i in range(n_surfaces):
            pg = gmsh.model.getPhysicalGroupsForEntity(surf_entities[i][0], surf_entities[i][1])
            name = gmsh.model.getPhysicalName(2, pg[0])
            surf_index = name.split('_')[1]

            elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, surf_entities[i][1])
            n_elements_surf = len(elemTags[0])
            Q4_elements = np.zeros((n_elements_surf, 5))
            Q4_elements[:, 0] = elemTags[0]
            Q4_elements[:, 1:] = elemNodeTags[0].reshape((n_elements_surf, 4))

            surfaces.append({'name': i+1, 'mesh': Q4_elements, 'n_panels': n_elements_surf})
            points = np.zeros((n_elements_surf, 4, 3)) 

            for ee in edge_entities:
                pg = gmsh.model.getPhysicalGroupsForEntity(ee[0], ee[1])
                if pg.shape[0] > 0:
                    if gmsh.model.getPhysicalName(1, pg[0]) == 'leading_edge_'+surf_index:
                        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, ee[1])
                        nodes_le = elemNodeTags[0].reshape((len(elemTags[0]), 2))
                    elif gmsh.model.getPhysicalName(1, pg[0]) == 'trailing_edge_'+surf_index:    
                        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(1, ee[1])
                        nodes_te = elemNodeTags[0].reshape((len(elemTags[0]), 2))
            
            leading_edge_flag = np.array([False]*surfaces[i]['n_panels'])
            trailing_edge_flag = np.array([False]*surfaces[i]['n_panels'])
            
            for j in range(surfaces[i]['n_panels']):
                panel = surfaces[i]['mesh'][j, :]
                panel_nodes = np.array([
                    nodes[np.argwhere(nodes[:, 0] == panel[1])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[2])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[3])[0, 0], 1:],
                    nodes[np.argwhere(nodes[:, 0] == panel[4])[0, 0], 1:]
                ])
                panel_nodes_id = np.array([
                    nodes[np.argwhere(nodes[:, 0] == panel[1])[0, 0], 0],
                    nodes[np.argwhere(nodes[:, 0] == panel[2])[0, 0], 0],
                    nodes[np.argwhere(nodes[:, 0] == panel[3])[0, 0], 0],
                    nodes[np.argwhere(nodes[:, 0] == panel[4])[0, 0], 0]
                ])
                if not np.array_equal(panel_nodes[:, 1], np.ones((4,)) * panel_nodes[0, 1]):
                    partition = np.argpartition(panel_nodes[:, 1], 2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2], 0]), :]
                    ind_A = panel_nodes_id[partition[:2]][np.argmin(panel_nodes[partition[:2], 0])]

                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2], 0]), :]
                    ind_D = panel_nodes_id[partition[:2]][np.argmax(panel_nodes[partition[:2], 0])]

                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:], 0]), :]
                    ind_B = panel_nodes_id[partition[2:]][np.argmin(panel_nodes[partition[2:], 0])]
                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:], 0]), :]
                    ind_C = panel_nodes_id[partition[2:]][np.argmax(panel_nodes[partition[2:], 0])]

                    #rearanged the mesh according to A,B,C,D
                    surfaces[i]['mesh'][j,1] = ind_A
                    surfaces[i]['mesh'][j,2] = ind_B
                    surfaces[i]['mesh'][j,3] = ind_C
                    surfaces[i]['mesh'][j,4] = ind_D
                else:
                    partition = np.argpartition(panel_nodes[:, 2], 2)
                    A = panel_nodes[partition[:2]][np.argmin(panel_nodes[partition[:2], 0]), :]
                    ind_A = panel_nodes_id[partition[:2]][np.argmin(panel_nodes[partition[:2], 0])]

                    D = panel_nodes[partition[:2]][np.argmax(panel_nodes[partition[:2], 0]), :]
                    ind_D = panel_nodes_id[partition[:2]][np.argmax(panel_nodes[partition[:2], 0])]

                    B = panel_nodes[partition[2:]][np.argmin(panel_nodes[partition[2:], 0]), :]
                    ind_B = panel_nodes_id[partition[2:]][np.argmin(panel_nodes[partition[2:], 0])]

                    C = panel_nodes[partition[2:]][np.argmax(panel_nodes[partition[2:], 0]), :]
                    ind_C = panel_nodes_id[partition[2:]][np.argmax(panel_nodes[partition[2:], 0])]

                    #rearanged the mesh according to A,B,C,D
                    surfaces[i]['mesh'][j,1] = ind_A
                    surfaces[i]['mesh'][j,2] = ind_B
                    surfaces[i]['mesh'][j,3] = ind_C
                    surfaces[i]['mesh'][j,4] = ind_D
                points[j, 0, :] = A
                points[j, 1, :] = B
                points[j, 2, :] = C
                points[j, 3, :] = D

                
                if np.isin(nodes_le, panel[1:]).any():   
                    leading_edge_flag[j] = True
                if np.isin(nodes_te, panel[1:]).any():
                    trailing_edge_flag[j] = True       

            surfaces[i]['leading_edge'] = leading_edge_flag
            surfaces[i]['trailing_edge'] = trailing_edge_flag
            surfaces[i]['points'] = points
            surfaces[i]['n_trailing_edge'] = np.sum(trailing_edge_flag)

            k = 0
            panel_candidate = np.arange(0, surfaces[i]['n_panels'], 1)
            panel_pairs = []
            for panel in surfaces[i]['points']:
                if surfaces[i]['leading_edge'][k] == False:
                    A = panel[0, :]
                    B = panel[1, :]
                    ind_1 = np.isclose(surfaces[i]['points'][:, 3, :], A).all(axis=1)
                    ind_2 = np.isclose(surfaces[i]['points'][:, 2, :], B).all(axis=1)
                    pair = panel_candidate[ind_1 & ind_2]
                    panel_pairs.append([k, pair[0]])
                k = k + 1
            
            surfaces[i]['panel_pairs'] = np.array(panel_pairs)
            
            # for pairs in surfaces[i]['panel_pairs']:
            #     surfaces[i]['ring_points'][pairs[1]][2] = surfaces[i]['ring_points'][pairs[0]][1]
            #     surfaces[i]['ring_points'][pairs[1]][3] = surfaces[i]['ring_points'][pairs[0]][0]

            surfaces[i]['mesh'] = jnp.array(surfaces[i]['mesh'],dtype=int)

        gmsh.finalize()
        return surfaces


    def compute_control_points(self, panel_nodes: jnp.ndarray) -> jnp.ndarray:
        """Computes the control points for a set of panels given by
        their node coordinates.
        
        Inputs:
            panel_nodes: array [n_panels, 4, 3], coordinates of the panel nodes
        
        Outputs:
            control_points: array [n_panels, 3], coordinates of the control points
        """
        n_panels = panel_nodes.shape[0]
        A = panel_nodes[:, 0, :]
        B = panel_nodes[:, 1, :]
        C = panel_nodes[:, 2, :]
        D = panel_nodes[:, 3, :]
        E = A + 3.0/4.0 * (D - A)
        F = B + 3.0/4.0 * (C - B)
        control_points = E + 0.5 * (F - E)
        return control_points

    def compute_ring_points(self, panel_nodes: jnp.ndarray) -> jnp.ndarray:
        """Computes the ring points for a set of panels given by
        their node coordinates.
        
        Inputs:
            panel_nodes: array [n_panels, 4, 3], coordinates of the panel nodes
        
        Outputs:
            ring_points: array [n_panels, 4, 3], coordinates of the ring points
        """
        n_panels = panel_nodes.shape[0]
        ring_points = jnp.zeros((n_panels, 4, 3))
        A = panel_nodes[:, 0, :]
        B = panel_nodes[:, 1, :]
        C = panel_nodes[:, 2, :]
        D = panel_nodes[:, 3, :]
        ring_points = ring_points.at[:, 0, :].set(A + 1.0/4.0 * (D - A))
        ring_points = ring_points.at[:, 1, :].set(B + 1.0/4.0 * (C - B))
        ring_points = ring_points.at[:, 2, :].set(B + 5.0/4.0 * (C - B))
        ring_points = ring_points.at[:, 3, :].set(A + 5.0/4.0 * (D - A))
        return ring_points

    def compute_control_points_quart(self, ring_points: jnp.ndarray) -> jnp.ndarray:
        """Computes the quarter-chord control points for a set of panels given by
        their ring point coordinates.
        
        Inputs:
            ring_points: array [n_panels, 4, 3], coordinates of the ring points of the panels
        
        Outputs:
            control_points_quart: array [n_panels, 3], coordinates of the quarter-chord control points
        """

        control_points_quart = ring_points[:, 0, :] + 0.5 * (ring_points[:, 1, :] - ring_points[:, 0, :])

        return control_points_quart

    def compute_segment_bounds(self, ring_points: jnp.ndarray,trailing_flag: List[bool]) -> Tuple[jnp.ndarray,jnp.ndarray]:
        """Computes the bounds of the vortex segments for a set of panels given by
        their ring point coordinates.
        
        Inputs:
            ring_points: array [n_panels, 4, 3], coordinates of the ring points of the panels
            
        
        Outputs:
            segment_bounds: array [n_segments,2,3] coordinates of the bounds of the vortex segments
        """

        n_panels = ring_points.shape[0]
        n_trailing = np.sum(np.array(trailing_flag,dtype=int))    
        
        segment_bounds = jnp.zeros((4*(n_panels+n_trailing)*2, 2, 3)) #*2 assuming symmetry
        # 4 segments of each pannel
        #A-B
        segment_bounds = segment_bounds.at[:4*n_panels:4, 0, :].set(ring_points[:, 0, :])
        segment_bounds = segment_bounds.at[:4*n_panels:4, 1, :].set(ring_points[:, 1, :])
        #B-C
        segment_bounds = segment_bounds.at[1:4*n_panels+1:4, 0, :].set(ring_points[:, 1, :])
        segment_bounds = segment_bounds.at[1:4*n_panels+1:4, 1, :].set(ring_points[:, 2, :])
        #C-D
        segment_bounds = segment_bounds.at[2:4*n_panels+2:4, 0, :].set(ring_points[:, 2, :])
        segment_bounds = segment_bounds.at[2:4*n_panels+2:4, 1, :].set(ring_points[:, 3, :]) 
        #D-A
        segment_bounds = segment_bounds.at[3:4*n_panels+3:4, 0, :].set(ring_points[:, 3, :])
        segment_bounds = segment_bounds.at[3:4*n_panels+3:4, 1, :].set(ring_points[:, 0, :])
        # trailing edge segments
        C = ring_points[trailing_flag, 2, :].copy()
        C = C.at[:,0].set(self.x_wake)
        C = C.at[:,2].set(C[:,0] * self.tana)
        D = ring_points[trailing_flag, 3, :].copy()
        D = D.at[:,0].set(self.x_wake)
        D = D.at[:,2].set(D[:,0] * self.tana)
        #A-B
        segment_bounds = segment_bounds.at[4*n_panels:4*(n_panels+n_trailing):4,0,:].set(ring_points[trailing_flag,3,:])
        segment_bounds = segment_bounds.at[4*n_panels:4*(n_panels+n_trailing):4,1,:].set(ring_points[trailing_flag,2,:])
        #B-C
        segment_bounds = segment_bounds.at[4*n_panels+1:4*(n_panels+n_trailing)+1:4,0,:].set(ring_points[trailing_flag,2,:])
        segment_bounds = segment_bounds.at[4*n_panels+1:4*(n_panels+n_trailing)+1:4,1,:].set(C)
        #C-D
        segment_bounds = segment_bounds.at[4*n_panels+2:4*(n_panels+n_trailing)+2:4,0,:].set(C)
        segment_bounds = segment_bounds.at[4*n_panels+2:4*(n_panels+n_trailing)+2:4,1,:].set(D)
        #D-A
        segment_bounds = segment_bounds.at[4*n_panels+3:4*(n_panels+n_trailing)+3:4,0,:].set(D)
        segment_bounds = segment_bounds.at[4*n_panels+3:4*(n_panels+n_trailing)+3:4,1,:].set(ring_points[trailing_flag, 3, :])

        #Symmetry segments - conditionally applied based on self.symmetry flag
        symmetry_flag = jnp.asarray(self.symmetry, dtype=jnp.float32)
        
        ring_A_sym = ring_points[:, 0, :].copy()
        ring_A_sym = ring_A_sym.at[:,1].set(-ring_A_sym[:,1])
        ring_B_sym = ring_points[:, 1, :].copy()
        ring_B_sym = ring_B_sym.at[:,1].set(-ring_B_sym[:,1])
        ring_C_sym = ring_points[:, 2, :].copy()
        ring_C_sym = ring_C_sym.at[:,1].set(-ring_C_sym[:,1])
        ring_D_sym = ring_points[:, 3, :].copy()
        ring_D_sym = ring_D_sym.at[:,1].set(-ring_D_sym[:,1])

        # Apply symmetry flag to bound segments (scale by 0 or 1)
        # A-B
        seg_ab_sym = jnp.where(symmetry_flag > 0.5, ring_B_sym, jnp.zeros_like(ring_B_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing):4*(2*n_panels+n_trailing):4,0,:].set(seg_ab_sym)
        seg_ab_sym_b = jnp.where(symmetry_flag > 0.5, ring_A_sym, jnp.zeros_like(ring_A_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing):4*(2*n_panels+n_trailing):4,1,:].set(seg_ab_sym_b)
        # B-C
        seg_bc_sym = jnp.where(symmetry_flag > 0.5, ring_A_sym, jnp.zeros_like(ring_A_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+1:4*(2*n_panels+n_trailing)+1:4,0,:].set(seg_bc_sym)
        seg_bc_sym_b = jnp.where(symmetry_flag > 0.5, ring_D_sym, jnp.zeros_like(ring_D_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+1:4*(2*n_panels+n_trailing)+1:4,1,:].set(seg_bc_sym_b)
        # C-D
        seg_cd_sym = jnp.where(symmetry_flag > 0.5, ring_D_sym, jnp.zeros_like(ring_D_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+2:4*(2*n_panels+n_trailing)+2:4,0,:].set(seg_cd_sym)
        seg_cd_sym_b = jnp.where(symmetry_flag > 0.5, ring_C_sym, jnp.zeros_like(ring_C_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+2:4*(2*n_panels+n_trailing)+2:4,1,:].set(seg_cd_sym_b)
        # D-A
        seg_da_sym = jnp.where(symmetry_flag > 0.5, ring_C_sym, jnp.zeros_like(ring_C_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+3:4*(2*n_panels+n_trailing)+3:4,0,:].set(seg_da_sym)
        seg_da_sym_b = jnp.where(symmetry_flag > 0.5, ring_B_sym, jnp.zeros_like(ring_B_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+3:4*(2*n_panels+n_trailing)+3:4,1,:].set(seg_da_sym_b)

        #Symmetry trailing edge segments - conditionally applied

        ring_B_sym_te = ring_points[trailing_flag, 2, :].copy()
        ring_A_sym_te = ring_points[trailing_flag, 3, :].copy()
        ring_C_sym_te = C.copy()
        ring_D_sym_te = D.copy()
        ring_A_sym_te = ring_A_sym_te.at[:,1].set(-ring_A_sym_te[:,1])
        ring_B_sym_te = ring_B_sym_te.at[:,1].set(-ring_B_sym_te[:,1])
        ring_C_sym_te = ring_C_sym_te.at[:,1].set(-ring_C_sym_te[:,1])
        ring_D_sym_te = ring_D_sym_te.at[:,1].set(-ring_D_sym_te[:,1])

        # A-B
        seg_ab_sym_te = jnp.where(symmetry_flag > 0.5, ring_B_sym_te, jnp.zeros_like(ring_B_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing):8*(n_panels+n_trailing):4,0,:].set(seg_ab_sym_te)
        seg_ab_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_A_sym_te, jnp.zeros_like(ring_A_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing):8*(n_panels+n_trailing):4,1,:].set(seg_ab_sym_te_b)
        # B-C
        seg_bc_sym_te = jnp.where(symmetry_flag > 0.5, ring_A_sym_te, jnp.zeros_like(ring_A_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+1:8*(n_panels+n_trailing)+1:4,0,:].set(seg_bc_sym_te)
        seg_bc_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_D_sym_te, jnp.zeros_like(ring_D_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+1:8*(n_panels+n_trailing)+1:4,1,:].set(seg_bc_sym_te_b)
        # C-D
        seg_cd_sym_te = jnp.where(symmetry_flag > 0.5, ring_D_sym_te, jnp.zeros_like(ring_D_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+2:8*(n_panels+n_trailing)+2:4,0,:].set(seg_cd_sym_te)
        seg_cd_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_C_sym_te, jnp.zeros_like(ring_C_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+2:8*(n_panels+n_trailing)+2:4,1,:].set(seg_cd_sym_te_b)
        # D-A
        seg_da_sym_te = jnp.where(symmetry_flag > 0.5, ring_C_sym_te, jnp.zeros_like(ring_C_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+3:8*(n_panels+n_trailing)+3:4,0,:].set(seg_da_sym_te)
        seg_da_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_B_sym_te, jnp.zeros_like(ring_B_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+3:8*(n_panels+n_trailing)+3:4,1,:].set(seg_da_sym_te_b)
        
        segments_ids_temp = jnp.repeat(jnp.arange(0, n_panels), 4)
        segments_ids = jnp.concatenate((segments_ids_temp,jnp.repeat(jnp.arange(0, n_panels)[trailing_flag], 4),segments_ids_temp,jnp.repeat(jnp.arange(0, n_panels)[trailing_flag], 4)))

        return segment_bounds, segments_ids
    
    def compute_areas_normals(self, panel_nodes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the areas and normals for a set of panels given by
        their node coordinates.
        
        Inputs:
            panel_nodes: array [n_panels, 4, 3], coordinates of the panel nodes
        
        Outputs:
            areas: array [n_panels, 1], areas of the panels
            normals: array [n_panels, 3], normals of the panels
        """
        n_panels = panel_nodes.shape[0]
        A = panel_nodes[:, 0, :]
        B = panel_nodes[:, 1, :]
        C = panel_nodes[:, 2, :]
        D = panel_nodes[:, 3, :]
        cross_prod = jnp.cross(A - C, D - B)
        areas = 0.5 * jnp.linalg.norm(cross_prod, axis=1, keepdims=True)
        normals = cross_prod / jnp.linalg.norm(cross_prod, axis=1, keepdims=True)
        return areas, normals   


    def compute_geometry(self, nodes_coords: jnp.ndarray, surfaces: List[Dict],alpha) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Unified geometry update for multiple surfaces.
        Ensures that panel indices (0 to N-1) are consistent across all geometry arrays.
        """
        all_control_points = []
        all_control_points_quart = []
        all_normals = []
        all_segment_bounds = []
        all_segment_ids = []
        all_ring_pts = []

        panel_offset = 0
        
        for surface in surfaces:
            # 1. Get panel nodes using surface mesh mapping
            # Subtract 1 because Gmsh tags are 1-based, Python is 0-based
            mesh_indices = jnp.array(surface['mesh'][:, 1:] - 1, dtype=int)
            panel_nodes = nodes_coords[mesh_indices]
            
            # 2. Basic Panel Geometry
            cp = self.compute_control_points(panel_nodes)
            areas, normals = self.compute_areas_normals(panel_nodes)
            ring_pts = self.compute_ring_points(panel_nodes)
            pairs = surface['panel_pairs']
            ring_pts=ring_pts.at[pairs[:,1],2,:].set(ring_pts[pairs[:,0]][:,1])
            ring_pts=ring_pts.at[pairs[:,1],3,:].set(ring_pts[pairs[:,0]][:,0])
            cp_quart = self.compute_control_points_quart(ring_pts)
            
            # 3. Vortex Segments with Global Offset
            # We pass the panel_offset to ensure IDs are [panel_offset : panel_offset + n_panels]
            seg_bounds, seg_ids = self._compute_segment_bounds_with_offset(
                ring_pts, surface['trailing_edge'], panel_offset,alpha
            )
            
            # 4. Collect results
            all_control_points.append(cp)
            all_control_points_quart.append(cp_quart)
            all_normals.append(normals)
            all_segment_bounds.append(seg_bounds)
            all_segment_ids.append(seg_ids)
            all_ring_pts.append(ring_pts)
            
            panel_offset += surface['n_panels']

        # Concatenate everything into final JAX arrays
        return (
            jnp.concatenate(all_segment_bounds, axis=0),
            jnp.concatenate(all_segment_ids, axis=0),
            jnp.concatenate(all_control_points, axis=0),
            jnp.concatenate(all_normals, axis=0),
            jnp.concatenate(all_control_points_quart, axis=0),
            jnp.concatenate(all_ring_pts,axis=0)
        )

    def _compute_segment_bounds_with_offset(self, ring_points, trailing_flag, offset, alpha):
        """Modified version of compute_segment_bounds that handles global indexing.
        Supports both symmetric and non-symmetric configurations using conditional masking.
        """
        n_panels = ring_points.shape[0]
        
        n_trailing = np.sum(np.array(trailing_flag,dtype=int))
        
        # Always allocate maximum size (with symmetry) for static JAX compilation
        # Non-symmetric entries will be zeroed out via masking
        segment_bounds = jnp.zeros((4*(n_panels+n_trailing)*2, 2, 3)) #*2 for symmetry allocation
        # 4 segments of each pannel
        #A-B
        segment_bounds = segment_bounds.at[:4*n_panels:4, 0, :].set(ring_points[:, 0, :])
        segment_bounds = segment_bounds.at[:4*n_panels:4, 1, :].set(ring_points[:, 1, :])
        #B-C
        segment_bounds = segment_bounds.at[1:4*n_panels+1:4, 0, :].set(ring_points[:, 1, :])
        segment_bounds = segment_bounds.at[1:4*n_panels+1:4, 1, :].set(ring_points[:, 2, :])
        #C-D
        segment_bounds = segment_bounds.at[2:4*n_panels+2:4, 0, :].set(ring_points[:, 2, :])
        segment_bounds = segment_bounds.at[2:4*n_panels+2:4, 1, :].set(ring_points[:, 3, :]) 
        #D-A
        segment_bounds = segment_bounds.at[3:4*n_panels+3:4, 0, :].set(ring_points[:, 3, :])
        segment_bounds = segment_bounds.at[3:4*n_panels+3:4, 1, :].set(ring_points[:, 0, :])
        # trailing edge segments
        #alpha 
        cosa = jnp.cos(alpha * np.pi / 180.)
        sina = jnp.sin(alpha * np.pi / 180.)
        tana = sina/cosa
        C = ring_points[trailing_flag, 2, :].copy()
        C = C.at[:,0].set(self.x_wake)
        C = C.at[:,2].set(C[:,0] * tana)
        D = ring_points[trailing_flag, 3, :].copy()
        D = D.at[:,0].set(self.x_wake)
        D = D.at[:,2].set(D[:,0] * tana)
        #A-B
        segment_bounds = segment_bounds.at[4*n_panels:4*(n_panels+n_trailing):4,0,:].set(ring_points[trailing_flag,3,:])
        segment_bounds = segment_bounds.at[4*n_panels:4*(n_panels+n_trailing):4,1,:].set(ring_points[trailing_flag,2,:])
        #B-C
        segment_bounds = segment_bounds.at[4*n_panels+1:4*(n_panels+n_trailing)+1:4,0,:].set(ring_points[trailing_flag,2,:])
        segment_bounds = segment_bounds.at[4*n_panels+1:4*(n_panels+n_trailing)+1:4,1,:].set(C)
        #C-D
        segment_bounds = segment_bounds.at[4*n_panels+2:4*(n_panels+n_trailing)+2:4,0,:].set(C)
        segment_bounds = segment_bounds.at[4*n_panels+2:4*(n_panels+n_trailing)+2:4,1,:].set(D)
        #D-A
        segment_bounds = segment_bounds.at[4*n_panels+3:4*(n_panels+n_trailing)+3:4,0,:].set(D)
        segment_bounds = segment_bounds.at[4*n_panels+3:4*(n_panels+n_trailing)+3:4,1,:].set(ring_points[trailing_flag, 3, :])

        #Symmetry segments - conditionally applied based on self.symmetry flag
        # When symmetry=False, these remain as zeros and don't contribute to circulation
        symmetry_flag = jnp.asarray(self.symmetry, dtype=jnp.float32)
        
        ring_A_sym = ring_points[:, 0, :].copy()
        ring_A_sym = ring_A_sym.at[:,1].set(-ring_A_sym[:,1])
        ring_B_sym = ring_points[:, 1, :].copy()
        ring_B_sym = ring_B_sym.at[:,1].set(-ring_B_sym[:,1])
        ring_C_sym = ring_points[:, 2, :].copy()
        ring_C_sym = ring_C_sym.at[:,1].set(-ring_C_sym[:,1])
        ring_D_sym = ring_points[:, 3, :].copy()
        ring_D_sym = ring_D_sym.at[:,1].set(-ring_D_sym[:,1])

        # Apply symmetry flag to bound segments (scale by 0 or 1)
        # A-B
        seg_ab_sym = jnp.where(symmetry_flag > 0.5, ring_B_sym, jnp.zeros_like(ring_B_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing):4*(2*n_panels+n_trailing):4,0,:].set(seg_ab_sym)
        seg_ab_sym_b = jnp.where(symmetry_flag > 0.5, ring_A_sym, jnp.zeros_like(ring_A_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing):4*(2*n_panels+n_trailing):4,1,:].set(seg_ab_sym_b)
        # B-C
        seg_bc_sym = jnp.where(symmetry_flag > 0.5, ring_A_sym, jnp.zeros_like(ring_A_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+1:4*(2*n_panels+n_trailing)+1:4,0,:].set(seg_bc_sym)
        seg_bc_sym_b = jnp.where(symmetry_flag > 0.5, ring_D_sym, jnp.zeros_like(ring_D_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+1:4*(2*n_panels+n_trailing)+1:4,1,:].set(seg_bc_sym_b)
        # C-D
        seg_cd_sym = jnp.where(symmetry_flag > 0.5, ring_D_sym, jnp.zeros_like(ring_D_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+2:4*(2*n_panels+n_trailing)+2:4,0,:].set(seg_cd_sym)
        seg_cd_sym_b = jnp.where(symmetry_flag > 0.5, ring_C_sym, jnp.zeros_like(ring_C_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+2:4*(2*n_panels+n_trailing)+2:4,1,:].set(seg_cd_sym_b)
        # D-A
        seg_da_sym = jnp.where(symmetry_flag > 0.5, ring_C_sym, jnp.zeros_like(ring_C_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+3:4*(2*n_panels+n_trailing)+3:4,0,:].set(seg_da_sym)
        seg_da_sym_b = jnp.where(symmetry_flag > 0.5, ring_B_sym, jnp.zeros_like(ring_B_sym))
        segment_bounds = segment_bounds.at[4*(n_panels+n_trailing)+3:4*(2*n_panels+n_trailing)+3:4,1,:].set(seg_da_sym_b)

        # Symmetry trailing edge segments - conditionally applied
        ring_B_sym_te = ring_points[trailing_flag, 2, :].copy()
        ring_A_sym_te = ring_points[trailing_flag, 3, :].copy()
        ring_C_sym_te = C.copy()
        ring_D_sym_te = D.copy()
        ring_A_sym_te = ring_A_sym_te.at[:,1].set(-ring_A_sym_te[:,1])
        ring_B_sym_te = ring_B_sym_te.at[:,1].set(-ring_B_sym_te[:,1])
        ring_C_sym_te = ring_C_sym_te.at[:,1].set(-ring_C_sym_te[:,1])
        ring_D_sym_te = ring_D_sym_te.at[:,1].set(-ring_D_sym_te[:,1])

        # A-B
        seg_ab_sym_te = jnp.where(symmetry_flag > 0.5, ring_B_sym_te, jnp.zeros_like(ring_B_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing):8*(n_panels+n_trailing):4,0,:].set(seg_ab_sym_te)
        seg_ab_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_A_sym_te, jnp.zeros_like(ring_A_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing):8*(n_panels+n_trailing):4,1,:].set(seg_ab_sym_te_b)
        # B-C
        seg_bc_sym_te = jnp.where(symmetry_flag > 0.5, ring_A_sym_te, jnp.zeros_like(ring_A_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+1:8*(n_panels+n_trailing)+1:4,0,:].set(seg_bc_sym_te)
        seg_bc_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_D_sym_te, jnp.zeros_like(ring_D_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+1:8*(n_panels+n_trailing)+1:4,1,:].set(seg_bc_sym_te_b)
        # C-D
        seg_cd_sym_te = jnp.where(symmetry_flag > 0.5, ring_D_sym_te, jnp.zeros_like(ring_D_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+2:8*(n_panels+n_trailing)+2:4,0,:].set(seg_cd_sym_te)
        seg_cd_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_C_sym_te, jnp.zeros_like(ring_C_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+2:8*(n_panels+n_trailing)+2:4,1,:].set(seg_cd_sym_te_b)
        # D-A
        seg_da_sym_te = jnp.where(symmetry_flag > 0.5, ring_C_sym_te, jnp.zeros_like(ring_C_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+3:8*(n_panels+n_trailing)+3:4,0,:].set(seg_da_sym_te)
        seg_da_sym_te_b = jnp.where(symmetry_flag > 0.5, ring_B_sym_te, jnp.zeros_like(ring_B_sym_te))
        segment_bounds = segment_bounds.at[4*(2*n_panels+n_trailing)+3:8*(n_panels+n_trailing)+3:4,1,:].set(seg_da_sym_te_b)
        
        # Correct Indexing: Repeat the GLOBAL panel indices
        global_indices = jnp.arange(offset, offset + n_panels)
        
        # Repeat indices for the 4 segments per panel
        segments_ids_temp = jnp.repeat(global_indices, 4)
        
        # Repeat for wake and symmetry as per your original logic
        # Make sure to slice your boolean flags correctly if they are local to the surface
        trailing_indices = global_indices[trailing_flag]
        
        segments_ids = jnp.concatenate((
            segments_ids_temp,                       # Bound Vortices
            jnp.repeat(trailing_indices, 4),        # Wake
            segments_ids_temp,                       # Symmetrical Bound
            jnp.repeat(trailing_indices, 4)         # Symmetrical Wake
        ))
        
        return segment_bounds, segments_ids

    def _assemble_AIC_mtx_parametrized(self,segments,segments_ids, control_points,normals) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Optimized version without Python loops."""
        #We have to compute the induced speed of each segment at each control point
        #List of all the vortex segments 
        points = control_points        
        n_points = points.shape[0]
        n_segments = segments.shape[0]
        
        # Direct construction (no .at[].set())
        # Repeat segments for each point
        A_all = jnp.tile(segments[:, 0, :], (n_points, 1))  # Shape: (n_points*n_segments, 3)
        B_all = jnp.tile(segments[:, 1, :], (n_points, 1))
        P_all = jnp.repeat(points, n_segments, axis=0)
        
        # Vectorized calculation of induced speeds
        induced_speed_vector_temp = self._calc_vorticity_vectorized(A_all, B_all, P_all)
        
        # Sum per panel with segment_sum (faster than reduceat)
        # We must repeat segment_ids for each point
        segment_ids_repeated = jnp.tile(segments_ids, n_points)
        
        # Add offset for each point
        point_offsets = jnp.repeat(jnp.arange(n_points) * n_points, n_segments)
        segment_ids_global = segment_ids_repeated + point_offsets
        
        # Sum of contributions
        induced_speed_vector = segment_sum(
            induced_speed_vector_temp,
            segment_ids_global,
            num_segments=n_points * n_points
        )
        
        # Reshape
        induced_speed_matrix = induced_speed_vector.reshape((n_points, n_points, 3))
        
        # Use pre-computed normals (no loop!)
        AIC_matrix = jnp.einsum('ijk,ik->ij', induced_speed_matrix, normals)
        
        return AIC_matrix, induced_speed_matrix  

    def _assemble_rhs_parametrized(self,normals,alpha,v_inf) -> jnp.ndarray:
        """Optimized version without loops."""
        # Use pre-computed normals
        cosa = jnp.cos(alpha * jnp.pi / 180.)
        sina = jnp.sin(alpha * jnp.pi / 180.)
        u = jnp.array([cosa, 0.0, sina])
        v = v_inf * u
        
        rhs = jnp.dot(normals, v)
        return rhs

    def compute_circulation_parametrized(self,segments,segments_ids, control_points,normals,alpha,v_inf) -> jnp.ndarray:
        """Optimized version."""
        aic,_ = self._assemble_AIC_mtx_parametrized(segments,segments_ids, control_points,normals)
        rhs = self._assemble_rhs_parametrized(normals,alpha,v_inf)
        gamma = jnp.linalg.solve(aic, -rhs)
        return gamma   

    @staticmethod
    @jit
    def _calc_vorticity_vectorized(A, B, P):
        """
        Complete vectorized calculation of induced velocity.
        A, B, P: arrays of shape (N, 3)
        Returns: array of shape (N, 3)
        """
        r1 = P - A
        r2 = P - B
        r0 = r1 - r2
        
        r1_mag = jnp.linalg.norm(r1, axis=1, keepdims=True)
        r2_mag = jnp.linalg.norm(r2, axis=1, keepdims=True)
        cross = jnp.cross(r1, r2)
        cross_mag_sq = jnp.sum(cross**2, axis=1, keepdims=True)
        
        dot_r1 = jnp.sum(r0 * r1, axis=1, keepdims=True)
        dot_r2 = jnp.sum(r0 * r2, axis=1, keepdims=True)
        
        # Avoid divisions by zero
        eps = 1e-10
        a = (dot_r1 / (r1_mag + eps) - dot_r2 / (r2_mag + eps)) / (cross_mag_sq + eps)
        
        # Mask singularities
        mask = (r1_mag > eps) & (r2_mag > eps) & (cross_mag_sq > eps)
        
        v = cross * a * mask
        v = v / (4.0 * jnp.pi)
        
        return v

    @partial(jit, static_argnums=(0,))
    def _assemble_AIC_mtx_optimized(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Optimized version without Python loops.
        Uses 3/4 chord control points for circulation computation.
        """
        #We have to compute the induced speed of each segment at each control point
        #List of all the vortex segments 
        segments = jnp.array(self.global_segment_bounds)
        points = jnp.array(self.global_control_points)  # Use 3/4 chord control points
        
        n_points = points.shape[0]
        n_segments = segments.shape[0]
        
        # Direct construction (no .at[].set())
        # Repeat segments for each point
        A_all = jnp.tile(segments[:, 0, :], (n_points, 1))  # Shape: (n_points*n_segments, 3)
        B_all = jnp.tile(segments[:, 1, :], (n_points, 1))
        P_all = jnp.repeat(points, n_segments, axis=0)
        
        # Vectorized calculation of induced speeds
        induced_speed_vector_temp = self._calc_vorticity_vectorized(A_all, B_all, P_all)
        
        # Sum per panel with segment_sum (faster than reduceat)
        # We must repeat segment_ids for each point
        segment_ids_repeated = jnp.tile(self.segment_ids, n_points)
        
        # Add offset for each point
        point_offsets = jnp.repeat(jnp.arange(n_points) * n_points, n_segments)
        segment_ids_global = segment_ids_repeated + point_offsets
        
        # Sum of contributions
        induced_speed_vector = segment_sum(
            induced_speed_vector_temp,
            segment_ids_global,
            num_segments=n_points * n_points
        )
        
        # Reshape
        induced_speed_matrix = induced_speed_vector.reshape((n_points, n_points, 3))
        
        # Use pre-computed normals (no loop!)
        AIC_matrix = jnp.einsum('ijk,ik->ij', induced_speed_matrix, self.all_normals)
        
        return AIC_matrix, induced_speed_matrix

    @partial(jit, static_argnums=(0,))
    def _assemble_induced_speed_quarter_chord(self,global_segment_bounds,global_control_points_quart,segment_ids) -> jnp.ndarray:
        """Compute induced velocity matrix at quarter-chord control points.
        Used specifically for force calculations.
        Returns the induced speed matrix (n_quart, n_panels, 3).
        """
        #We have to compute the induced speed of each segment at each quarter-chord control point
        #List of all the vortex segments 
        segments = jnp.array(global_segment_bounds)
        points = jnp.array(global_control_points_quart)  # Use quarter-chord control points (1/4 chord)
        
        n_points = points.shape[0]
        n_segments = segments.shape[0]
        
        # Direct construction (no .at[].set())
        # Repeat segments for each point
        A_all = jnp.tile(segments[:, 0, :], (n_points, 1))  
        B_all = jnp.tile(segments[:, 1, :], (n_points, 1))
        P_all = jnp.repeat(points, n_segments, axis=0)
        
        # Vectorized calculation of induced speeds
        induced_speed_vector_temp = self._calc_vorticity_vectorized(A_all, B_all, P_all)
        
        # Sum per panel with segment_sum
        # We must repeat segment_ids for each point
        segment_ids_repeated = jnp.tile(segment_ids, n_points)
        
        # Add offset for each point
        point_offsets = jnp.repeat(jnp.arange(n_points) * n_points, n_segments)
        segment_ids_global = segment_ids_repeated + point_offsets
        
        # Sum of contributions
        induced_speed_vector = segment_sum(
            induced_speed_vector_temp,
            segment_ids_global,
            num_segments=n_points * n_points
        )
        
        # Reshape
        induced_speed_matrix = induced_speed_vector.reshape((n_points, n_points, 3))
        
        return induced_speed_matrix

    @partial(jit, static_argnums=(0,))
    def compute_circulation_optimized(self) -> jnp.ndarray:
        """Version optimisee."""
        aic = self._assemble_AIC_mtx_optimized()[0]
        rhs = self._assemble_rhs_optimized()
        gamma = jnp.linalg.solve(aic, -rhs)
        return gamma

    @partial(jit, static_argnums=(0,))
    def compute_CL_CD_forces(self, gamma,alpha,S_ref,global_segment_bounds,global_control_points_quart,segment_ids, all_ring_points):
        """
        Computes lift, drag coefficients and forces using Kutta-Joukowski theorem.
        Fully vectorized version compatible with JAX.
        
        Inputs:
            gamma: array [n_panels_tot], vector of circulations
        
        Outputs:
            CL: float, lift coefficient
            CD: float, drag coefficient (induced drag only)
            forces: array [n_panels_tot, 3], local force vector at each panel
            delta_L: array [n_panels_tot], lift contribution per panel
            delta_D: array [n_panels_tot], drag contribution per panel
        """
        # alpha 
        cosa = jnp.cos(alpha * np.pi / 180.)
        sina = jnp.sin(alpha * np.pi / 180.)
        tana = sina/cosa
        # Freestream velocity vector
        v_inf_vec = jnp.array([cosa, 0.0, sina]) * self.v_inf
        
        # Calculation of induced velocity at quarter-chord points
        induced_speed_matrix = self._assemble_induced_speed_quarter_chord(global_segment_bounds,global_control_points_quart,segment_ids)
        
        # Induced velocity at each quarter-chord control point [n_panels, 3]
        w_induced = jnp.einsum('ijk,j->ik', induced_speed_matrix, gamma)
        
        
        # Local velocity = V_inf + w_induced
        n_panels = self.all_ring_points_np.shape[0]
        v_local = jnp.tile(v_inf_vec, (n_panels, 1)) + w_induced
        
        # Vector of the bound segment (bound vortex) for each panel
        # bound = B - A, where A is the 1st ring point and B is the 2nd
        bound = jnp.array(all_ring_points[:, 1, :] - all_ring_points[:, 0, :])
        
        # Determination of effective circulation
        # For leading-edge panels: full gamma
        # For internal panels: circulation difference with upstream panel
        
        # is_leading_edge flags
        is_leading_edge = jnp.array(self.all_leading_edge_np)
        
        # Effective gamma for internal panels
        # gamma_eff[j] = gamma[panel_pair_indices[j]] - gamma[j]
        panel_pair_indices = jnp.array(self.panel_pair_indices_np)

        # Number of leading-edge panels (stored as numpy on the object, safe to read)
        n_leading_edge = int(np.sum(self.all_leading_edge_np))
        # Number of total panels
        n_total_panels = len(gamma)

        # Compute gamma differences for paired (internal) panels
        gamma_from_pair = gamma[jnp.maximum(panel_pair_indices, 0)]  # Get gamma from paired panel (or gamma[0] for -1)
        gamma_diff_full = gamma_from_pair - gamma
        gamma_diff = jnp.zeros((n_total_panels,))

        gamma_diff = gamma_diff.at[panel_pair_indices[self.ind_panel_pairs]].set(gamma_diff_full[self.ind_panel_pairs])
        
        # Select effective gamma according to panel type
        gamma_eff = jnp.where(is_leading_edge, gamma, gamma_diff)
        
        # Kutta-Joukowski theorem: F = rho * Gamma * (V_local x l)
        # where l is the bound segment vector
        forces = self.rho * gamma_eff[:, jnp.newaxis] * jnp.cross(v_local, bound)
        
        # Projection in aerodynamic frame (lift/drag)
        # Lift = component perpendicular to V_inf (generally in z)
        # Drag = component parallel to V_inf (generally in x)
        delta_L = forces[:, 2] * cosa - forces[:, 0] * sina
        delta_D = forces[:, 0] * cosa + forces[:, 2] * sina
        
        # Calculation of total forces and coefficients
        L = jnp.sum(delta_L)
        D = jnp.sum(delta_D)
        
        # Aerodynamic coefficients
        q_inf = 0.5 * self.rho * self.v_inf**2
        CL = L / (q_inf * S_ref)
        CD = D / (q_inf * S_ref)
        
        # Note: CD only contains induced drag
        # Friction and form drag are not calculated in VLM
        
        return CL, CD, forces, delta_L, delta_D
    
    
    def compute_cl_distribution_span(self, delta_L, all_ring_points,tol=1e-3):
        """
        Computes the integrated lift coefficient distribution over the span.
        Integrates cl distribution across chord direction for each span column.
        Handles multiple surfaces by organizing output by surface ID.
        
        Inputs:
            delta_L: array [n_panels], lift contribution per panel
            all_ring_points: array [n_panels, 4, 3], ring point coordinates for each panel
            tol: float, tolerance for grouping panels into spanwise columns (default: 1e-3)
        Outputs:
            y_span: array [n_panels], span-wise position (y-coordinate) at panel center
            cl_local: array [n_panels], local lift coefficient for each panel
            cl_distribution: dict with keys:
                'y_column': span positions of panel columns (aggregated over all surfaces)
                'cl_integrated': integrated Cl for each column (integral of cl over chord at each span position)
                'surfaces': dict organized by surface_id with surface-specific data:
                    - 'y_column': span positions for this surface
                    - 'cl_integrated': integrated Cl for this surface
                    - 'y_all': all y-positions for panels in this surface
                    - 'cl_all': all local Cl values for panels in this surface
        """
        n_panels = all_ring_points.shape[0]
        
        # Extract span-wise position from ring points (y-coordinate at the middle of the bound segment)
        # Ring points: [0] = A, [1] = B (leading edge left), [2] = C (trailing edge left), [3] = D
        # The span position is the y-coordinate of the bound segment (between A and B)
        y_span = (all_ring_points[:, 0, 1] + all_ring_points[:, 1, 1]) / 2.0
        
        # Compute chord length (distance from leading edge to trailing edge in x-direction)
        # This is the x-distance between ring points [1] (B) and [2] (C)
        # This is the chordwise extent (delta_x) for each panel
        chord = jnp.abs(all_ring_points[:, 2, 0] - all_ring_points[:, 1, 0])
        
        # Compute span-wise extent (delta_y) from the y-coordinate difference
        # Between ring points [0] (A) and [1] (B)
        delta_y = jnp.abs(all_ring_points[:, 1, 1] - all_ring_points[:, 0, 1])
        
        # Dynamic pressure
        q_inf = 0.5 * self.rho * self.v_inf**2
        
        # Local lift coefficient: cl = 2 * delta_L / (q_inf * chord * delta_y)
        # Avoid division by zero
        eps = 1e-10
        cl_local = 2.0 * delta_L / (q_inf * (chord + eps) * (delta_y + eps))
        
        # Convert to numpy for grouping and sorting
        y_span_np = np.array(y_span)
        cl_local_np = np.array(cl_local)
        chord_np = np.array(chord)
        delta_y_np = np.array(delta_y)
        surface_ids_np = self.surface_ids_np
        
        # Sort by y-position
        sorted_indices = np.argsort(y_span_np)
        y_sorted = y_span_np[sorted_indices]
        cl_sorted = cl_local_np[sorted_indices]
        chord_sorted = chord_np[sorted_indices]
        delta_y_sorted = delta_y_np[sorted_indices]
        surface_ids_sorted = surface_ids_np[sorted_indices]
        
        # Group panels into columns: find discontinuities in y-position
        tolerance = tol * (np.max(y_sorted) - np.min(y_sorted)) if np.max(y_sorted) > np.min(y_sorted) else tol
        column_edges = [0]
        
        for i in range(1, len(y_sorted)):
            if np.abs(y_sorted[i] - y_sorted[i-1]) > tolerance:
                column_edges.append(i)
        column_edges.append(len(y_sorted))
        
        # Integrate cl over chord for each column
        y_column = []
        cl_integrated = []
        
        for i in range(len(column_edges) - 1):
            start_idx = column_edges[i]
            end_idx = column_edges[i + 1]
            
            # Column span position (average y of panels in this column)
            y_col = np.mean(y_sorted[start_idx:end_idx])
            
            # Integrated Cl over chord: integral of cl * dx over the chord direction
            # For discretized panels: sum of cl * chord_length
            cl_int = np.sum(cl_sorted[start_idx:end_idx] * chord_sorted[start_idx:end_idx])
            
            # # Normalize by the total chord size (sum of delta_x) for this column
            chord_total = np.sum(chord_sorted[start_idx:end_idx])
            eps = 1e-10
            cl_int_normalized = cl_int / (chord_total + eps)
            
            y_column.append(y_col)
            cl_integrated.append(cl_int_normalized)
        
        # Organize data by surface
        all_surfaces = sorted(set(self.surface_ids_np))
        surface_distributions = {}
        
        for surface_id in all_surfaces:
            # Find indices for this surface
            surface_mask = surface_ids_np == surface_id
            surface_panel_indices = np.where(surface_mask)[0]
            
            if len(surface_panel_indices) == 0:
                continue
            
            # Extract data for this surface
            y_surf = y_span_np[surface_panel_indices]
            cl_surf = cl_local_np[surface_panel_indices]
            chord_surf = chord_np[surface_panel_indices]
            delta_y_surf = delta_y_np[surface_panel_indices]
            
            # Sort by y-position for this surface
            surf_sorted_indices = np.argsort(y_surf)
            y_surf_sorted = y_surf[surf_sorted_indices]
            cl_surf_sorted = cl_surf[surf_sorted_indices]
            chord_surf_sorted = chord_surf[surf_sorted_indices]
            delta_y_surf_sorted = delta_y_surf[surf_sorted_indices]
            
            # Group into columns for this surface
            y_surf_min = np.min(y_surf_sorted) if len(y_surf_sorted) > 0 else 0
            y_surf_max = np.max(y_surf_sorted) if len(y_surf_sorted) > 0 else 1
            tolerance_surf = tol * (y_surf_max - y_surf_min) if y_surf_max > y_surf_min else tol
            
            column_edges_surf = [0]
            for i in range(1, len(y_surf_sorted)):
                if np.abs(y_surf_sorted[i] - y_surf_sorted[i-1]) > tolerance_surf:
                    column_edges_surf.append(i)
            column_edges_surf.append(len(y_surf_sorted))
            
            # Integrate cl over chord for each column in this surface
            y_col_surf = []
            cl_int_surf = []
            
            for i in range(len(column_edges_surf) - 1):
                start_idx = column_edges_surf[i]
                end_idx = column_edges_surf[i + 1]
                
                y_col = np.mean(y_surf_sorted[start_idx:end_idx])
                cl_int = np.sum(cl_surf_sorted[start_idx:end_idx] * chord_surf_sorted[start_idx:end_idx])
                
                # Normalize by the total chord size (sum of delta_x) for this column
                chord_total = np.sum(chord_surf_sorted[start_idx:end_idx])
                eps = 1e-10
                cl_int_normalized = cl_int/ (chord_total + eps)
                
                y_col_surf.append(y_col)
                cl_int_surf.append(cl_int_normalized)
            
            surface_distributions[surface_id] = {
                'y_column': np.array(y_col_surf),
                'cl_integrated': np.array(cl_int_surf),
                'y_all': y_surf,
                'cl_all': cl_surf
            }
        
        cl_distribution = {
            'y_column': np.array(y_column),
            'cl_integrated': np.array(cl_integrated),
            'y_all': y_span_np,
            'cl_all': cl_local_np,
            'surfaces': surface_distributions  # New: organized by surface
        }
        
        return y_span_np, cl_local_np, cl_distribution
    
    @partial(jit, static_argnums=(0,))
    def compute_force_at_nodes(self,forces):
        #forces are computed at the middle 1/4 chord of each panel, we have to project this forces at nodes
        forces_at_nodes = jnp.zeros((self.nodes.shape[0],3))
        i = 0
        for surface in self.surfaces:
            for points in surface['mesh']:
                A = int(points[1]-1)
                B = int(points[2]-1)
                C = int(points[3]-1)
                D = int(points[4]-1)
                for x in [A,B,C,D]:
                    forces_at_nodes = forces_at_nodes.at[x,:].add(forces[i,:]/4.0)
                i = i + 1    
        return forces_at_nodes 
    
    def post_process_deformed_mesh_file(self,Ua,mesh_file_out):
        #copying the mesh file and replacing the points coordinates by the new one
        gmsh.initialize()
        gmsh.open(self.mesh_file)
        nodes_tag,nodes_coord,par_coord = gmsh.model.mesh.getNodes()
        n_nodes = int(len(nodes_tag))
        nodes_coord = nodes_coord.reshape((n_nodes,3))
        new_nodes_coord = nodes_coord+Ua
        for i in range(n_nodes):
            gmsh.model.mesh.setNode(nodes_tag[i],new_nodes_coord[i],par_coord)
        #saving the deformed mesh
        gmsh.write(mesh_file_out)    
        return 

    #post processing of the effort for visualization in GMSH
    #inputs: \n
    #gamma: array, circulations vector computed by compute_cirdulation\n    
    #forces: array, forces vector computed by compute_velocity_and_forces \n
    #output:\n
    #New mesh_file with post processing data
    def post_processing(self,input_mesh,file_name,forces):
        #copying mesh file
        with open(input_mesh) as f:
            with open(file_name+".msh", "w") as f1:
                for line in f:
                        f1.write(line)
        f.close()
        f1.close()
        n_panels = 0
        for surface in self.surfaces:
            n_panels = n_panels + len(surface['mesh'])
        f=open(file_name+".msh",'a')
        f.write('$ElementData\n')
        f.write('1\n')
        f.write('"Force"\n')
        f.write('1\n')
        f.write('0.0\n')
        f.write('3\n')
        f.write('0\n')
        f.write('3\n')
        f.write(str(int(n_panels))+'\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f %f %f \n' \
                %(forces[j,0],forces[j,1],forces[j,2]))
                j = j + 1
        f.write('$EndElementData\n')
        f.write('$ElementData\n')
        f.write('1\n')
        f.write('"Delta P"\n')
        f.write('1\n')
        f.write('0.0\n')
        f.write('3\n')
        f.write('0\n')
        f.write('1\n')
        f.write(str(int(n_panels))+'\n')
        j = 0
        for surface in self.surfaces:
            for i in range(len(surface['mesh'])):
                f.write(str(int(surface['mesh'][i,0]))+' %f \n' \
                %((np.linalg.norm(forces[j,:])/surface['areas'][i])[0]))
                j = j + 1
        f.write('$EndElementData\n')
        f.close()        
        return   
