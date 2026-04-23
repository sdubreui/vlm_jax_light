#!/usr/bin/env python3
"""Comprehensive test for symmetric vs non-symmetric VLM calculations"""

import sys
sys.path.insert(0, '/stck/sdubreui/vlm_jax_light')

import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from VLM_light.VLM import VlmStudyOptimized

mesh_file = '/stck/sdubreui/vlm_jax_light/VLM_light/test/meshes/rectangular_wing_20_40.msh'

# Test parameters
alpha = 5.0  # 5 degree angle of attack
v_inf = 100.0
rho = 0.0023770
S_ref = 10.0

print("="*60)
print("VLM Non-Symmetric Configuration Test")
print("="*60)
print(f"Alpha: {alpha}°")
print(f"V_inf: {v_inf} m/s")
print(f"Rho: {rho} kg/m³")
print(f"S_ref: {S_ref} m²")
print("="*60)

# Test 1: Run with symmetry=True
print("\n[TEST 1] Running with symmetry=True (original symmetric configuration)")
try:
    study_sym = VlmStudyOptimized(
        mesh_file=mesh_file,
        alpha=alpha,
        v_inf=v_inf,
        rho=rho,
        x_wake=1e6,
        symmetry=True
    )
    
    surfaces_sym = study_sym.compute_topology()
    nodes_sym = study_sym.nodes[:,1:]
    segments_sym, segments_ids_sym, control_points_sym, normals_sym, control_point_quart_sym, ring_pts_sym = \
        study_sym.compute_geometry(nodes_sym, surfaces_sym, alpha)
    
    gamma_sym = study_sym.compute_circulation_parametrized(
        segments_sym, segments_ids_sym, control_points_sym, normals_sym, alpha, v_inf
    )
    
    CL_sym, CD_sym, forces_sym, delta_L_sym, delta_D_sym = study_sym.compute_CL_CD_forces(
        gamma_sym, alpha, S_ref, segments_sym, control_point_quart_sym, segments_ids_sym, ring_pts_sym
    )
    
    print(f"✓ Computation successful")
    print(f"  CL = {CL_sym:.6f}")
    print(f"  CD = {CD_sym:.6f}")
    print(f"  Number of vortices: {len(gamma_sym)}")
    
except Exception as e:
    print(f"✗ Computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Run with symmetry=False
mesh_file = '/stck/sdubreui/vlm_jax_light/VLM_light/test/meshes/rectangular_wing_20_40_no_sym.msh'
print("\n[TEST 2] Running with symmetry=False (non-symmetric configuration)")
try:
    study_nosym = VlmStudyOptimized(
        mesh_file=mesh_file,
        alpha=alpha,
        v_inf=v_inf,
        rho=rho,
        x_wake=1e6,
        symmetry=False
    )
    
    surfaces_nosym = study_nosym.compute_topology()
    nodes_nosym = study_nosym.nodes[:,1:]
    segments_nosym, segments_ids_nosym, control_points_nosym, normals_nosym, control_point_quart_nosym, ring_pts_nosym = \
        study_nosym.compute_geometry(nodes_nosym, surfaces_nosym, alpha)
    
    gamma_nosym = study_nosym.compute_circulation_parametrized(
        segments_nosym, segments_ids_nosym, control_points_nosym, normals_nosym, alpha, v_inf
    )
    
    CL_nosym, CD_nosym, forces_nosym, delta_L_nosym, delta_D_nosym = study_nosym.compute_CL_CD_forces(
        gamma_nosym, alpha, 2*S_ref, segments_nosym, control_point_quart_nosym, segments_ids_nosym, ring_pts_nosym
    )
    
    print(f"✓ Computation successful")
    print(f"  CL = {CL_nosym:.6f}")
    print(f"  CD = {CD_nosym:.6f}")
    print(f"  Number of vortices: {len(gamma_nosym)}")
    
except Exception as e:
    print(f"✗ Computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"CL ratio (sym/nosym):  {CL_sym/CL_nosym:.4f}")
print(f"CD ratio (sym/nosym):  {CD_sym/CD_nosym:.4f}")
print(f"Gamma size sym:    {len(gamma_sym)}")
print(f"Gamma size nosym:  {len(gamma_nosym)}")
print("\n✓ All tests completed successfully!")
