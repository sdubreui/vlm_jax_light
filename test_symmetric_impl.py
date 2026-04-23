#!/usr/bin/env python3
"""Test script for non-symmetric VLM implementation"""

import sys
import os
sys.path.insert(0, '/stck/sdubreui/vlm_jax_light')

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

# Try importing VLM module
try:
    from VLM_light.VLM import VlmStudyOptimized
    print("✓ VlmStudyOptimized module imported successfully")
except Exception as e:
    print(f"✗ Failed to import VlmStudyOptimized: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with existing symmetric mesh
mesh_file_sym = '/stck/sdubreui/vlm_jax_light/VLM_light/test/meshes/rectangular_wing_20_40.msh'

# Test 1: Try creating VLM instance with symmetry=True
try:
    vlm_sym = VlmStudyOptimized(
        mesh_file=mesh_file_sym,
        alpha=0.0,
        v_inf=100.0,
        rho=0.0023770,
        x_wake=1e6,
        symmetry=True
    )
    print("✓ VlmStudyOptimized instance created with symmetry=True")
except Exception as e:
    print(f"✗ Failed to create VlmStudyOptimized with symmetry=True: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Try creating VLM instance with symmetry=False
try:
    vlm_nosym = VlmStudyOptimized(
        mesh_file=mesh_file_sym,
        alpha=0.0,
        v_inf=100.0,
        rho=0.0023770,
        x_wake=1e6,
        symmetry=False
    )
    print("✓ VlmStudyOptimized instance created with symmetry=False")
except Exception as e:
    print(f"✗ Failed to create VlmStudyOptimized with symmetry=False: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All initialization tests passed!")

