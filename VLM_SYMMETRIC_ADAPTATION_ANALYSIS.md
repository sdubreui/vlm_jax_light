# VLM Non-Symmetric Configuration Analysis & Adaptation Strategy

## Overview
The current VLM implementation assumes symmetric configuration (about the xz-plane, y-mirror). The symmetry is hardcoded in several functions rather than being properly parameterized. To support non-symmetric cases while maintaining JAX compatibility, we need to refactor these areas without using Python runtime `if` statements.

---

## Issues Identified & Solutions

### 1. **Segment Generation in `read_gmsh_mesh()` (lines 187-283)**

**Current Issue:**
- Lines 215-265: Symmetry segments are always duplicated using y-axis reflection
- Lines 216-233: Wake segments are doubled when `self.symmetry = True`
- This hardcodes the assumption that `global_segment_bounds` always contains ~2x segments

**Problem for JAX:**
- Array shape is fixed but wastes memory when `symmetry=False`
- Segment IDs and logic are deeply coupled to symmetry assumption
- The `global_sum_index` array tracks variable segment counts per panel

**Proposed Solution:**
```python
# Create fixed-size arrays for both symmetric and non-symmetric cases
# Use masking to zero out unused segments rather than changing array shape

def _precompute_segment_bounds_adaptive(self):
    """Pre-compute segment bounds with conditional symmetry handling."""
    
    # Maximum possible segments per panel: 16 (4 bound + 4 wake + 4 sym_bound + 4 sym_wake)
    max_segments_per_panel = 16
    
    all_segments = []
    all_segment_masks = []  # Boolean mask: 1 if segment is active, 0 if padded
    segment_to_panel = []
    
    for surface in self.surfaces:
        segments_surface = []
        masks_surface = []
        
        for panel_idx, panel in enumerate(surface['ring_points']):
            is_trailing = panel_idx in surface['trailing_edge']
            
            # Basic 4 segments (always present)
            seg_count = 4
            segments_surface.append([
                [panel[0], panel[1]],  # A-B
                [panel[1], panel[2]],  # B-C
                [panel[2], panel[3]],  # C-D
                [panel[3], panel[0]]   # D-A
            ])
            
            # Wake segments (conditional on trailing edge)
            if is_trailing:
                wake_C = panel[2].copy()
                wake_C[0] = self.x_wake
                wake_C[2] = wake_C[0] * self.tana
                wake_D = panel[3].copy()
                wake_D[0] = self.x_wake
                wake_D[2] = wake_D[0] * self.tana
                
                segments_surface.append([
                    [panel[3], panel[2]],   # D-C
                    [panel[2], wake_C],     # C-C_wake
                    [wake_C, wake_D],       # C_wake-D_wake
                    [wake_D, panel[3]]      # D_wake-D
                ])
                seg_count = 8
            else:
                # Pad with zeros
                segments_surface.append([
                    [np.zeros(3), np.zeros(3)],
                    [np.zeros(3), np.zeros(3)],
                    [np.zeros(3), np.zeros(3)],
                    [np.zeros(3), np.zeros(3)]
                ])
            
            # Symmetry segments (conditional)
            if self.symmetry:
                # Mirror y-coordinates for all 8 segments already added
                # This requires creating the symmetric versions
                # ... (symmetry logic with padding if no wake)
                seg_count *= 2
            else:
                # Pad with zeros
                segments_surface.append([
                    [np.zeros(3), np.zeros(3)] for _ in range(8)
                ])
            
            # Create mask for active vs padded segments
            mask = np.ones(max_segments_per_panel, dtype=bool)
            mask[seg_count:] = False
            masks_surface.append(mask)
    
    # Convert to JAX arrays
    self.all_segment_bounds = jnp.array(all_segments)  # Shape: (n_panels, 16, 2, 3)
    self.segment_active_mask = jnp.array(all_segment_masks)  # Shape: (n_panels, 16)
```

---

### 2. **`compute_segment_bounds()` Function (lines 543-643)**

**Current Issue:**
- Always allocates space for symmetric segments: `4*(n_panels+n_trailing)*2`
- Unconditionally creates symmetric segments (lines 582-618)
- This must be called with `self.symmetry` context

**Proposed Solution:**
```python
@partial(jit, static_argnums=(0, 4))
def compute_segment_bounds_adaptive(self, ring_points, trailing_flag, 
                                     panel_offset, symmetry_flag, alpha):
    """
    Compute segment bounds with adaptive symmetry handling using JAX operations.
    
    Args:
        symmetry_flag: boolean, whether to include symmetric segments
    """
    n_panels = ring_points.shape[0]
    n_trailing = jnp.sum(trailing_flag.astype(jnp.int32))
    
    # Calculate required segment size
    base_segments = 4 * n_panels + 4 * n_trailing
    total_segments = jnp.where(symmetry_flag, 2 * base_segments, base_segments)
    
    # Pre-allocate maximum size (for static shape)
    segment_bounds = jnp.zeros((4 * (n_panels + n_trailing) * 2, 2, 3))
    segment_mask = jnp.zeros((4 * (n_panels + n_trailing) * 2,), dtype=bool)
    
    # ... Compute segments for all cases ...
    
    # Use jnp.where to conditionally include symmetry segments
    # Active segments indices
    base_idx = 4 * n_panels + 4 * n_trailing
    segment_mask = segment_mask.at[:base_idx].set(True)
    segment_mask = segment_mask.at[base_idx:2*base_idx].set(symmetry_flag)
    
    return segment_bounds, segment_mask
```

---

### 3. **`_compute_segment_bounds_with_offset()` (lines 704-816)**

**Current Issue:**
- Lines 770-816: Hardcoded symmetry calculations
- Always creates symmetric wake segments

**Proposed Solution:**
```python
def _compute_segment_bounds_with_offset_adaptive(self, ring_points, trailing_flag, 
                                                 offset, alpha, include_symmetry):
    """
    Compute segment bounds with optional symmetry using JAX operations.
    
    Uses jnp.where() to conditionally apply symmetry without Python if statements.
    """
    n_panels = ring_points.shape[0]
    n_trailing = jnp.sum(trailing_flag.astype(jnp.int32))
    
    # Maximum allocation (worst case: with symmetry and trailing edge)
    max_segments = 4 * (2*n_panels + 2*n_trailing)
    segment_bounds = jnp.zeros((max_segments, 2, 3))
    
    # Base segments (bound vortex)
    idx = 0
    segment_bounds = segment_bounds.at[idx:idx+4*n_panels:4, 0, :].set(ring_points[:, 0, :])
    segment_bounds = segment_bounds.at[idx:idx+4*n_panels:4, 1, :].set(ring_points[:, 1, :])
    # ... other bound segments ...
    
    # Trailing edge segments (conditional on trailing_flag)
    cosa = jnp.cos(alpha * jnp.pi / 180.)
    sina = jnp.sin(alpha * jnp.pi / 180.)
    tana = sina / cosa
    
    wake_C = ring_points[trailing_flag, 2, :]
    wake_C = wake_C.at[:, 0].set(self.x_wake)
    wake_C = wake_C.at[:, 2].set(wake_C[:, 0] * tana)
    
    # Conditionally add wake segments
    # ... use jnp.where() to select between zero and wake segment ...
    
    # Symmetry segments (conditional)
    sym_segments_active = include_symmetry
    
    ring_points_sym = ring_points.at[:, 1].set(-ring_points[:, 1])  # Mirror y
    # ... apply symmetry logic conditionally using jnp.where() ...
    
    return segment_bounds, segment_mask
```

---

### 4. **Force Calculation in `compute_CL_CD_forces()` (lines 1020-1080)**

**Current Issue:**
- Line 1043: Assumes `self.all_ring_points_np` contains only half-wing when symmetry is applied
- Force calculations don't account for full wingspan when symmetry=False
- The factor applied to CL/CD implicitly assumes symmetry

**Proposed Solution:**
```python
@partial(jit, static_argnums=(0,))
def compute_CL_CD_forces_adaptive(self, gamma, alpha, S_ref, 
                                  global_segment_bounds, global_control_points_quart,
                                  segment_ids, all_ring_points, symmetry_flag):
    """
    Compute lift/drag with adaptive symmetry handling.
    
    When symmetry=False: use S_ref for full configuration
    When symmetry=True: multiply forces by 2 (or adjust S_ref accordingly)
    """
    # ... basic calculations same as before ...
    
    # Compute raw forces
    forces = self.rho * gamma_eff[:, jnp.newaxis] * jnp.cross(v_local, bound)
    
    # Apply symmetry factor to forces
    symmetry_factor = jnp.where(symmetry_flag, 2.0, 1.0)
    forces = forces * symmetry_factor
    
    # ... rest of calculation ...
    
    # Coefficients accounting for symmetry
    CL = L / (q_inf * S_ref)
    CD = D / (q_inf * S_ref)
    
    return CL, CD, forces, delta_L, delta_D
```

---

### 5. **Circulation Assembly in `_assemble_rhs_parametrized()` (lines 905-911)**

**Current Issue:**
- Lines 905-911: Assumes symmetric configuration for RHS computation
- When symmetry=False, the RHS should only account for the actual surfaces

**Proposed Solution:**
```python
@partial(jit, static_argnums=(0,))
def _assemble_rhs_parametrized_adaptive(self, normals, alpha, v_inf, 
                                        symmetry_flag, normal_mask=None):
    """
    Assemble RHS with adaptive symmetry handling.
    
    Args:
        normal_mask: boolean array to mask which panels are active (for padded arrays)
    """
    cosa = jnp.cos(alpha * jnp.pi / 180.)
    sina = jnp.sin(alpha * jnp.pi / 180.)
    u = jnp.array([cosa, 0.0, sina])
    v = v_inf * u
    
    rhs = jnp.dot(normals, v)
    
    # Mask out padded entries if needed
    if normal_mask is not None:
        rhs = jnp.where(normal_mask, rhs, 0.0)
    
    return rhs
```

---

## Summary of Required Changes

| Function | Lines | Change Type | JAX Compatible |
|----------|-------|-------------|-----------------|
| `read_gmsh_mesh()` | 187-283 | Use masking arrays | ✅ Yes |
| `compute_segment_bounds()` | 543-643 | Add `@static_argnums` for symmetry_flag | ✅ Yes |
| `_compute_segment_bounds_with_offset()` | 704-816 | Use `jnp.where()` instead of `if` | ✅ Yes |
| `_assemble_rhs_parametrized()` | 905-911 | Add masking capability | ✅ Yes |
| `compute_CL_CD_forces()` | 1020-1080 | Apply symmetry factor multiplier | ✅ Yes |
| `__init__()` | 24-50 | Pass symmetry to all computation functions | ✅ Yes |

---

## Implementation Strategy

1. **Preserve backward compatibility**: Keep existing functions but add `*_adaptive` versions
2. **Use static argnums**: For boolean symmetry flags so JAX JIT compiles separate versions
3. **Use masking**: Instead of changing array shapes, use boolean masks for active segments
4. **Apply symmetry factor**: Multiply forces by 2 when symmetry=True rather than doubling segments in pre-computation
5. **Test incrementally**: Symmetric case first (should match current), then non-symmetric

---

## Example Usage After Refactoring

```python
# Symmetric case (current behavior)
vlm_sym = VlmStudyOptimized(mesh_file="wing.msh", symmetry=True)
gamma_sym = vlm_sym.compute_circulation_adaptive()
CL_sym, CD_sym = vlm_sym.compute_CL_CD_forces_adaptive(..., symmetry_flag=True)

# Non-symmetric case (new capability)
vlm_full = VlmStudyOptimized(mesh_file="wing.msh", symmetry=False)
gamma_full = vlm_full.compute_circulation_adaptive()
CL_full, CD_full = vlm_full.compute_CL_CD_forces_adaptive(..., symmetry_flag=False)
```

---

## Benefits

✅ **JAX Compatible**: No runtime Python conditionals, only `jnp.where()` and masking
✅ **Backward Compatible**: Symmetric case unchanged
✅ **Efficient**: Single compilation path with static symmetry flag
✅ **Extensible**: Can easily add other symmetry planes (xz, xy) in future
✅ **Cleaner Code**: Symmetry logic explicitly parameterized, not hidden in arrays
