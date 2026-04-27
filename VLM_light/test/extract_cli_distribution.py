#!/usr/bin/env python
"""
Extract Cli (local lift coefficient) distribution from VSP Aero .lod file
for different angles of attack.
"""

import numpy as np
import matplotlib.pyplot as plt
import re

def parse_lod_file(filename):
    """
    Parse VSP Aero .lod file and extract CLI distributions for each AoA and surface.
    Handles multiple surfaces (identified by VortexSheet).
    
    Returns:
        dict: Nested structure {aoa_value: {surface_id: {'cli': array, 'yavg': array}}}
    """
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by angle of attack markers
    sections = re.split(r'AoA_\s+([0-9.-]+)\s+deg', content)
    
    # Extract data
    aoa_data = {}
    
    # Skip first element (before first AoA_)
    for i in range(1, len(sections), 2):
        aoa_value = float(sections[i])
        data_section = sections[i + 1] if i + 1 < len(sections) else ""
        
        # Extract the data table from this section
        # Look for the line starting with "Iter"
        lines = data_section.split('\n')
        
        # Find header line
        header_idx = None
        for j, line in enumerate(lines):
            if line.strip().startswith('Iter'):
                header_idx = j
                break
        
        if header_idx is None:
            continue
        
        # Parse header
        header_line = lines[header_idx]
        headers = header_line.split()
        
        # Find column indices
        try:
            cli_idx = headers.index('Cli')
        except ValueError:
            print(f"Warning: Cli column not found at AoA = {aoa_value}°")
            continue
        
        try:
            vortex_sheet_idx = headers.index('VortexSheet')
        except ValueError:
            vortex_sheet_idx = None
        
        try:
            yavg_idx = headers.index('Yavg')
        except ValueError:
            yavg_idx = None
        
        # Group data by surface (VortexSheet)
        surface_data = {}
        
        for k in range(header_idx + 1, len(lines)):
            line = lines[k].strip()
            if not line or line.startswith('#'):
                continue  # Skip blank lines and comments, don't break
            
            try:
                values = line.split()
                if len(values) > cli_idx:
                    # Get surface ID from VortexSheet, default to 1 if not present
                    surface_id = int(float(values[vortex_sheet_idx])) if vortex_sheet_idx is not None and len(values) > vortex_sheet_idx else 1
                    
                    # Initialize surface dict if not present
                    if surface_id not in surface_data:
                        surface_data[surface_id] = {'cli': [], 'yavg': []}
                    
                    # Append Cli value
                    surface_data[surface_id]['cli'].append(float(values[cli_idx]))
                    
                    # Append Yavg value if available
                    if yavg_idx is not None and len(values) > yavg_idx:
                        surface_data[surface_id]['yavg'].append(float(values[yavg_idx]))
            except (ValueError, IndexError):
                continue
        
        # Convert lists to numpy arrays and store
        if surface_data:
            aoa_data[aoa_value] = {}
            for surface_id, data in surface_data.items():
                aoa_data[aoa_value][surface_id] = {
                    'cli': np.array(data['cli']),
                    'yavg': np.array(data['yavg']) if data['yavg'] else np.arange(len(data['cli']))
                }
    
    return aoa_data

def print_cli_summary(aoa_data):
    """Print summary of CLI distributions for each surface."""
    print("=" * 80)
    print("CLI Distribution Summary for Different Angles of Attack and Surfaces")
    print("=" * 80)
    
    # Get all unique surfaces
    all_surfaces = set()
    for aoa_dict in aoa_data.values():
        all_surfaces.update(aoa_dict.keys())
    
    for surface_id in sorted(all_surfaces):
        print(f"\n{'=' * 80}")
        print(f"SURFACE {surface_id}")
        print(f"{'=' * 80}")
        
        for aoa in sorted(aoa_data.keys()):
            if surface_id not in aoa_data[aoa]:
                continue
                
            data = aoa_data[aoa][surface_id]
            cli = data['cli']
            
            print(f"\nAngle of Attack: {aoa:6.1f}°")
            print(f"  Number of stations: {len(cli)}")
            print(f"  CLI range:     [{cli.min():10.6f}, {cli.max():10.6f}]")
            print(f"  CLI mean:      {cli.mean():10.6f}")
            print(f"  CLI std dev:   {cli.std():10.6f}")
            print(f"  First 5 values: {cli[:5]}")
            print(f"  Last 5 values:  {cli[-5:]}")

def save_cli_to_csv(aoa_data, output_file_prefix="cli_distribution"):
    """Save CLI distributions to CSV file(s), one per surface."""
    # Get all unique surfaces
    all_surfaces = set()
    for aoa_dict in aoa_data.values():
        all_surfaces.update(aoa_dict.keys())
    
    for surface_id in sorted(all_surfaces):
        # Create filename for this surface
        if len(all_surfaces) > 1:
            output_file = f"{output_file_prefix}_surface_{surface_id}.csv"
        else:
            output_file = f"{output_file_prefix}.csv"
        
        with open(output_file, 'w') as f:
            # Find maximum number of stations for this surface
            max_stations = 0
            for aoa in aoa_data.keys():
                if surface_id in aoa_data[aoa]:
                    max_stations = max(max_stations, len(aoa_data[aoa][surface_id]['cli']))
            
            # Write header
            header = ['Station']
            for aoa in sorted(aoa_data.keys()):
                if surface_id in aoa_data[aoa]:
                    header.append(f'AoA_{aoa:.1f}deg')
            f.write(','.join(header) + '\n')
            
            # Write data
            for i in range(max_stations):
                row = [str(i + 1)]
                for aoa in sorted(aoa_data.keys()):
                    if surface_id in aoa_data[aoa]:
                        cli_array = aoa_data[aoa][surface_id]['cli']
                        if i < len(cli_array):
                            row.append(f"{cli_array[i]:.8f}")
                        else:
                            row.append("")
                f.write(','.join(row) + '\n')
        
        print(f"CLI data saved to: {output_file}")

def plot_cli_distributions(aoa_data, output_file_prefix=None):
    """Plot CLI distributions for all angles of attack and surfaces."""
    # Get all unique surfaces
    all_surfaces = sorted(set(surface_id for aoa_dict in aoa_data.values() for surface_id in aoa_dict.keys()))
    num_surfaces = len(all_surfaces)
    
    if num_surfaces == 0:
        print("No data to plot")
        return
    
    # Create subplots: 2 rows per surface
    fig, axes = plt.subplots(2 * num_surfaces, 2, figsize=(14, 5 * num_surfaces))
    
    # Handle single surface case (axes won't be 2D)
    if num_surfaces == 1:
        axes = axes.reshape(1, -1)
    
    aoa_list = sorted(aoa_data.keys())
    
    for surf_idx, surface_id in enumerate(all_surfaces):
        # Plot 1: All curves on one plot for this surface
        ax = axes[2 * surf_idx, 0]
        for aoa in aoa_list:
            if surface_id not in aoa_data[aoa]:
                continue
            data = aoa_data[aoa][surface_id]
            yavg = data['yavg']
            ax.plot(yavg, data['cli'], marker='o', label=f'{aoa:.1f}°', markersize=3)
        ax.set_xlabel('Spanwise Position (Yavg)')
        ax.set_ylabel('Cli (Local Lift Coefficient)')
        ax.set_title(f'All Angles of Attack - Surface {surface_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Multiple subplots for individual AoA for this surface
        ax = axes[2 * surf_idx, 1]
        aoa_subset = aoa_list[::max(1, len(aoa_list)//4)]  # Every 3-4 angles to avoid crowding
        for aoa in aoa_subset:
            if surface_id not in aoa_data[aoa]:
                continue
            data = aoa_data[aoa][surface_id]
            yavg = data['yavg']
            ax.plot(yavg, data['cli'], marker='s', label=f'{aoa:.1f}°', linewidth=2)
        ax.set_xlabel('Spanwise Position (Yavg)')
        ax.set_ylabel('Cli')
        ax.set_title(f'Selected Angles of Attack - Surface {surface_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: CLI at first and last station vs AoA
        ax = axes[2 * surf_idx + 1, 0]
        root_cli = []
        tip_cli = []
        aoa_valid = []
        for aoa in aoa_list:
            if surface_id in aoa_data[aoa]:
                cli = aoa_data[aoa][surface_id]['cli']
                if len(cli) > 0:
                    root_cli.append(cli[0])
                    tip_cli.append(cli[-1])
                    aoa_valid.append(aoa)
        
        if root_cli:
            ax.plot(aoa_valid, root_cli, 'o-', linewidth=2, markersize=8, label='Station 0')
            ax.plot(aoa_valid, tip_cli, 's-', linewidth=2, markersize=8, label='Station -1')
            ax.set_xlabel('Angle of Attack (deg)')
            ax.set_ylabel('Cli')
            ax.set_title(f'Root and Tip Cli vs AoA - Surface {surface_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Mean CLI vs AoA
        ax = axes[2 * surf_idx + 1, 1]
        mean_cli = []
        max_cli = []
        min_cli = []
        aoa_valid = []
        for aoa in aoa_list:
            if surface_id in aoa_data[aoa]:
                cli = aoa_data[aoa][surface_id]['cli']
                if len(cli) > 0:
                    mean_cli.append(cli.mean())
                    max_cli.append(cli.max())
                    min_cli.append(cli.min())
                    aoa_valid.append(aoa)
        
        if mean_cli:
            ax.plot(aoa_valid, mean_cli, 'o-', linewidth=2, markersize=8, label='Mean')
            ax.fill_between(aoa_valid, min_cli, max_cli, alpha=0.3, label='Min-Max Range')
            ax.set_xlabel('Angle of Attack (deg)')
            ax.set_ylabel('Cli')
            ax.set_title(f'Mean Cli and Range vs AoA - Surface {surface_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('Lift Coefficient (Cli) Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file_prefix:
        if num_surfaces > 1:
            output_file = f"{output_file_prefix}_all_surfaces.png"
        else:
            output_file = f"{output_file_prefix}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    # File path
    lod_file = "ref_results/rectangular_wing_20_40.lod"
    
    print(f"Parsing {lod_file}...")
    aoa_data = parse_lod_file(lod_file)
    
    if not aoa_data:
        print("No data extracted!")
        exit(1)
    
    # Count surfaces and statistics
    all_surfaces = set(surface_id for aoa_dict in aoa_data.values() for surface_id in aoa_dict.keys())
    print(f"Successfully extracted data for {len(aoa_data)} angles of attack")
    print(f"Number of surfaces found: {len(all_surfaces)}")
    print(f"Surface IDs: {sorted(all_surfaces)}")
    
    # Print summary
    print_cli_summary(aoa_data)
    
    # Save to CSV
    save_cli_to_csv(aoa_data, "cli_distribution")
    
    # Create plots
    plot_cli_distributions(aoa_data, "cli_distributions")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
