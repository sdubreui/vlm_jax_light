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
    Parse VSP Aero .lod file and extract CLI distributions for each AoA.
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
        
        # Find the Cli column index
        try:
            cli_idx = headers.index('Cli')
        except ValueError:
            print(f"Warning: Cli column not found at AoA = {aoa_value}°")
            continue
        
        # Find Yavg column for spanwise position
        try:
            yavg_idx = headers.index('Yavg')
        except ValueError:
            yavg_idx = None
        
        # Extract data rows
        cli_values = []
        yavg_values = []
        
        for k in range(header_idx + 1, len(lines)):
            line = lines[k].strip()
            if not line or line.startswith('#'):
                break
            
            try:
                values = line.split()
                if len(values) > cli_idx:
                    cli_values.append(float(values[cli_idx]))
                    if yavg_idx is not None and len(values) > yavg_idx:
                        yavg_values.append(float(values[yavg_idx]))
            except (ValueError, IndexError):
                continue
        
        if cli_values:
            aoa_data[aoa_value] = {
                'cli': np.array(cli_values),
                'yavg': np.array(yavg_values) if yavg_values else np.arange(len(cli_values))
            }
    
    return aoa_data

def print_cli_summary(aoa_data):
    """Print summary of CLI distributions."""
    print("=" * 80)
    print("CLI Distribution Summary for Different Angles of Attack")
    print("=" * 80)
    
    for aoa in sorted(aoa_data.keys()):
        data = aoa_data[aoa]
        cli = data['cli']
        
        print(f"\nAngle of Attack: {aoa:6.1f}°")
        print(f"  Number of stations: {len(cli)}")
        print(f"  CLI range:     [{cli.min():10.6f}, {cli.max():10.6f}]")
        print(f"  CLI mean:      {cli.mean():10.6f}")
        print(f"  CLI std dev:   {cli.std():10.6f}")
        print(f"  First 5 values: {cli[:5]}")
        print(f"  Last 5 values:  {cli[-5:]}")

def save_cli_to_csv(aoa_data, output_file):
    """Save CLI distributions to CSV file."""
    with open(output_file, 'w') as f:
        # Find maximum number of stations
        max_stations = max(len(data['cli']) for data in aoa_data.values())
        
        # Write header
        header = ['Station']
        for aoa in sorted(aoa_data.keys()):
            header.append(f'AoA_{aoa:.1f}deg')
        f.write(','.join(header) + '\n')
        
        # Write data
        for i in range(max_stations):
            row = [str(i + 1)]
            for aoa in sorted(aoa_data.keys()):
                cli_array = aoa_data[aoa]['cli']
                if i < len(cli_array):
                    row.append(f"{cli_array[i]:.8f}")
                else:
                    row.append("")
            f.write(','.join(row) + '\n')
    
    print(f"\nCLI data saved to: {output_file}")

def plot_cli_distributions(aoa_data, output_file=None):
    """Plot CLI distributions for all angles of attack."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Lift Coefficient (Cli) Distribution Along Wing Span', fontsize=14, fontweight='bold')
    
    aoa_list = sorted(aoa_data.keys())
    
    # Plot 1: All curves on one plot
    ax = axes[0, 0]
    for aoa in aoa_list:
        data = aoa_data[aoa]
        spanwise_pos = np.linspace(0, 1, len(data['cli']))
        ax.plot(spanwise_pos, data['cli'], marker='o', label=f'{aoa:.1f}°', markersize=3)
    ax.set_xlabel('Spanwise Position (0=root, 1=tip)')
    ax.set_ylabel('Cli (Local Lift Coefficient)')
    ax.set_title('All Angles of Attack')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Multiple subplots for individual AoA
    ax = axes[0, 1]
    for aoa in aoa_list[::3]:  # Every 3rd angle to avoid crowding
        data = aoa_data[aoa]
        spanwise_pos = np.linspace(0, 1, len(data['cli']))
        ax.plot(spanwise_pos, data['cli'], marker='s', label=f'{aoa:.1f}°', linewidth=2)
    ax.set_xlabel('Spanwise Position')
    ax.set_ylabel('Cli')
    ax.set_title('Selected Angles of Attack')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: CLI at root and tip vs AoA
    ax = axes[1, 0]
    root_cli = [aoa_data[aoa]['cli'][0] for aoa in aoa_list]
    tip_cli = [aoa_data[aoa]['cli'][-1] for aoa in aoa_list]
    ax.plot(aoa_list, root_cli, 'o-', linewidth=2, markersize=8, label='Root')
    ax.plot(aoa_list, tip_cli, 's-', linewidth=2, markersize=8, label='Tip')
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('Cli')
    ax.set_title('Root and Tip Cli vs AoA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean CLI vs AoA
    ax = axes[1, 1]
    mean_cli = [aoa_data[aoa]['cli'].mean() for aoa in aoa_list]
    max_cli = [aoa_data[aoa]['cli'].max() for aoa in aoa_list]
    min_cli = [aoa_data[aoa]['cli'].min() for aoa in aoa_list]
    ax.plot(aoa_list, mean_cli, 'o-', linewidth=2, markersize=8, label='Mean')
    ax.fill_between(aoa_list, min_cli, max_cli, alpha=0.3, label='Min-Max Range')
    ax.set_xlabel('Angle of Attack (deg)')
    ax.set_ylabel('Cli')
    ax.set_title('Mean Cli and Range vs AoA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
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
    
    print(f"Successfully extracted data for {len(aoa_data)} angles of attack")
    
    # Print summary
    print_cli_summary(aoa_data)
    
    # Save to CSV
    save_cli_to_csv(aoa_data, "cli_distribution.csv")
    
    # Create plots
    plot_cli_distributions(aoa_data, "cli_distributions.png")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
