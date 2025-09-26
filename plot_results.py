#!/usr/bin/env python3
"""
Parse and summarize FID and IS results from the results directory.

This script parses result JSON files and creates a summary table showing FID and IS scores
vs number of inference steps for different configurations.
"""

import os
import json
import re
from collections import defaultdict
import argparse

# Try to import matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not available. Install with 'pip install matplotlib' for plotting functionality.")


def parse_filename(filename):
    """
    Parse result filename to extract configuration parameters.
    
    Expected format: randar_0.3b_360k_llamagen_{steps}-randar_0.3b_llamagen_360k_bs_1024_lr_0.0004-size-256-size-256-search.json
    
    Args:
        filename: Result filename
        
    Returns:
        dict: Configuration parameters
    """
    # Remove .json extension if present
    basename = filename.replace('.json', '').replace('-search', '')
    
    # Debug print (only if verbose)
    verbose = False  # Will be set by caller if needed
    
    # Extract inference steps from the beginning of the filename
    # Pattern: randar_{size}_360k_llamagen_{steps}-...
    # Note: \w+ doesn't match dots, so use [\w\.]+ for model size like "0.3b"
    steps_match = re.match(r'randar_[\w\.]+_\w+_llamagen_(\d+)', basename)
    if not steps_match:
        return None
    
    num_inference_steps = int(steps_match.group(1))
    
    # For the experiment name, just use the part before the first dash
    # This handles cases like: randar_0.3b_360k_llamagen_64-randar_0.3b_llamagen_360k_bs_1024_lr_0.0004-size-256-size-256
    parts = basename.split('-', 1)
    exp_name = parts[0]
    
    # Extract base model name without inference steps for grouping
    base_match = re.match(r'(randar_[\w\.]+_\w+_llamagen)_\d+', exp_name)
    if base_match:
        base_config = base_match.group(1)
    else:
        base_config = exp_name
    
    config = {
        'exp_name': exp_name,
        'base_config': base_config,
        'num_inference_steps': num_inference_steps,
        'filename': filename
    }
    
    # Extract image sizes if present
    if len(parts) > 1:
        rest = parts[1]
        size_match = re.search(r'size-(\d+)-size-(\d+)', rest)
        if size_match:
            config['image_size'] = int(size_match.group(1))
            config['image_size_eval'] = int(size_match.group(2))
    
    return config


def load_results(results_dir):
    """
    Load all result files from the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        dict: Results organized by configuration
    """
    results = defaultdict(list)
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist!")
        return results
    
    # Find all result JSON files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('-search.json')]
    
    print(f"Found {len(result_files)} result files")
    
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        
        # Parse filename to extract configuration
        config = parse_filename(filename)
        if not config:
            print(f"Warning: Could not parse filename {filename}")
            continue
        
        # Load JSON data
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract the final report results (with "-report" suffix)
            report_key = None
            for key in data.keys():
                if key.endswith('-report'):
                    report_key = key
                    break
            
            if report_key:
                metrics = data[report_key]
                config.update(metrics)
                
                # Use the base_config from parsing for grouping
                base_config = config['base_config']
                results[base_config].append(config)
                
                print(f"Loaded: {filename} -> {config['num_inference_steps']} steps, FID: {metrics['fid']:.2f}, IS: {metrics['IS']:.2f}")
            else:
                print(f"Warning: No report data found in {filename}")
                
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return results


def print_summary_table(results):
    """
    Print a summary table of all results.
    
    Args:
        results: Results dictionary from load_results()
    """
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    for config_name, config_results in results.items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 70)
        print(f"{'Steps':<8} {'FID':<10} {'IS':<10} {'SFID':<10} {'Precision':<12} {'Recall':<8}")
        print("-" * 70)
        
        # Sort by inference steps
        config_results.sort(key=lambda x: x['num_inference_steps'])
        
        for r in config_results:
            print(f"{r['num_inference_steps']:<8} {r['fid']:<10.2f} {r['IS']:<10.2f} "
                  f"{r['sfid']:<10.2f} {r['precision']:<12.3f} {r['recall']:<8.3f}")


def print_best_results(results):
    """
    Print the best results for each configuration.
    
    Args:
        results: Results dictionary from load_results()
    """
    print("\n" + "="*80)
    print("BEST RESULTS SUMMARY")
    print("="*80)
    
    for config_name, config_results in results.items():
        if not config_results:
            continue
        
        # Find best FID and best IS
        best_fid = min(config_results, key=lambda x: x['fid'])
        best_is = max(config_results, key=lambda x: x['IS'])
        
        print(f"\nConfiguration: {config_name}")
        print("-" * 50)
        print(f"Best FID: {best_fid['fid']:.2f} (at {best_fid['num_inference_steps']} steps)")
        print(f"Best IS:  {best_is['IS']:.2f} (at {best_is['num_inference_steps']} steps)")


def create_plots(results, output_dir=None):
    """
    Create plots for FID and IS vs inference steps.
    
    Args:
        results: Results dictionary from load_results()
        output_dir: Optional directory to save plots
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create plots: matplotlib not available")
        return
        
    if not results:
        print("No results to plot!")
        return
    
    # Set up larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 16,           # Base font size
        'axes.titlesize': 20,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 16,     # X-axis tick label size
        'ytick.labelsize': 16,     # Y-axis tick label size
        'legend.fontsize': 16,     # Legend font size
        'lines.linewidth': 3,      # Line width
        'lines.markersize': 10     # Marker size
    })
    
    # Create a figure with subplots
    fig, (ax_fid, ax_is) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Generate colors for different configurations
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (config_name, config_results), color in zip(results.items(), colors):
        # Sort by inference steps
        config_results.sort(key=lambda x: x['num_inference_steps'])
        
        # Extract data
        steps = [r['num_inference_steps'] for r in config_results]
        fids = [r['fid'] for r in config_results]
        is_scores = [r['IS'] for r in config_results]
        
        # Plot FID (linewidth and markersize now set globally)
        ax_fid.plot(steps, fids, 'o-', label=config_name, color=color)
        
        # Plot IS (linewidth and markersize now set globally)
        ax_is.plot(steps, is_scores, 'o-', label=config_name, color=color)
    
    # Configure FID plot
    ax_fid.set_xlabel('Number of Inference Steps')
    ax_fid.set_ylabel('FID Score (lower is better)')
    ax_fid.set_title('FID vs Inference Steps')
    ax_fid.grid(True, alpha=0.3, linewidth=1)
    ax_fid.set_xscale('log')
    ax_fid.legend(frameon=True, fancybox=True, shadow=True)
    
    # Configure IS plot
    ax_is.set_xlabel('Number of Inference Steps')
    ax_is.set_ylabel('IS Score (higher is better)')
    ax_is.set_title('IS vs Inference Steps')
    ax_is.grid(True, alpha=0.3, linewidth=1)
    ax_is.set_xscale('log')
    ax_is.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout(pad=2.0)  # Add more padding between subplots
    
    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'fid_is_vs_inference_steps.png')
    else:
        output_path = 'fid_is_vs_inference_steps.png'
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def create_csv_output(results, output_file):
    """
    Create CSV output with all results.
    
    Args:
        results: Results dictionary from load_results()
        output_file: Path to output CSV file
    """
    import csv
    
    print(f"\nCreating CSV output: {output_file}")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['configuration', 'num_inference_steps', 'fid', 'IS', 'sfid', 'precision', 'recall']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for config_name, config_results in results.items():
            for r in config_results:
                writer.writerow({
                    'configuration': config_name,
                    'num_inference_steps': r['num_inference_steps'],
                    'fid': r['fid'],
                    'IS': r['IS'],
                    'sfid': r['sfid'],
                    'precision': r['precision'],
                    'recall': r['recall']
                })
    
    print(f"CSV file saved to: {output_file}")


def analyze_trends(results):
    """
    Analyze trends in the results.
    
    Args:
        results: Results dictionary from load_results()
    """
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80)
    
    for config_name, config_results in results.items():
        if len(config_results) < 2:
            continue
        
        print(f"\nConfiguration: {config_name}")
        print("-" * 50)
        
        # Sort by inference steps
        config_results.sort(key=lambda x: x['num_inference_steps'])
        
        # Analyze FID trend
        steps = [r['num_inference_steps'] for r in config_results]
        fids = [r['fid'] for r in config_results]
        is_scores = [r['IS'] for r in config_results]
        
        # Find the step with best performance
        min_fid_idx = fids.index(min(fids))
        max_is_idx = is_scores.index(max(is_scores))
        
        print(f"Steps tested: {steps}")
        print(f"FID trend: {fids[0]:.2f} -> {fids[-1]:.2f} (best: {min(fids):.2f} at {steps[min_fid_idx]} steps)")
        print(f"IS trend:  {is_scores[0]:.2f} -> {is_scores[-1]:.2f} (best: {max(is_scores):.2f} at {steps[max_is_idx]} steps)")
        
        # Check if performance improves with more steps
        if fids[-1] < fids[0]:
            print("✓ FID improves with more inference steps")
        else:
            print("✗ FID does not consistently improve with more steps")
        
        if is_scores[-1] > is_scores[0]:
            print("✓ IS improves with more inference steps")
        else:
            print("✗ IS does not consistently improve with more steps")


def main():
    parser = argparse.ArgumentParser(description='Analyze FID and IS results from evaluation runs')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing result JSON files')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='Path to save CSV output file')
    parser.add_argument('--plot', action='store_true',
                       help='Create plots (requires matplotlib)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots and other outputs')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose parsing information')
    
    args = parser.parse_args()
    
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    
    if not results:
        print("No valid results found!")
        return
    
    print(f"\nFound {len(results)} configuration(s):")
    for config_name, config_results in results.items():
        print(f"  - {config_name}: {len(config_results)} inference step configurations")
    
    # Print summary table
    print_summary_table(results)
    
    # Print best results
    print_best_results(results)
    
    # Analyze trends
    analyze_trends(results)
    
    # Create CSV output if requested
    if args.csv_output:
        create_csv_output(results, args.csv_output)
    
    # Create plots if requested
    if args.plot:
        print("\nCreating plots...")
        create_plots(results, args.output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
