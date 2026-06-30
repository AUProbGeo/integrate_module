#!/usr/bin/env python
"""
INTEGRATE Timing CLI

Command-line interface for timing benchmarks of the INTEGRATE workflow.
This module imports timing functions from the main integrate module.

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

# Configure matplotlib for non-interactive plotting
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Import timing functions from integrate module
try:
    # Try relative import first (when run as module)
    from . import integrate as ig
    from .integrate import timing_compute, timing_plot, allocate_large_page
except ImportError:
    try:
        # Try absolute import (when run directly)
        import integrate as ig
        from integrate import timing_compute, timing_plot, allocate_large_page
    except ImportError:
        print("Error: Could not import integrate module. Please ensure it is properly installed.")
        import sys
        sys.exit(1)


def print_timing_summary(f_timing):
    """Print a concise summary of timing benchmark results."""
    import numpy as np

    try:
        data = np.load(f_timing)
        T_prior = data['T_prior']
        T_forward = data['T_forward']
        T_rejection = data['T_rejection']
        T_poststat = data['T_poststat']
        N_arr = data['N_arr']
        Nproc_arr = data['Nproc_arr']

        try:
            T_total = data['T_total']
        except:
            T_total = T_prior + T_forward + T_rejection + T_poststat
            
        try:
            nobs = data['nobs']
        except:
            nobs = 11693  # Default fallback value

        try:
            backend = str(data['backend'])
        except:
            backend = 'numpy'

        print(f"\n{'='*60}")
        print(f"INTEGRATE Timing Summary: {f_timing}")
        print(f"Rejection backend: {backend}")
        print(f"{'='*60}")

        print(f"\nDataset sizes tested: {N_arr.astype(int)}")
        print(f"CPU counts tested: {Nproc_arr.astype(int)}")

        # Calculate key performance metrics
        T_forward_sounding_per_sec = N_arr[:,np.newaxis]/T_forward
        T_forward_sounding_per_sec_per_cpu = T_forward_sounding_per_sec/Nproc_arr[np.newaxis,:]

        # For rejection sampling, use correct metric: data soundings per second (not lookup table models per second)
        T_rejection_per_data = nobs/T_rejection  # Data soundings per second
        T_rejection_per_data_per_cpu = T_rejection_per_data/Nproc_arr[np.newaxis,:]

        # Find best performance cases (remove NaN values)
        forward_valid = ~np.isnan(T_forward_sounding_per_sec)
        rejection_valid = ~np.isnan(T_rejection_per_data)

        if np.any(forward_valid):
            max_forward_total = np.nanmax(T_forward_sounding_per_sec)
            max_forward_per_cpu = np.nanmax(T_forward_sounding_per_sec_per_cpu)

            print(f"\n🚀 FORWARD MODELING PERFORMANCE:")
            print(f"   Max soundings/sec (all CPUs): {max_forward_total:.1f}")
            print(f"   Max soundings/sec/CPU:        {max_forward_per_cpu:.2f}")

            # Show performance for different CPU counts
            print(f"\n   Forward Performance by CPU Count:")
            for j, ncpu in enumerate(Nproc_arr):
                if np.any(~np.isnan(T_forward_sounding_per_sec[:, j])):
                    best_perf = np.nanmax(T_forward_sounding_per_sec[:, j])
                    best_perf_per_cpu = best_perf / ncpu
                    print(f"     {int(ncpu):2d} CPUs: {best_perf:8.1f} sounds/sec ({best_perf_per_cpu:.2f} per CPU)")

        if np.any(rejection_valid):
            # Focus on largest lookup table (largest N value)
            max_n_idx = np.argmax(N_arr)
            
            # Extract rejection performance for largest lookup table only
            T_rejection_largest = T_rejection_per_data[max_n_idx, :]
            T_rejection_largest_per_cpu = T_rejection_per_data_per_cpu[max_n_idx, :]
            
            # Check if we have valid data for the largest lookup table
            if np.any(~np.isnan(T_rejection_largest)):
                max_rejection_total = np.nanmax(T_rejection_largest)
                max_rejection_per_cpu = np.nanmax(T_rejection_largest_per_cpu)
                
                # Find which CPU count achieved the maximum performance
                max_total_cpu_idx = np.nanargmax(T_rejection_largest)
                max_per_cpu_cpu_idx = np.nanargmax(T_rejection_largest_per_cpu)
                max_total_cpus = int(Nproc_arr[max_total_cpu_idx])
                max_per_cpu_cpus = int(Nproc_arr[max_per_cpu_cpu_idx])

                print(f"\n⚡ REJECTION SAMPLING PERFORMANCE (Largest Lookup Table: {int(N_arr[max_n_idx]):,} models):")
                print(f"   Max data soundings/sec (all CPUs): {max_rejection_total:.1f} (achieved with {max_total_cpus} CPUs)")
                print(f"   Max data soundings/sec/CPU:       {max_rejection_per_cpu:.2f} (achieved with {max_per_cpu_cpus} CPUs)")

                # Show performance for different CPU counts (largest lookup table only)
                print(f"\n   Rejection Performance by CPU Count (Largest Lookup Table):")
                for j, ncpu in enumerate(Nproc_arr):
                    if not np.isnan(T_rejection_largest[j]):
                        best_perf = T_rejection_largest[j]
                        best_perf_per_cpu = best_perf / ncpu
                        print(f"     {int(ncpu):2d} CPUs: {best_perf:8.1f} data sounds/sec ({best_perf_per_cpu:.2f} per CPU)")
                    elif int(ncpu) == 1:
                        # Always show 1 CPU entry even if data is NaN, for reference
                        print(f"     {int(ncpu):2d} CPUs: No valid data")
            else:
                print(f"\n⚡ REJECTION SAMPLING PERFORMANCE: No valid data for largest lookup table ({int(N_arr[max_n_idx]):,} models)")

        # Overall timing breakdown for best case using largest lookup table
        if np.any(~np.isnan(T_total)):
            # Focus on largest lookup table (largest N value)
            max_n_idx = np.argmax(N_arr)
            
            # Extract timing data for largest lookup table only
            T_total_largest = T_total[max_n_idx, :]
            
            # Check if we have valid data for the largest lookup table
            if np.any(~np.isnan(T_total_largest)):
                # Find best CPU count for the largest lookup table
                best_cpu_idx = np.nanargmax(N_arr[max_n_idx]/T_total_largest)
                best_cpu = Nproc_arr[best_cpu_idx]
                best_n = N_arr[max_n_idx]

                t_pri = T_prior[max_n_idx, best_cpu_idx]
                t_fwd = T_forward[max_n_idx, best_cpu_idx]
                t_rej = T_rejection[max_n_idx, best_cpu_idx]
                t_post = T_poststat[max_n_idx, best_cpu_idx]
                t_tot = t_pri + t_fwd + t_rej + t_post

                print(f"\n📊 BEST OVERALL PERFORMANCE (Largest Lookup Table: {int(best_n):,} models):")
                print(f"   Best CPU configuration: {int(best_cpu)} CPUs")
                print(f"   Total throughput: {best_n/t_tot:.1f} models/sec")
                print(f"\n   Time breakdown:")
                print(f"     Prior generation: {t_pri:6.1f}s ({100*t_pri/t_tot:4.1f}%)")
                print(f"     Forward modeling: {t_fwd:6.1f}s ({100*t_fwd/t_tot:4.1f}%)")
                print(f"     Rejection sample: {t_rej:6.1f}s ({100*t_rej/t_tot:4.1f}%)")
                print(f"     Post statistics:  {t_post:6.1f}s ({100*t_post/t_tot:4.1f}%)")
                print(f"     Total time:       {t_tot:6.1f}s")
            else:
                print(f"\n📊 BEST OVERALL PERFORMANCE: No valid data for largest lookup table ({int(N_arr[max_n_idx]):,} models)")

        print(f"\n{'='*60}")

    except Exception as e:
        print(f"Error reading timing summary from {f_timing}: {str(e)}")

# %% The main function
def main():
    """Entry point for the integrate_timing command."""
    import argparse
    import sys
    import os
    import glob
    import psutil
    import numpy as np

    import multiprocessing
    multiprocessing.freeze_support()
    
    # Set a lower limit for processes to avoid handle limit issues on Windows
    import platform
    if platform.system() == 'Windows':
        # On Windows, limit the max processes to avoid handle limit issues
        multiprocessing.set_start_method('spawn')
        
        # Optional - can help with some multiprocessing issues
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='INTEGRATE timing benchmark tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
INTEGRATE Timing Benchmark Tool

This tool benchmarks the performance of the complete INTEGRATE workflow including:
1. Prior model generation (layered geological models)
2. Forward modeling using GA-AEM electromagnetic simulation  
3. Rejection sampling for Bayesian inversion
4. Posterior statistics computation

USAGE EXAMPLES:

Basic benchmarks:
  integrate_timing time small                    # Quick test with default settings
  integrate_timing time medium                   # Balanced benchmark  
  integrate_timing time large                    # Comprehensive benchmark

Custom dataset sizes:
  integrate_timing time small --Nmin 5000        # Test with 5000 models
  integrate_timing time small --N 100000         # Test with exactly 100000 models
  integrate_timing time medium --Nmin 10000      # Medium test starting from 10000 models

Custom CPU configurations:
  integrate_timing time small --Ncpu 16          # Test with exactly 16 CPUs
  integrate_timing time medium --cpu-scale linear # Test all CPU counts [1,2,3,...,64]
  integrate_timing time large --cpu-scale log    # Test log scale [1,2,4,8,16,32,64]

Combined options:
  integrate_timing time small --Ncpu 32 --Nmin 50000    # 50k models on 32 CPUs
  integrate_timing time medium --N 25000 --cpu-scale linear  # 25k models, all CPU counts

Plotting results:
  integrate_timing plot timing_results.npz       # Plot specific results file
  integrate_timing plot --all                    # Plot all .npz files in directory

PARAMETER PRIORITY:
- Dataset sizes: --N (highest) > --Nmin > default
- CPU counts: --Ncpu (highest) > --cpu-scale/--Nmin > default

BENCHMARK SIZES:
- small:  ~1,000 models, quick test
- medium: 1,000-100,000 models, balanced test  
- large:  10,000-1,000,000 models, comprehensive test

Results are saved as .npz files and automatically plotted with performance analysis.
        """
    )
    
    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Plot timing results from benchmark files')
    plot_parser.add_argument('file', nargs='?', default='time', 
                           help='NPZ file containing timing results to plot')
    plot_parser.add_argument('--all', action='store_true',
                           help='Plot all NPZ timing files in the current directory')
    plot_parser.add_argument('--no-summary', action='store_true',
                           help='Disable timing summary output (enabled by default)')
    plot_parser.add_argument('--fontsize', type=int, default=16,
                           help='Base font size for plot text (default: 16)')
    
    # Time command
    time_parser = subparsers.add_parser('time', help='Run performance benchmark of INTEGRATE workflow')
    time_parser.add_argument('size', choices=['small', 'medium', 'large'], 
                            default='medium', nargs='?', 
                            help='Benchmark size: small (~1k models, quick), medium (1k-100k models), large (10k-1M models)')
    time_parser.add_argument('--cpu-scale', choices=['linear', 'log'], 
                            default='log', 
                            help='CPU scaling method: linear tests [1,2,3,...,Ncpu], log tests [1,2,4,8,...,Ncpu] (default: log)')
    time_parser.add_argument('--Nmin', type=int, default=0,
                            help='Dataset size control: For small benchmark, use exactly this many models. For medium/large, use as starting point in range (default: use benchmark defaults)')
    time_parser.add_argument('--Ncpu', type=int, default=0,
                            help='Use exactly this many CPU cores, overriding all other CPU options (default: 0, use scaling)')
    time_parser.add_argument('--N', type=int, default=0,
                            help='Use exactly this dataset size (number of models), overriding size and Nmin options (default: 0, use size-based defaults)')
    time_parser.add_argument('--no-summary', action='store_true',
                            help='Disable timing summary output (enabled by default)')
    time_parser.add_argument('--backend', choices=['numpy', 'jax'], default='numpy',
                            help='Rejection sampling backend: numpy (default) or jax')
    time_parser.add_argument('--NcpuForward', type=int, default=0,
                            help='Fix the number of CPUs used for forward modeling only. '
                                 'When set, forward modeling always uses this many CPUs while '
                                 'inversion (rejection sampling) varies over the usual CPU range. '
                                 '(default: 0, forward uses same count as inversion)')
    
    # Add special case handling for '-time' without size argument
    if '-time' in sys.argv and len(sys.argv) == 2:
        print("Please specify a size for the timing benchmark:")
        print("  small  - Quick test with minimal resources")
        print("  medium - Balanced benchmark (default)")
        print("  large  - Comprehensive benchmark (may take hours)")
        print("\nExample: integrate_timing -time medium")
        sys.exit(0)
        
    # Parse arguments
    args = parser.parse_args()
    
    # Set default command if none is provided
    if args.command is None:
        # Show help when no command is specified
        parser.print_help()
        sys.exit(0)
   
    # Execute command
    if args.command == 'plot':
        if args.all:
            # Plot all NPZ files in the current directory
            files = glob.glob('*.npz')
            for f in files:
                try:
                    # Show summary by default
                    if not args.no_summary:
                        print_timing_summary(f)

                    timing_plot(f, fontsize=args.fontsize)
                    plt.close('all')  # Close all figures after plotting
                    print(f"Successfully plotted: {f}")
                except Exception as e:
                    print(f"Error plotting {f}: {str(e)}")
                finally:
                    plt.close('all')  # Ensure figures are closed even on error
        elif args.file:
            # Plot specified file
            if not os.path.exists(args.file):
                print(f"File not found: {args.file}")
                sys.exit(1)
            try:
                # Show summary by default
                if not args.no_summary:
                    print_timing_summary(args.file)

                timing_plot(args.file, fontsize=args.fontsize)
                plt.close('all')  # Close all figures after plotting
                print(f"Successfully plotted: {args.file}")
            except Exception as e:
                print(f"Error plotting {args.file}: {str(e)}")
            finally:
                plt.close('all')  # Ensure figures are closed even on error
        else:
            print("Please specify a file to plot or use --all")
    
    elif args.command == 'time':
        Ncpu = psutil.cpu_count(logical=False)
        
        # Handle Ncpu option for processors
        if args.Ncpu > 0:
            # Use only the specified number of CPUs
            Nproc_arr = np.array([args.Ncpu])
        else:
            # Determine CPU scaling based on command line option
            if args.cpu_scale == 'linear':
                Nproc_arr = np.arange(1, Ncpu+1)
            else:  # log scaling
                k = int(np.floor(np.log2(Ncpu)))
                Nproc_arr = 2**np.linspace(0,k,(k)+1)
                Nproc_arr = np.append(Nproc_arr, Ncpu)
                Nproc_arr = np.unique(Nproc_arr)

        # Handle dataset sizes
        if args.N > 0:
            # Use only the specified dataset size
            N_arr = np.array([args.N])
        elif args.Nmin > 0 and args.size == 'small':
            # For small benchmark with Nmin: use only that value for dataset size
            N_arr = np.array([args.Nmin])
        elif args.size == 'small':
            # Small benchmark default
            N_arr = np.array([1000])
        elif args.size == 'medium':
            # Medium benchmark
            if args.Nmin > 0:
                # Use Nmin as starting point for medium benchmark
                N_arr = np.ceil(np.logspace(np.log10(args.Nmin), 5, 9))
            else:
                N_arr = np.ceil(np.logspace(3,5,9))
        elif args.size == 'large':
            # Large benchmark
            if args.Nmin > 0:
                # Use Nmin as starting point for large benchmark
                N_arr = np.ceil(np.logspace(np.log10(args.Nmin), 6, 7))
            else:
                N_arr = np.ceil(np.logspace(3,6,7))

        # JAX doesn't use multiprocessing — default to a single "process" entry
        # but respect an explicit --Ncpu value if the user provided one
        if args.backend == 'jax' and args.Ncpu == 0:
            Nproc_arr = np.array([1])

        f_timing = timing_compute(
            N_arr=N_arr,
            Nproc_arr=Nproc_arr,
            backend=args.backend,
            NcpuForward=args.NcpuForward,
        )

        # Show summary and plot the results
        try:
            # Show summary by default
            if not args.no_summary:
                print_timing_summary(f_timing)

            timing_plot(f_timing, fontsize=getattr(args, 'fontsize', 16))
            plt.close('all')  # Close all figures after plotting
            print(f"Timing plots saved successfully.")
        except Exception as e:
            print(f"Error generating timing plots: {str(e)}")
        finally:
            plt.close('all')  # Ensure figures are closed even on error

if __name__ == '__main__':
    main()