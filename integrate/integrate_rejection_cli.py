#!/usr/bin/env python
"""
INTEGRATE Rejection Sampling CLI

Command-line interface for probabilistic inversion using rejection sampling.
Provides access to the integrate_rejection function with various options for
Bayesian inversion and posterior sampling.

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import argparse
import sys
import os
import multiprocessing

# Import the integrate module
try:
    import integrate as ig
except ImportError:
    print("Error: Could not import integrate module. Please ensure it is properly installed.")
    sys.exit(1)

def main():
    """Entry point for the integrate_rejection command."""
    
    # Set up multiprocessing support
    multiprocessing.freeze_support()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='INTEGRATE rejection sampling for Bayesian inversion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  integrate_rejection --prior prior.h5 --data data.h5 --output post.h5
  integrate_rejection --prior prior.h5 --data data.h5 --samples 1000000 --parallel
  integrate_rejection --prior prior.h5 --data data.h5 --auto-temp --cpus 4

For more information, see the INTEGRATE documentation.
        """
    )
    
    # Required arguments
    parser.add_argument('--prior', '-p', 
                       type=str, 
                       required=True,
                       help='Path to HDF5 file containing prior model and data samples')
    
    parser.add_argument('--data', '-d', 
                       type=str, 
                       required=True,
                       help='Path to HDF5 file containing observed data for inversion')
    
    # Optional arguments
    parser.add_argument('--output', '-o', 
                       type=str, 
                       default='',
                       help='Output path for posterior samples (auto-generated if not specified)')
    
    parser.add_argument('--samples', '-n', 
                       type=int, 
                       default=100000000,
                       help='Maximum number of prior samples to use for inversion (default: 100000000)')
    
    parser.add_argument('--auto-temp', '-T', 
                       action='store_true',
                       help='Enable automatic temperature estimation (default: disabled)')
    
    parser.add_argument('--temp-base', 
                       type=float, 
                       default=1.0,
                       help='Base temperature for sampling (default: 1.0)')
    
    parser.add_argument('--nr', 
                       type=int, 
                       default=400,
                       help='Number of resamples for temperature estimation (default: 400)')
    
    parser.add_argument('--cpus', '-c', 
                       type=int, 
                       default=0,
                       help='Number of CPU cores to use (0 = auto-detect, default: 0)')
    
    parser.add_argument('--no-parallel', 
                       action='store_true',
                       help='Disable parallel processing')
    
    parser.add_argument('--chunks', 
                       type=int, 
                       default=0,
                       help='Number of chunks for processing (0 = auto, default: 0)')
    
    parser.add_argument('--id-use', 
                       type=str, 
                       default='',
                       help='Comma-separated list of data IDs to use for inversion')
    
    parser.add_argument('--ip-range', 
                       type=str, 
                       default='',
                       help='Comma-separated IP range for distributed processing')
    
    parser.add_argument('--use-n-best',
                       type=int,
                       default=0,
                       help='Use N best samples for analysis (default: 0)')

    parser.add_argument('--backend',
                       choices=['numpy', 'jax'],
                       default='numpy',
                       help='Rejection sampling backend: numpy (default) or jax')

    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--version', 
                       action='store_true',
                       help='Show version information')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle version request
    if args.version:
        try:
            from integrate import __version__
            print(f"INTEGRATE version: {__version__}")
        except ImportError:
            print("INTEGRATE version: unknown")
        return 0
    
    # Validate input files
    if not os.path.exists(args.prior):
        print(f"Error: Prior file not found: {args.prior}")
        return 1
        
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    # Parse comma-separated arguments
    id_use = []
    if args.id_use:
        try:
            id_use = [int(x.strip()) for x in args.id_use.split(',')]
        except ValueError:
            print(f"Error: Invalid ID list format: {args.id_use}")
            return 1
    
    ip_range = []
    if args.ip_range:
        ip_range = [x.strip() for x in args.ip_range.split(',')]
    
    # Set up parallel processing
    parallel = not args.no_parallel
    if parallel:
        # Check if parallel processing is supported
        parallel = ig.use_parallel(showInfo=1 if args.verbose else 0)
    
    # Print configuration if verbose
    if args.verbose:
        print("Configuration:")
        print(f"  Prior file: {args.prior}")
        print(f"  Data file: {args.data}")
        print(f"  Output file: {args.output if args.output else 'auto-generated'}")
        print(f"  Max samples: {args.samples}")
        print(f"  Auto temperature: {args.auto_temp}")
        print(f"  Base temperature: {args.temp_base}")
        print(f"  Parallel processing: {parallel}")
        print(f"  CPU cores: {args.cpus if args.cpus > 0 else 'auto-detect'}")
        print(f"  Backend: {args.backend}")
        print("")
    
    try:
        # Call the integrate_rejection function
        f_post_h5 = ig.integrate_rejection(
            f_prior_h5=args.prior,
            f_data_h5=args.data,
            f_post_h5=args.output,
            N_use=args.samples,
            id_use=id_use,
            ip_range=ip_range,
            nr=args.nr,
            autoT=1 if args.auto_temp else 0,
            T_base=args.temp_base,
            Nchunks=args.chunks,
            Ncpu=args.cpus,
            parallel=parallel,
            use_N_best=args.use_n_best,
            backend=args.backend,
            showInfo=1 if args.verbose else 0
        )
        
        print(f"Rejection sampling completed successfully.")
        print(f"Posterior samples saved to: {f_post_h5}")
        
        return 0
        
    except Exception as e:
        print(f"Error during rejection sampling: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
