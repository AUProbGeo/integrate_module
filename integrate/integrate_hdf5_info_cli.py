#!/usr/bin/env python
"""
INTEGRATE HDF5 Info CLI

Command-line interface for analyzing HDF5 files used in INTEGRATE.
Provides fast inspection of DATA, PRIOR, POST, and FORWARD files.

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import argparse
import sys
import os

# Import the integrate module
try:
    import integrate as ig
except ImportError:
    print("Error: Could not import integrate module. Please ensure it is properly installed.")
    sys.exit(1)

def main():
    """Entry point for the hdf5_info command."""

    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Analyze HDF5 files used in the INTEGRATE module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hdf5_info DATA.h5
  hdf5_info --load-data PRIOR.h5
  hdf5_info -q *.h5
  hdf5_info --load-data PRIOR.h5 POST.h5

File Types:
  DATA.h5    - Observed data and geometry
  PRIOR.h5   - Prior model realizations and forward data
  POST.h5    - Posterior results and statistics
  FORWARD.h5 - Forward model configuration

Performance:
  By default, only metadata is read (fast).
  Use --load-data to include data ranges (slower).

For more information, see the INTEGRATE documentation.
        """
    )

    # Required arguments (unless --version is used)
    parser.add_argument('files',
                       nargs='*',
                       help='HDF5 file(s) to analyze')

    # Optional arguments
    parser.add_argument('--load-data', '-d',
                       action='store_true',
                       help='Load actual data to compute ranges and statistics (slower)')

    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Only show summary information (no detailed output)')

    parser.add_argument('--version', '-v',
                       action='store_true',
                       help='Show version information')

    # Parse arguments
    args = parser.parse_args()

    # Handle version request
    if args.version:
        try:
            from integrate import __version__
            print(f"INTEGRATE version: {__version__}")
        except (ImportError, AttributeError):
            print("INTEGRATE version: unknown")
        return 0

    # Validate that files are provided
    if not args.files:
        parser.error("the following arguments are required: files")
        return 1

    # Track success
    all_success = True

    # Process each file
    for i, file_path in enumerate(args.files):
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            all_success = False
            continue

        # Add separator between multiple files
        if i > 0 and not args.quiet:
            print("\n" + "=" * 80)
            print()

        try:
            # Analyze the file
            info = ig.hdf5_info(file_path, verbose=not args.quiet, load_data=args.load_data)

            # Print summary if in quiet mode
            if args.quiet and info:
                print(f"{os.path.basename(file_path):50s} | "
                      f"Type: {info['file_type']:8s} | "
                      f"Datasets: {len(info['datasets']):3d} | "
                      f"Size: {os.path.getsize(file_path)/(1024**2):6.2f} MB")

        except Exception as e:
            print(f"ERROR analyzing file {file_path}: {str(e)}")
            all_success = False
            continue

    # Return appropriate exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
