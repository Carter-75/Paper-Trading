#!/usr/bin/env python3
"""
Test Runner Script - Run all unit tests with coverage reporting
Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run only unit tests (skip slow tests)
    python run_tests.py --coverage   # Generate detailed coverage report
"""

import sys
import os
import subprocess
import argparse


def install_dependencies():
    """Check if pytest is installed, install if missing"""
    try:
        import pytest
        import pytest_cov
        print("[OK] Test dependencies are installed")
        return True
    except ImportError:
        print("[!] Installing test dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "pytest>=7.4.0", "pytest-cov>=4.1.0",
                "pytest-timeout>=2.1.0", "pytest-mock>=3.11.1"
            ])
            print("[OK] Test dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("[X] Failed to install test dependencies")
            print("    Please run: pip install -r requirements.txt")
            return False


def run_tests(args):
    """Run pytest with specified options"""
    pytest_args = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
    ]
    
    if args.fast:
        # Skip slow tests
        pytest_args.extend(["-m", "not slow and not integration"])
    
    if args.coverage or args.html:
        # Add coverage options
        pytest_args.extend([
            "--cov=.",
            "--cov-report=term-missing",
        ])
        
        if args.html:
            pytest_args.append("--cov-report=html")
            print("\n[!] HTML coverage report will be generated in htmlcov/")
    
    if args.specific:
        # Run specific test file
        pytest_args.append(args.specific)
    
    if args.verbose:
        pytest_args.append("-vv")
    
    if args.markers:
        # Run tests with specific marker
        pytest_args.extend(["-m", args.markers])
    
    # Run pytest
    print(f"\n{'='*70}")
    print("RUNNING TESTS")
    print(f"{'='*70}\n")
    print(f"Command: pytest {' '.join(pytest_args)}\n")
    
    try:
        import pytest as pytest_module
        exit_code = pytest_module.main(pytest_args)
        
        print(f"\n{'='*70}")
        if exit_code == 0:
            print("[OK] ALL TESTS PASSED!")
        else:
            print("[X] SOME TESTS FAILED")
        print(f"{'='*70}\n")
        
        if args.html:
            print("View HTML coverage report:")
            print(f"  file://{os.path.abspath('htmlcov/index.html')}\n")
        
        return exit_code
    
    except Exception as e:
        print(f"\n[X] Error running tests: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run unit tests for Paper Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --fast             # Run only fast unit tests
  python run_tests.py --coverage --html  # Generate HTML coverage report
  python run_tests.py --specific test_portfolio_manager.py  # Run specific test file
  python run_tests.py --markers unit     # Run only tests marked as 'unit'
        """
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only fast unit tests (skip slow and integration tests)"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate detailed coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (implies --coverage)"
    )
    
    parser.add_argument(
        "--specific",
        type=str,
        help="Run specific test file (e.g., test_portfolio_manager.py)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Extra verbose output"
    )
    
    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Run tests with specific pytest markers (e.g., 'unit', 'integration')"
    )
    
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip dependency installation check"
    )
    
    args = parser.parse_args()
    
    # Check/install dependencies
    if not args.no_install:
        if not install_dependencies():
            return 1
    
    # Run tests
    exit_code = run_tests(args)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

