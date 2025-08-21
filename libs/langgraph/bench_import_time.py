#!/usr/bin/env python3
"""
Benchmark script to measure LangGraph import time.

This script measures the time it takes to import the langgraph module
and can be used to compare performance before and after changes.
"""

import time
import importlib
import sys

def benchmark_import():
    """Benchmark the import time of langgraph module."""
    print("Benchmarking LangGraph import time...")
    
    # Clear any existing imports
    if "langgraph" in sys.modules:
        del sys.modules["langgraph"]
    
    # Measure import time
    start_time = time.perf_counter()
    import langgraph
    end_time = time.perf_counter()
    
    import_time = end_time - start_time
    
    print(f"Import time: {import_time:.6f} seconds")
    print(f"Version: {langgraph.__version__}")
    
    return import_time

def benchmark_importlib_import():
    """Benchmark using importlib.import_module for comparison."""
    print("\nBenchmarking with importlib.import_module...")
    
    # Clear any existing imports
    if "langgraph" in sys.modules:
        del sys.modules["langgraph"]
    
    # Measure import time
    start_time = time.perf_counter()
    langgraph_module = importlib.import_module("langgraph")
    end_time = time.perf_counter()
    
    import_time = end_time - start_time
    
    print(f"Import time: {import_time:.6f} seconds")
    print(f"Version: {langgraph_module.__version__}")
    
    return import_time

if __name__ == "__main__":
    print("=" * 50)
    print("LangGraph Import Time Benchmark")
    print("=" * 50)
    
    # Run benchmarks
    direct_import_time = benchmark_import()
    importlib_import_time = benchmark_importlib_import()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Direct import:     {direct_import_time:.6f}s")
    print(f"Importlib import:  {importlib_import_time:.6f}s")
    print("=" * 50)
