# -*- coding: utf-8 -*-
"""
Main Entry Point
Food Waste Reduction Platform - Simulation Harness

Supports both CSV data loading and synthetic data generation.
"""

from simulation import compare_strategies
import os

if __name__ == "__main__":
    print("="*70)
    print("Comparing Baseline (Greedy) vs my Ranking Algorithm")
    print("Using Multinomial Logit Model for Customer Choice")
    print("="*70)
    print()
    
    import argparse
    from restaurant_api import load_store_data
    from customer_api import generate_customer
    from data_loader import save_stores_to_csv, save_customers_to_csv
    import numpy as np

    parser = argparse.ArgumentParser(description="Food Waste Simulation")
    parser.add_argument("--data", choices=["default", "generate", "use_generated"], default="default",
                        help="Data source: 'default' (example_*.csv), 'generate' (make new), 'use_generated' (generated_*.csv)")
    args = parser.parse_args()

    # Determine files to use based on args
    stores_file = "example_stores.csv"
    customers_file = "example_customers.csv"
    
    should_generate = False
    
    if args.data == "generate":
        print("Mode: Generate new data -> Run Simulation")
        stores_file = "generated_stores.csv"
        customers_file = "generated_customers.csv"
        should_generate = True
    elif args.data == "use_generated":
        print("Mode: Use previously generated data")
        stores_file = "generated_stores.csv"
        customers_file = "generated_customers.csv"
    else:
        print("Mode: Default (use existing example files if present)")

    # Data Generation Logic
    if should_generate:
        print(f"Generating synthetic data...")
        # 1. Generate Stores
        # seed=None ensures random generation each run
        stores = load_store_data(num_stores=15, num_customers=70, seed=None)
        save_stores_to_csv(stores, stores_file)
        
        # 2. Generate Customers
        # Generate random arrival times
        arrival_times = sorted(np.random.uniform(0, 24.0, 70))
        customers = generate_customer(70, arrival_times, seed=None)
        save_customers_to_csv(customers, stores, customers_file)
        print(f"Data generation complete.")
        print()

    # Check existence of selected files
    has_csv_data = os.path.exists(stores_file) and os.path.exists(customers_file)
    
    if has_csv_data:
        print(f"Found CSV files ({stores_file} and {customers_file})")
        print("  Running 10-day comparison with CSV data...")
        print()
        
        # Run comparison with CSV data
        results = compare_strategies(
            stores_csv=stores_file,
            customers_csv=customers_file,
            n=5,  # Number of stores to display to each customer
            verbose=True,
            seed=None,
            output_dir="simulation_results"
        )
    else:
        print(f"CSV files not found: {stores_file}, {customers_file}")
        print("Running 10-day comparison with purely synthetic data (no file loading).")
        print()
        
        # Run comparison with synthetic data
        results = compare_strategies(
            num_stores=15,
            num_customers=70,
            n=7,
            verbose=True,
            seed=None,
            output_dir="simulation_results"
        )
