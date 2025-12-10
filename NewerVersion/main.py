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
    
    # Check if CSV files exist
    has_csv_data = os.path.exists("stores.csv") and os.path.exists("customers.csv")
    
    if has_csv_data:
        print("Found CSV files (stores.csv and customers.csv)")
        print("  Running 10-day comparison with CSV data...")
        print()
        
        # Run comparison with CSV data (seed=None means use time-based seed for variation)
        results = compare_strategies(
            stores_csv="stores.csv",
            customers_csv="customers.csv",
            n=5,  # Number of stores to display to each customer
            verbose=True,
            seed=None,  # None = use time-based seed (different results each run)
            output_dir="simulation_results"
        )
    else:
        print("No CSV files found. Running 10-day comparison with synthetic data.")
        print()
        
        # Run comparison with synthetic data (seed=None means use time-based seed for variation)
        results = compare_strategies(
            num_stores=15,
            num_customers=70,
            n=7,  # Number of stores to display to each customer
            verbose=True,
            seed=None,  # None = use time-based seed (different results each run)
            output_dir="simulation_results"
        )
