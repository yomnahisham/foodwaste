# -*- coding: utf-8 -*-
"""
Simulation Module
Handles marketplace simulation, scenario configuration, and simulation harness.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

from restaurant_api import (
    Restaurant, load_store_data, get_all_stores, initialize_day,
    end_of_day_processing, update_reservation, update_exposure
)
from customer_api import (
    Customer, generate_customer, customer_arrives, display_stores_to_customer,
    customer_makes_decision
)
from ranking_algorithm import select_stores, RankingStrategy
import ranking_algorithm
from data_loader import load_data_from_csv, load_stores_from_csv, load_customers_from_csv


class Marketplace:
    """Marketplace structure containing stores, customers, and state"""
    
    def __init__(self, stores: List[Restaurant]):
        self.stores = stores
        self.customers: List[Customer] = []
        self.current_time = 0.0
        self.total_revenue = 0.0
        self.total_cancellations = 0
        self.total_waste = 0
        self.total_customers_seen = 0
        self.n = None  # number of stores to show (constant for the day)


def calculate_n(num_stores: int, expected_customers: int, total_estimated_inventory: int = None) -> int:
    """Calculate n (number of stores to show) based on demand"""
    min_n = 1
    max_n = num_stores

    if num_stores > 0:
        customers_per_store = expected_customers / num_stores
        if customers_per_store > 10:  # high demand
            n = min(max_n, max(min_n, int(num_stores * 0.8)))  # show 80% of stores
        elif customers_per_store > 5:  # medium demand
            n = min(max_n, max(min_n, int(num_stores * 0.6)))  # show 60% of stores
        else:  # low demand
            n = min(max_n, max(min_n, int(num_stores * 0.4)))  # show 40% of stores

        if total_estimated_inventory is not None:
            avg_inventory_per_store = total_estimated_inventory / num_stores if num_stores > 0 else 0
            if avg_inventory_per_store > 30:  # high inventory
                n = min(max_n, n + 1)
            elif avg_inventory_per_store < 10:  # low inventory
                n = max(min_n, n - 1)
    else:
        n = min_n

    return max(min_n, min(max_n, n))


def initialize_marketplace(num_stores: int = 10, actual_inventories: Optional[Dict[int, int]] = None,
                           expected_customers: int = 100, seed: int = 42,
                           stores_csv: Optional[str] = None) -> Marketplace:
    """
    Initialize marketplace with stores and calculate n for the day.
    
    If stores_csv is provided and file exists, loads from CSV.
    Otherwise, generates synthetic data.
    """
    # Try to load from CSV if file exists
    if stores_csv and os.path.exists(stores_csv):
        stores = load_stores_from_csv(stores_csv)
        num_stores = len(stores)
    else:
        stores = load_store_data(num_stores, num_customers=expected_customers, seed=seed)
    
    initialize_day(stores, actual_inventories)

    marketplace = Marketplace(stores)

    total_est_inventory = sum(store.est_inventory for store in stores)
    marketplace.n = calculate_n(num_stores, expected_customers, total_est_inventory)

    return marketplace


def simulate_customer_arrival(marketplace: Marketplace, customer: Customer) -> Dict:
    """
    Simulate a single customer arrival and decision using proper MNL model.
    
    First, customer decides if they will even open the app (arrival probability).
    Then, if they arrive, they see stores and make a choice using MNL.
    """
    # Stage 1: Customer decides if they will open the app today
    # Realistic behavior: Not all customers open the app every day
    # Base arrival probability (can be adjusted based on customer characteristics)
    arrival_probability = 0.7  # 70% chance customer opens app on a given day
    
    # Adjust based on customer satisfaction (satisfied customers more likely to return)
    arrival_probability *= (0.5 + customer.satisfaction_level)
    
    arrival_probability = min(1.0, arrival_probability)
    
    # Customer decides if they open the app
    if np.random.uniform() > arrival_probability:
        # Customer doesn't open app today
        return {
            'action': 'no_arrival',  # New action: customer didn't even arrive
            'store_id': None
        }
    
    # Customer opens app - proceed with decision
    customer_arrives(customer)
    marketplace.current_time = customer.arrival_time
    marketplace.total_customers_seen += 1

    all_stores = get_all_stores()

    # Select n stores to display using ranking algorithm
    t = marketplace.total_customers_seen
    n = marketplace.n
    displayed_stores = ranking_algorithm.select_stores(customer, n, all_stores, t)

    # Display stores to customer
    display_stores_to_customer(customer, displayed_stores)

    # Update exposure for displayed stores
    for store in displayed_stores:
        update_exposure(store.restaurant_id)

    # Customer makes decision using proper MNL model
    # Note: current_hour parameter is kept for API compatibility but not used (food waste collected at 10 PM)
    decision = customer_makes_decision(customer, displayed_stores, customer.arrival_time)

    # Update reservations if customer bought
    if decision['action'] == 'buy':
        update_reservation(decision['store_id'], 1)
        store = next((s for s in all_stores if s.restaurant_id == decision['store_id']), None)
        if store:
            marketplace.total_revenue += store.price

    # Add customer to marketplace
    marketplace.customers.append(customer)

    return decision


def process_end_of_day(marketplace: Marketplace) -> Dict:
    """Process end of day: calculate final metrics"""
    results = end_of_day_processing(marketplace)

    marketplace.total_cancellations = results['total_cancellations']
    marketplace.total_waste = results['total_waste']
    marketplace.total_revenue = results['total_revenue']

    return {
        'total_cancellations': results['total_cancellations'],
        'total_waste': results['total_waste'],
        'total_waste_monetary': results.get('total_waste_monetary', 0.0),
        'total_revenue': results['total_revenue'],
        'total_completed_orders': results['total_completed_orders'],
        'total_customers': results.get('total_customers', marketplace.total_customers_seen),
        'customer_satisfaction': results.get('customer_satisfaction', 0.0),
        'stores': results['stores']
    }


def run_single_strategy_simulation(strategy, strategy_name: str, stores, customers, n: int,
                                   duration: float, seed: int, output_dir: str, 
                                   verbose: bool = False) -> Dict:
    """
    Run 10-day simulation with a specific strategy.
    """
    import pandas as pd
    from ranking_algorithm import GreedyStrategy
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Deep copy stores and customers to avoid state contamination
    import copy
    stores_copy = copy.deepcopy(stores)
    customers_copy = copy.deepcopy(customers)
    
    # Monkey patch select_stores to use the strategy
    original_select_stores = ranking_algorithm.select_stores
    ranking_algorithm.select_stores = strategy.select_stores
    
    # Initialize marketplace
    marketplace = Marketplace(stores_copy)
    marketplace.n = n
    
    # Tracking data structures
    customer_daily_activities = []
    daily_kpis_list = []
    
    # Run 10 days
    for day in range(1, 11):
        if verbose and (day % 5 == 0 or day == 1):
            print(f"  [{strategy_name}] Day {day}/10...")
        
        # Initialize day for stores
        initialize_day(stores_copy)
        
        # Reset marketplace counters for the day
        marketplace.total_revenue = 0.0
        marketplace.total_cancellations = 0
        marketplace.total_waste = 0
        marketplace.total_customers_seen = 0
        marketplace.current_time = 0.0
        marketplace.customers = []
        
        # Generate arrival times for this day
        arrival_times = sorted(np.random.uniform(0, duration, len(customers_copy)))
        for i, customer in enumerate(customers_copy):
            customer.arrival_time = arrival_times[i]
            customer.decision = None
            customer.chosen_store_id = None
            customer.displayed_stores = []
        
        # Sort customers by arrival time to process in order
        customers_sorted = sorted(customers_copy, key=lambda c: c.arrival_time)
        
        # Track customer orders for cancellation tracking
        # Dictionary preserves insertion order (Python 3.7+), so order = arrival order
        customer_orders_today = {}  # {customer_id: store_id} - ordered by arrival time
        
        # Process each customer arrival in order
        # Customers cannot cancel their own orders - only restaurant inventory shortage causes cancellations
        for customer in customers_sorted:
            decision = simulate_customer_arrival(marketplace, customer)
            
            # Track activity (including no_arrival)
            activity = {
                'day': day,
                'customer_id': customer.customer_id,
                'action': decision['action'],  # 'buy', 'leave', or 'no_arrival'
                'store_id': decision.get('store_id', None),
                'cancelled': False
            }
            customer_daily_activities.append(activity)
            
            # Only track orders (not no_arrival or leave)
            if decision['action'] == 'buy':
                customer_orders_today[customer.customer_id] = decision['store_id']
        
        # End of day processing
        day_results = process_end_of_day(marketplace)
        
        # Update cancellation status
        # Cancellations occur when restaurant's actual inventory < reservations
        # The LAST customers to order from that restaurant get their orders cancelled (LIFO)
        # Customers cannot cancel their own orders - only inventory shortage causes cancellations
        for store in stores_copy:
            if store.cancellation_count > 0:
                # Get all customers who ordered from this store today, ordered by arrival time
                store_customer_ids = [cid for cid, sid in customer_orders_today.items() 
                                     if sid == store.restaurant_id]
                # Sort by arrival time (last to arrive = last in list if customer_orders_today preserves order)
                # Since we process customers in arrival order, the last customers in the list are the last to order
                # These are the ones who get cancelled when actual_inventory < reservation_count
                cancelled_customer_ids = store_customer_ids[-store.cancellation_count:]
                
                for activity in customer_daily_activities:
                    if (activity['day'] == day and 
                        activity['customer_id'] in cancelled_customer_ids and
                        activity['store_id'] == store.restaurant_id and
                        activity['action'] == 'buy'):
                        activity['cancelled'] = True
        
        # Store daily KPIs
        daily_kpis = {
            'day': day,
            'total_customers': day_results['total_customers'],
            'total_completed_orders': day_results['total_completed_orders'],
            'total_cancellations': day_results['total_cancellations'],
            'total_revenue': day_results['total_revenue'],
            'total_waste': day_results['total_waste'],
            'total_waste_monetary': day_results.get('total_waste_monetary', 0.0),
            'customer_satisfaction': day_results.get('customer_satisfaction', 0.0),
            'conversion_rate': day_results.get('conversion_rate', 0.0)  # Pre-calculated conversion rate
        }
        daily_kpis_list.append(daily_kpis)
    
    # Restore original select_stores
    ranking_algorithm.select_stores = original_select_stores
    
    # Calculate average KPIs
    avg_kpis = {
        'strategy': strategy_name,
        'avg_total_customers': np.mean([k['total_customers'] for k in daily_kpis_list]),
        'avg_total_completed_orders': np.mean([k['total_completed_orders'] for k in daily_kpis_list]),
        'avg_total_cancellations': np.mean([k['total_cancellations'] for k in daily_kpis_list]),
        'avg_total_revenue': np.mean([k['total_revenue'] for k in daily_kpis_list]),
        'avg_total_waste': np.mean([k['total_waste'] for k in daily_kpis_list]),
        'avg_total_waste_monetary': np.mean([k['total_waste_monetary'] for k in daily_kpis_list]),
        'avg_customer_satisfaction': np.mean([k['customer_satisfaction'] for k in daily_kpis_list]),
        'avg_conversion_rate': np.mean([k.get('conversion_rate', 0.0) for k in daily_kpis_list]),
        'total_days': 10
    }
    
    return {
        'strategy_name': strategy_name,
        'customer_summary': customer_daily_activities,
        'average_kpis': avg_kpis,
        'daily_kpis': daily_kpis_list
    }


def compare_strategies(num_stores: int = 10, num_customers: int = 100, n: Optional[int] = None,
                      duration: float = 24.0, verbose: bool = True, seed: Optional[int] = None,
                      stores_csv: Optional[str] = None, customers_csv: Optional[str] = None,
                      output_dir: str = "simulation_results") -> Dict:
    """
    Run 10-day simulation with Greedy (baseline) and Near-Optimal strategies and compare KPIs.
    
    Returns:
        Dict with:
        - greedy_results: Results from GreedyStrategy (baseline)
        - near_optimal_results: Results from NearOptimalStrategy
        - comparison: Side-by-side KPI comparison
    """
    import pandas as pd
    import time
    from ranking_algorithm import GreedyStrategy, NearOptimalStrategy
    
    # Use time-based seed if not provided (ensures different results each run)
    if seed is None:
        seed = int(time.time() * 1000) % 1000000  # Use milliseconds for seed
    
    if verbose:
        print("="*70)
        print("COMPARING RANKING STRATEGIES")
        print(f"Random Seed: {seed} (results will vary each run)")
        print("="*70)
        print()
    
    # Generate stores randomly (don't read from CSV)
    stores = load_store_data(num_stores, num_customers=num_customers, seed=seed)
    num_stores = len(stores)
    
    # Generate customers randomly (don't read from CSV)
    arrival_times = sorted(np.random.uniform(0, duration, num_customers))
    customers = generate_customer(num_customers, arrival_times, seed=seed)
    num_customers = len(customers)
    
    # Calculate n
    total_est_inventory = sum(store.est_inventory for store in stores)
    n_value = n if n is not None else calculate_n(num_stores, num_customers, total_est_inventory)
    
    if verbose:
        print(f"Configuration:")
        print(f"  Stores: {num_stores}, Customers: {num_customers}, n={n_value}")
        print()
        print("Running Greedy Strategy (Baseline)...")
    
    # Run Greedy Strategy (Baseline)
    greedy_strategy = GreedyStrategy()
    greedy_results = run_single_strategy_simulation(
        greedy_strategy, "Greedy", stores, customers, n_value, duration, seed + 1, output_dir, verbose
    )
    
    if verbose:
        print("\nRunning Near-Optimal Strategy...")
    
    # Run Near-Optimal Strategy
    near_optimal_strategy = NearOptimalStrategy(exploration_rate=0.03)
    near_optimal_results = run_single_strategy_simulation(
        near_optimal_strategy, "NearOptimal", stores, customers, n_value, duration, seed + 2, output_dir, verbose
    )
    
    # Calculate metrics for comparison (excluding Conversion Rate, Waste Monetary, Customer Satisfaction)
    def calc_metrics(results):
        kpis = results['average_kpis']
        return [
            kpis['avg_total_customers'],
            kpis['avg_total_completed_orders'],
            kpis['avg_total_cancellations'],
            kpis['avg_total_revenue'],
            kpis['avg_total_waste'],
            (kpis['avg_total_completed_orders'] / (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) * 100) if (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) > 0 else 0,
            (kpis['avg_total_cancellations'] / (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) * 100) if (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) > 0 else 0
        ]
    
    # Create comparison with clear metric explanations (only Greedy vs Near-Optimal)
    comparison = {
        'metric': [
            'Average Customers per Day',
            'Average Completed Orders per Day',
            'Average Cancellations per Day (raw count)',
            'Average Revenue per Day',
            'Average Waste (units) per Day',
            'Fulfillment Rate (%) - % of orders fulfilled',
            'Cancellation Rate (%) - % of orders cancelled'
        ],
        'greedy': calc_metrics(greedy_results),
        'near_optimal': calc_metrics(near_optimal_results)
    }
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison)
    comparison_file = os.path.join(output_dir, "strategy_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    # Save individual strategy KPIs
    greedy_kpis_df = pd.DataFrame([greedy_results['average_kpis']])
    greedy_file = os.path.join(output_dir, "greedy_strategy_kpis.csv")
    greedy_kpis_df.to_csv(greedy_file, index=False)
    
    near_optimal_kpis_df = pd.DataFrame([near_optimal_results['average_kpis']])
    near_optimal_file = os.path.join(output_dir, "near_optimal_strategy_kpis.csv")
    near_optimal_kpis_df.to_csv(near_optimal_file, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON COMPLETE")
        print(f"{'='*70}")
        print(f"\nKPI COMPARISON (Baseline vs Near-Optimal):")
        print(f"{'Metric':<45} {'Greedy (Baseline)':<20} {'Near-Optimal':<20} {'Best':<10}")
        print("-" * 95)
        for i, metric in enumerate(comparison['metric']):
            greedy_val = comparison['greedy'][i]
            near_optimal_val = comparison['near_optimal'][i]
            
            # Determine best value (higher is better for most metrics, except cancellations and waste)
            # For cancellations and waste, lower is better
            if 'Cancellation' in metric or ('Waste' in metric and 'Rate' not in metric):
                # For cancellation rate, lower is better
                # For waste (units), lower is better
                best_val = min(greedy_val, near_optimal_val)
                best_str = "Greedy" if abs(greedy_val - best_val) < 0.01 else "Near-Optimal"
            else:
                # For revenue, completed orders, fulfillment rate - higher is better
                best_val = max(greedy_val, near_optimal_val)
                best_str = "Greedy" if abs(greedy_val - best_val) < 0.01 else "Near-Optimal"
            
            # Format values based on metric type
            if 'Revenue' in metric:
                greedy_str = f"${greedy_val:.2f}"
                near_opt_str = f"${near_optimal_val:.2f}"
            elif 'Rate' in metric or '%' in metric:
                greedy_str = f"{greedy_val:.2f}%"
                near_opt_str = f"{near_optimal_val:.2f}%"
            else:
                greedy_str = f"{greedy_val:.2f}"
                near_opt_str = f"{near_optimal_val:.2f}"
            
            print(f"{metric:<45} {greedy_str:<20} {near_opt_str:<20} {best_str:<10}")
        
        print(f"\nOutput Files:")
        print(f"  1. Strategy Comparison: {comparison_file}")
        print(f"  2. Greedy Strategy KPIs: {greedy_file}")
        print(f"  3. Near-Optimal Strategy KPIs: {near_optimal_file}")
        print(f"{'='*70}\n")
    
    return {
        'greedy_results': greedy_results,
        'near_optimal_results': near_optimal_results,
        'comparison': comparison
    }


def run_50_day_simulation(num_stores: int = 10, num_customers: int = 100, n: Optional[int] = None,
                          duration: float = 24.0, verbose: bool = True, seed: int = 42,
                          stores_csv: Optional[str] = None, customers_csv: Optional[str] = None,
                          output_dir: str = "simulation_results") -> Dict:
    """
    Run 10-day simulation with multinomial logit model and greedy ranking.
    
    Returns:
        Dict with:
        - customer_summary: List of daily customer activities
        - average_kpis: Average KPIs over 10 days
        - daily_kpis: List of KPIs for each day
    """
    import pandas as pd
    from ranking_algorithm import GreedyStrategy
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate stores randomly (don't read from CSV)
    stores = load_store_data(num_stores, num_customers=num_customers, seed=seed)
    num_stores = len(stores)
    
    # Generate customers randomly (don't read from CSV)
    arrival_times = sorted(np.random.uniform(0, duration, num_customers))
    customers = generate_customer(num_customers, arrival_times, seed=seed)
    num_customers = len(customers)
    
    # Use greedy strategy
    greedy_strategy = GreedyStrategy()
    # Monkey patch select_stores to use greedy strategy
    original_select_stores = ranking_algorithm.select_stores
    ranking_algorithm.select_stores = greedy_strategy.select_stores
    
    # Initialize marketplace
    marketplace = Marketplace(stores)
    total_est_inventory = sum(store.est_inventory for store in stores)
    marketplace.n = n if n is not None else calculate_n(num_stores, num_customers, total_est_inventory)
    
    # Tracking data structures
    customer_daily_activities = []  # List of dicts: {day, customer_id, action, store_id, cancelled}
    daily_kpis_list = []  # List of KPIs for each day
    
    if verbose:
        print(f"Starting 10-day simulation...")
        print(f"Stores: {num_stores}, Customers: {num_customers}, n={marketplace.n}")
        print()
    
    # Run 10 days
    for day in range(1, 11):
        if verbose and (day % 5 == 0 or day == 1):
            print(f"Day {day}/10...")
        
        # Initialize day for stores
        initialize_day(stores)
        
        # Reset marketplace counters for the day
        marketplace.total_revenue = 0.0
        marketplace.total_cancellations = 0
        marketplace.total_waste = 0
        marketplace.total_customers_seen = 0
        marketplace.current_time = 0.0
        marketplace.customers = []
        
        # Generate arrival times for this day
        arrival_times = sorted(np.random.uniform(0, duration, num_customers))
        for i, customer in enumerate(customers):
            customer.arrival_time = arrival_times[i]
            # Reset customer decision for the day
            customer.decision = None
            customer.chosen_store_id = None
            customer.displayed_stores = []
        
        # Sort customers by arrival time to process in order
        customers_sorted = sorted(customers, key=lambda c: c.arrival_time)
        
        # Track customer orders for cancellation tracking
        # Dictionary preserves insertion order (Python 3.7+), so order = arrival order
        customer_orders_today = {}  # {customer_id: store_id} - ordered by arrival time
        
        # Process each customer arrival in order
        # Customers cannot cancel their own orders - only restaurant inventory shortage causes cancellations
        for customer in customers_sorted:
            decision = simulate_customer_arrival(marketplace, customer)
            
            # Track activity (including no_arrival)
            activity = {
                'day': day,
                'customer_id': customer.customer_id,
                'action': decision['action'],  # 'buy', 'leave', or 'no_arrival'
                'store_id': decision.get('store_id', None),
                'cancelled': False  # Will be updated at end of day if restaurant has inventory shortage
            }
            customer_daily_activities.append(activity)
            
            # Only track orders (not no_arrival or leave)
            if decision['action'] == 'buy':
                customer_orders_today[customer.customer_id] = decision['store_id']
        
        # End of day processing
        day_results = process_end_of_day(marketplace)
        
        # Update cancellation status for customers
        # Cancellations occur when restaurant's actual inventory < reservations
        # The LAST customers to order from that restaurant get their orders cancelled (LIFO)
        # Customers cannot cancel their own orders - only inventory shortage causes cancellations
        for store in stores:
            if store.cancellation_count > 0:
                # Find customers who ordered from this store today
                # customer_orders_today preserves order of insertion (which follows arrival order)
                store_customer_ids = [cid for cid, sid in customer_orders_today.items() 
                                     if sid == store.restaurant_id]
                # Mark the last cancellation_count customers as cancelled (LIFO - last ordered, first cancelled)
                # These are the customers who ordered when the store was already at/over capacity
                cancelled_customer_ids = store_customer_ids[-store.cancellation_count:]
                
                # Update the activity records
                for activity in customer_daily_activities:
                    if (activity['day'] == day and 
                        activity['customer_id'] in cancelled_customer_ids and
                        activity['store_id'] == store.restaurant_id and
                        activity['action'] == 'buy'):
                        activity['cancelled'] = True
        
        # Store daily KPIs
        daily_kpis = {
            'day': day,
            'total_customers': day_results['total_customers'],
            'total_completed_orders': day_results['total_completed_orders'],
            'total_cancellations': day_results['total_cancellations'],
            'total_revenue': day_results['total_revenue'],
            'total_waste': day_results['total_waste'],
            'total_waste_monetary': day_results.get('total_waste_monetary', 0.0),
            'customer_satisfaction': day_results.get('customer_satisfaction', 0.0)
        }
        daily_kpis_list.append(daily_kpis)
        
        # Update data for next day (stores learn from today's accuracy)
        # This happens automatically in end_of_day_processing via calculate_accuracy()
        # Customer history persists across days (already in Customer object)
    
    # Restore original select_stores
    ranking_algorithm.select_stores = original_select_stores
    
    # Calculate average KPIs
    avg_kpis = {
        'avg_total_customers': np.mean([k['total_customers'] for k in daily_kpis_list]),
        'avg_total_completed_orders': np.mean([k['total_completed_orders'] for k in daily_kpis_list]),
        'avg_total_cancellations': np.mean([k['total_cancellations'] for k in daily_kpis_list]),
        'avg_total_revenue': np.mean([k['total_revenue'] for k in daily_kpis_list]),
        'avg_total_waste': np.mean([k['total_waste'] for k in daily_kpis_list]),
        'avg_total_waste_monetary': np.mean([k['total_waste_monetary'] for k in daily_kpis_list]),
        'avg_customer_satisfaction': np.mean([k['customer_satisfaction'] for k in daily_kpis_list]),
        'total_days': 10
    }
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Output File 1: Customer Summary
    customer_df = pd.DataFrame(customer_daily_activities)
    customer_file = os.path.join(output_dir, "customer_daily_summary.csv")
    customer_df.to_csv(customer_file, index=False)
    
    # Output File 2: Average KPIs
    avg_kpis_df = pd.DataFrame([avg_kpis])
    kpis_file = os.path.join(output_dir, "average_kpis_10_days.csv")
    avg_kpis_df.to_csv(kpis_file, index=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print("10-DAY SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nOutput Files:")
        print(f"  1. Customer Summary: {customer_file}")
        print(f"  2. Average KPIs: {kpis_file}")
        print(f"\nAverage KPIs over 10 days:")
        print(f"  Average Customers per Day: {avg_kpis['avg_total_customers']:.1f}")
        print(f"  Average Completed Orders per Day: {avg_kpis['avg_total_completed_orders']:.1f}")
        print(f"  Average Cancellations per Day: {avg_kpis['avg_total_cancellations']:.1f}")
        print(f"  Average Revenue per Day: ${avg_kpis['avg_total_revenue']:.2f}")
        print(f"  Average Waste per Day: {avg_kpis['avg_total_waste']:.1f} units")
        print(f"  Average Customer Satisfaction: {avg_kpis['avg_customer_satisfaction']*100:.1f}%")
        print(f"{'='*70}\n")
    
    return {
        'customer_summary': customer_daily_activities,
        'average_kpis': avg_kpis,
        'daily_kpis': daily_kpis_list
    }


def run_simulations(num_stores: int = 10, num_customers: int = 100, n: Optional[int] = None,
                   duration: float = 24.0, actual_inventories: Optional[Dict[int, int]] = None,
                   verbose: bool = True, seed: int = 42,
                   stores_csv: Optional[str] = None, customers_csv: Optional[str] = None) -> Dict:
    """
    Run complete simulation.
    
    If CSV files are provided and exist, loads data from CSV.
    Otherwise, generates synthetic data.
    """
    marketplace = initialize_marketplace(num_stores, actual_inventories, num_customers, seed=seed, stores_csv=stores_csv)
    if n is not None:
        marketplace.n = n
    else:
        n = marketplace.n

    if verbose:
        print(f"Initialized marketplace with {len(marketplace.stores)} stores")
        print(f"Simulating {num_customers} customers over {duration} hours")
        print(f"Calculated n = {n} stores to show each customer (constant for the day)\n")

    # Try to load customers from CSV if file exists
    if customers_csv and os.path.exists(customers_csv):
        arrival_times = sorted(np.random.uniform(0, duration, num_customers))
        customers = load_customers_from_csv(customers_csv, arrival_times, seed=seed)
        num_customers = len(customers)
        if verbose:
            print(f"Loaded {num_customers} customers from {customers_csv}")
    else:
        # Generate customers with arrival times
        arrival_times = sorted(np.random.uniform(0, duration, num_customers))
        customers = generate_customer(num_customers, arrival_times, seed=seed)

    # Process each customer
    decisions = []
    for i, customer in enumerate(customers):
        decision = simulate_customer_arrival(marketplace, customer)
        decisions.append(decision)

        if verbose and (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{num_customers} customers...")

    # End of day processing
    if verbose:
        print("\nProcessing end of day...")
    results = process_end_of_day(marketplace)

    if verbose:
        print("\nSimulation Results:")
        print(f"Total Customers: {results['total_customers']}")
        print(f"Total Completed Orders: {results['total_completed_orders']}")
        print(f"Total Revenue: ${results['total_revenue']:.2f}")
        print(f"Total Cancellations: {results['total_cancellations']}")
        print(f"Total Waste: {results['total_waste']}")

    return {
        'marketplace': marketplace,
        'results': results,
        'decisions': decisions
    }


# ==========================================
# SIMULATION HARNESS
# ==========================================

class InventoryAccuracy(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MIXED = "mixed"
    POOR = "poor"
    VERY_POOR = "very_poor"


class PriceDist(Enum):
    BUDGET = "budget"
    AFFORDABLE = "affordable"
    MIXED = "mixed"
    PREMIUM = "premium"
    LUXURY = "luxury"


class PopularitySkew(Enum):
    UNIFORM = "uniform"
    MODERATE = "moderate"
    SKEWED = "skewed"
    EXTREME = "extreme"


class ArrivalPattern(Enum):
    UNIFORM = "uniform"
    MORNING_RUSH = "morning"
    LUNCH_PEAK = "lunch"
    EVENING_SURGE = "evening"
    BIMODAL = "bimodal"
    WEEKEND = "weekend"


class CustomerSegment(Enum):
    PRICE_SENSITIVE = "price_sensitive"
    QUALITY_FOCUSED = "quality_focused"
    BALANCED = "balanced"
    EXPLORERS = "explorers"
    LOYAL = "loyal"


class MarketCondition(Enum):
    OVERSUPPLIED = "oversupplied"
    BALANCED = "balanced"
    COMPETITIVE = "competitive"
    UNDERSUPPLIED = "undersupplied"


class TimePeriod(Enum):
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    LOW_SEASON = "low_season"


@dataclass
class ScenarioConfig:
    scenario_id: str
    random_seed: int
    num_stores: int
    num_customers: int
    
    # Store characteristics
    inventory_accuracy: InventoryAccuracy = InventoryAccuracy.MIXED
    price_dist: PriceDist = PriceDist.MIXED
    popularity_skew: PopularitySkew = PopularitySkew.UNIFORM
    
    # Customer behavior
    arrival_pattern: ArrivalPattern = ArrivalPattern.UNIFORM
    customer_segment: CustomerSegment = CustomerSegment.BALANCED
    
    # Market conditions
    market_condition: MarketCondition = MarketCondition.BALANCED
    time_period: TimePeriod = TimePeriod.WEEKDAY
    
    # Display settings
    n_stores_to_show: Optional[int] = None
    
    # Multi-day simulation
    num_days: int = 1
    
    # Metadata
    description: str = ""


class SimulationHarness:
    """Simulation harness for batch testing and scenario management"""
    
    def __init__(self, output_dir: str = "simulation_results"):
        self.results_log = []
        self.customer_decisions_log = []
        self.store_state_log = []
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    def run_scenario(self, config: ScenarioConfig, strategy: RankingStrategy) -> Dict:
        """Run a single scenario with the given configuration and strategy"""
        days_suffix = f"{config.num_days} days" if config.num_days > 1 else "1 day"
        print(f"--- Running Scenario: {config.scenario_id} ({days_suffix}, Seed: {config.random_seed}) ---")
        
        # Monkey patch select_stores with strategy
        global select_stores
        original_select_stores = select_stores
        select_stores = strategy.select_stores
        
        try:
            # Multi-day simulation: accumulate results
            cumulative_revenue = 0.0
            cumulative_waste = 0
            cumulative_waste_monetary = 0.0
            cumulative_cancellations = 0
            cumulative_completed = 0
            cumulative_customers = 0
            
            for day_num in range(config.num_days):
                day_seed = config.random_seed + day_num
                
                sim_output = run_simulations(
                    num_stores=config.num_stores,
                    num_customers=config.num_customers,
                    n=config.n_stores_to_show,
                    duration=24.0,
                    verbose=False,
                    seed=day_seed
                )
                
                results = sim_output['results']
                cumulative_revenue += results['total_revenue']
                cumulative_waste += results['total_waste']
                cumulative_waste_monetary += results.get('total_waste_monetary', 0.0)
                cumulative_cancellations += results['total_cancellations']
                cumulative_completed += results['total_completed_orders']
                cumulative_customers += results['total_customers']
                
                if config.num_days > 7 and (day_num + 1) % 30 == 0:
                    print(f"    Day {day_num + 1}/{config.num_days} completed...")
            
            # Calculate aggregate KPIs
            kpis = {
                'scenario_id': config.scenario_id,
                'seed': config.random_seed,
                'strategy': strategy.__class__.__name__,
                'num_stores': config.num_stores,
                'num_customers': config.num_customers,
                'num_days': config.num_days,
                'total_revenue': cumulative_revenue,
                'total_waste': cumulative_waste,
                'total_waste_monetary': cumulative_waste_monetary,
                'total_cancellations': cumulative_cancellations,
                'total_completed_orders': cumulative_completed,
                'customer_satisfaction': cumulative_completed / cumulative_customers if cumulative_customers > 0 else 0,
                'avg_revenue_per_order': cumulative_revenue / cumulative_completed if cumulative_completed > 0 else 0,
                'avg_revenue_per_day': cumulative_revenue / config.num_days,
                'avg_waste_per_day': cumulative_waste / config.num_days,
                'avg_waste_monetary_per_day': cumulative_waste_monetary / config.num_days,
                'avg_cancellations_per_day': cumulative_cancellations / config.num_days
            }
            self.results_log.append(kpis)
            
            # Log details from last day
            marketplace = sim_output['marketplace']
            decisions = sim_output['decisions']
            
            if config.num_days <= 7:
                for customer, decision in zip(marketplace.customers, decisions):
                    self.customer_decisions_log.append({
                        'scenario_id': config.scenario_id,
                        'strategy': strategy.__class__.__name__,
                        'day': config.num_days,
                        'customer_id': customer.customer_id,
                        'arrival_time': customer.arrival_time,
                        'action': decision['action'],
                        'chosen_store_id': decision['store_id'],
                        'num_stores_displayed': len(customer.displayed_stores)
                    })
            
            for store_id, store_metrics in results['stores'].items():
                self.store_state_log.append({
                    'scenario_id': config.scenario_id,
                    'strategy': strategy.__class__.__name__,
                    'day': config.num_days,
                    'store_id': store_metrics['store_id'],
                    'store_name': store_metrics['name'],
                    'category': store_metrics['category'],
                    'rating': store_metrics['rating'],
                    'price': store_metrics['price'],
                    'est_inventory': store_metrics['est_inventory'],
                    'actual_inventory': store_metrics['actual_inventory'],
                    'accuracy_score': store_metrics['accuracy_score'],
                    'reservation_count': store_metrics['reservation_count'],
                    'exposure_count': store_metrics['exposure_count'],
                    'completed_orders': store_metrics['completed_order_count'],
                    'cancellations': store_metrics['cancellation_count']
                })
            
            days_label = "day" if config.num_days == 1 else f"{config.num_days} days"
            print(f"  Completed ({days_label}): Revenue=${kpis['total_revenue']:.2f}, Waste={kpis['total_waste']}, Cancellations={kpis['total_cancellations']}")
            return kpis
            
        finally:
            # Restore original select_stores
            select_stores = original_select_stores

    def generate_scenarios(self, mode: str = "comprehensive") -> List[ScenarioConfig]:
        """Generate scenarios based on mode"""
        scenarios = []
        base_seed = 1000
        scenario_counter = 0
        
        if mode == "quick":
            for scale in [(10, 150), (30, 500)]:
                num_stores, num_customers = scale
                for market in [MarketCondition.BALANCED, MarketCondition.COMPETITIVE]:
                    for time in [TimePeriod.WEEKDAY, TimePeriod.WEEKEND]:
                        scenario_counter += 1
                        scenarios.append(ScenarioConfig(
                            scenario_id=f"Q{scenario_counter:03d}_{num_stores}s_{num_customers}c_{market.value}_{time.value}",
                            random_seed=base_seed + scenario_counter,
                            num_stores=num_stores,
                            num_customers=num_customers,
                            market_condition=market,
                            time_period=time,
                            description=f"Quick test: {num_stores} stores, {num_customers} customers"
                        ))
        elif mode == "standard":
            # Standard scenarios (simplified for brevity - add more as needed)
            for scale_name, num_stores, num_customers in [
                ("small", 10, 150),
                ("medium", 30, 500),
                ("large", 80, 1200)
            ]:
                for market in [MarketCondition.OVERSUPPLIED, MarketCondition.BALANCED, MarketCondition.UNDERSUPPLIED]:
                    for time in [TimePeriod.WEEKDAY, TimePeriod.WEEKEND]:
                        scenario_counter += 1
                        scenarios.append(ScenarioConfig(
                            scenario_id=f"S{scenario_counter:03d}_scale_{scale_name}_{market.value[:4]}_{time.value[:4]}",
                            random_seed=base_seed + scenario_counter,
                            num_stores=num_stores,
                            num_customers=num_customers,
                            market_condition=market,
                            time_period=time,
                            description=f"Scale test: {scale_name} market"
                        ))
        else:  # comprehensive
            # Comprehensive scenarios (simplified - can expand)
            print("Generating comprehensive scenario matrix...")
            for scale_name, num_stores, num_customers in [
                ("tiny", 5, 80),
                ("small", 15, 250),
                ("medium", 40, 600),
                ("large", 100, 1500)
            ]:
                for market in MarketCondition:
                    for time in [TimePeriod.WEEKDAY, TimePeriod.WEEKEND, TimePeriod.HOLIDAY]:
                        scenario_counter += 1
                        scenarios.append(ScenarioConfig(
                            scenario_id=f"C{scenario_counter:03d}_scale_{scale_name}_{market.value[:6]}_{time.value[:4]}",
                            random_seed=base_seed + scenario_counter,
                            num_stores=num_stores,
                            num_customers=num_customers,
                            market_condition=market,
                            time_period=time,
                            description=f"{scale_name.title()} market: {num_stores}s, {num_customers}c"
                        ))
        
        print(f"Generated {len(scenarios)} scenarios in '{mode}' mode")
        return scenarios

    def run_batch(self, strategies: List[RankingStrategy], mode: str = "quick", save_results: bool = True):
        """Run all scenarios against all strategies"""
        scenarios = self.generate_scenarios(mode=mode)
        
        print(f"\n{'='*70}")
        print(f"Running {len(scenarios)} scenarios × {len(strategies)} strategies = {len(scenarios) * len(strategies)} total runs")
        print(f"{'='*70}\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[Scenario {i}/{len(scenarios)}] {scenario.scenario_id}")
            if scenario.description:
                print(f"  Description: {scenario.description}")
            for strategy in strategies:
                self.run_scenario(scenario, strategy)
        
        if save_results:
            self.save_results()

    def get_results_df(self):
        """Get summary results as DataFrame"""
        return pd.DataFrame(self.results_log)

    def get_customer_decisions_df(self):
        """Get detailed customer decisions as DataFrame"""
        return pd.DataFrame(self.customer_decisions_log)

    def get_store_state_df(self):
        """Get detailed store state as DataFrame"""
        return pd.DataFrame(self.store_state_log)

    def save_results(self):
        """Save all results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary_df = self.get_results_df()
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSaved summary results to: {summary_file}")
        
        if self.customer_decisions_log:
            decisions_df = self.get_customer_decisions_df()
            decisions_file = os.path.join(self.output_dir, f"customer_decisions_{timestamp}.csv")
            decisions_df.to_csv(decisions_file, index=False)
            print(f"Saved customer decisions to: {decisions_file}")
        
        if self.store_state_log:
            store_df = self.get_store_state_df()
            store_file = os.path.join(self.output_dir, f"store_state_{timestamp}.csv")
            store_df.to_csv(store_file, index=False)
            print(f"Saved store state to: {store_file}")
        
        print(f"\nAll results saved in directory: {self.output_dir}\n")

