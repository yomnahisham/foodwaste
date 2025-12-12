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
from multiprocessing import Pool, cpu_count
import functools

from restaurant_api import (
    Restaurant, load_store_data, get_all_stores, initialize_day,
    end_of_day_processing, end_of_day_processing_enhanced, update_reservation, update_exposure
)
from customer_api import (
    Customer, generate_customer, customer_arrives, display_stores_to_customer,
    customer_makes_decision, PRICE_ELASTICITY
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
        # network effects: track popular stores for social proof
        self.store_popularity = {}  # {store_id: popularity_score} - updated based on recent orders


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


def is_peak_hour(hour: float) -> bool:
    """check if given hour falls within peak demand periods"""
    # lunch peak: 11am-2pm (11-14)
    # dinner peak: 5pm-9pm (17-21)
    return (11.0 <= hour < 14.0) or (17.0 <= hour < 21.0)


def get_seasonal_multiplier(day: int) -> float:
    """
    get seasonal demand multiplier based on day number.
    simulates monthly/quarterly patterns and holidays.
    
    patterns:
    - december (holidays): 1.4x demand
    - january (new year): 1.2x demand
    - summer months (june-august): 1.1x demand
    - spring/fall: 1.0x (baseline)
    - thanksgiving week: 1.5x demand
    """
    # approximate month from day (assuming ~30 days per month)
    month = (day // 30) % 12 + 1  # 1-12
    
    # base seasonal multipliers
    seasonal_mult = 1.0
    if month == 12:  # december (holidays)
        seasonal_mult = 1.4
    elif month == 1:  # january (new year)
        seasonal_mult = 1.2
    elif month in [6, 7, 8]:  # summer
        seasonal_mult = 1.1
    elif month == 11:  # november (thanksgiving)
        seasonal_mult = 1.3
    
    # holiday effects (thanksgiving week, christmas week)
    day_in_month = day % 30
    if month == 11 and 20 <= day_in_month <= 27:  # thanksgiving week
        seasonal_mult = 1.5
    elif month == 12 and 20 <= day_in_month <= 27:  # christmas week
        seasonal_mult = 1.6
    
    return seasonal_mult


def generate_time_weighted_arrival_times(num_customers: int, duration: float = 24.0, 
                                        rng: np.random.RandomState = None, is_weekend: bool = False,
                                        seasonal_multiplier: float = 1.0) -> List[float]:
    """
    generate arrival times with realistic time-of-day demand patterns.
    creates peaks at lunch (11am-2pm) and dinner (5pm-9pm).
    adjusts for weekday vs weekend patterns and seasonal effects.
    
    weekday time multipliers:
    - morning (6am-11am): 0.3x
    - lunch peak (11am-2pm): 1.5x
    - afternoon (2pm-5pm): 0.7x
    - dinner peak (5pm-9pm): 1.8x
    - late night (9pm-6am): 0.2x
    
    weekend adjustments:
    - overall demand: 1.3x multiplier
    - dinner peak later: 6pm-9pm instead of 5pm-9pm
    - lunch less pronounced: 1.2x instead of 1.5x
    
    seasonal_multiplier: additional multiplier for seasonal/holiday effects
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # define time periods and their multipliers (weekday base)
    if is_weekend:
        # weekend patterns: later dinner, less lunch focus, higher overall demand
        time_periods = [
            (0.0, 6.0, 0.2 * 1.3),    # late night: 0.2x * 1.3
            (6.0, 11.0, 0.3 * 1.3),   # morning: 0.3x * 1.3
            (11.0, 14.0, 1.2 * 1.3),  # lunch peak: 1.2x * 1.3 (less pronounced)
            (14.0, 18.0, 0.7 * 1.3),  # afternoon: 0.7x * 1.3 (extended)
            (18.0, 21.0, 1.8 * 1.3),  # dinner peak: 1.8x * 1.3 (later: 6-9pm)
            (21.0, 24.0, 0.2 * 1.3),  # late night: 0.2x * 1.3
        ]
    else:
        # weekday patterns
        time_periods = [
            (0.0, 6.0, 0.2),    # late night: 0.2x
            (6.0, 11.0, 0.3),   # morning: 0.3x
            (11.0, 14.0, 1.5),  # lunch peak: 1.5x
            (14.0, 17.0, 0.7),  # afternoon: 0.7x
            (17.0, 21.0, 1.8),  # dinner peak: 1.8x
            (21.0, 24.0, 0.2),  # late night: 0.2x
        ]
    
    # apply seasonal multiplier to all periods
    time_periods = [(start, end, mult * seasonal_multiplier) for start, end, mult in time_periods]
    
    # calculate total weight for normalization
    total_weight = sum((end - start) * mult for start, end, mult in time_periods)
    
    # generate arrival times using weighted distribution
    # seasonal multiplier affects the weights, not the count (to maintain fair comparison)
    arrival_times = []
    for _ in range(num_customers):
        # sample a time period weighted by duration * multiplier
        rand = rng.uniform(0, total_weight)
        cumulative = 0.0
        selected_period = None
        
        for start, end, mult in time_periods:
            period_weight = (end - start) * mult
            cumulative += period_weight
            if rand <= cumulative:
                selected_period = (start, end)
                break
        
        if selected_period:
            start, end = selected_period
            arrival_time = rng.uniform(start, end)
            arrival_times.append(arrival_time)
        else:
            # fallback (shouldn't happen)
            arrival_times.append(rng.uniform(0, duration))
    
    return sorted(arrival_times)


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


def simulate_customer_arrival(marketplace: Marketplace, customer: Customer, skip_arrival_check: bool = False) -> Dict:
    """
    Simulate a single customer arrival and decision using proper MNL model.
    
    First, customer decides if they will even open the app (arrival probability).
    Then, if they arrive, they see stores and make a choice using MNL.
    
    Args:
        skip_arrival_check: If True, skip the arrival probability check (used when
                          arrival decision is pre-generated for fair comparison)
    """
    # stage 1: customer decides if they will open the app today
    # realistic behavior: not all customers open the app every day
    if not skip_arrival_check:
        # base arrival probability (can be adjusted based on customer characteristics)
        base_arrival_probability = 0.7  # 70% base chance customer opens app on a given day
        
        # adjust based on customer satisfaction (satisfied customers more likely to return)
        # satisfaction_level is typically 0.5-1.0, so multiplier ranges from 1.0 to 1.5
        satisfaction_multiplier = 0.5 + customer.satisfaction_level
        arrival_probability = base_arrival_probability * satisfaction_multiplier
        
        # clamp to valid probability range [0, 1]
        arrival_probability = min(1.0, max(0.0, arrival_probability))
        
        # customer decides if they open the app
        if np.random.uniform() > arrival_probability:
            # customer doesn't open app today
            return {
                'action': 'no_arrival',  # new action: customer didn't even arrive
                'store_id': None
            }
    
    # customer opens app - proceed with decision
    customer_arrives(customer)
    marketplace.current_time = customer.arrival_time
    marketplace.total_customers_seen += 1

    all_stores = marketplace.stores

    # select n stores to display using ranking algorithm
    t = marketplace.total_customers_seen
    n = marketplace.n
    displayed_stores = ranking_algorithm.select_stores(customer, n, all_stores, t)

    # display stores to customer
    display_stores_to_customer(customer, displayed_stores)

    # update exposure for displayed stores
    for store in displayed_stores:
        store.exposure_count += 1

    # customer makes decision using proper MNL model
    # note: current_hour parameter is kept for API compatibility but not used (food waste collected at 10 PM)
    decision = customer_makes_decision(customer, displayed_stores, customer.arrival_time)

    # update reservations if customer bought
    if decision['action'] == 'buy':
        # find store in marketplace.stores and update it directly
        store = next((s for s in all_stores if s.restaurant_id == decision['store_id']), None)
        if store:
            store.reserve_order()
            marketplace.total_revenue += store.price

    # add customer to marketplace
    marketplace.customers.append(customer)

    return decision


def process_end_of_day(marketplace: Marketplace) -> Dict:
    """
    Process end of day logic using ENHANCED processing and return full results.
    """
    results = end_of_day_processing_enhanced(marketplace)
    
    # DEBUG PRINT
    print(f"DEBUG: profit_margin_proxy={results.get('profit_margin_proxy')}, keys={list(results.keys())}")
    
    # Map results to marketplace state for consistency
    marketplace.total_cancellations = results['total_cancellations']
    marketplace.total_waste = results['total_waste']
    marketplace.total_revenue = results['total_revenue']

    # Return the FULL enhanced results dictionary
    return results


def run_single_strategy_simulation(strategy, strategy_name: str, stores, customers, n: int,
                                   duration: float, seed: int, output_dir: str, 
                                   verbose: bool = False, num_days: int = 10) -> Dict:
    """
    Run multi-day simulation with a specific strategy.
    
    Args:
        num_days: Number of days to simulate (default: 10)
    """
    import pandas as pd
    from ranking_algorithm import GreedyStrategy
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Deep copy stores and customers to avoid state contamination
    # each strategy run starts with fresh state
    import copy
    stores_copy = copy.deepcopy(stores)
    customers_copy = copy.deepcopy(customers)
    
    # update strategy's internal references to point to the copies instead of originals
    # this ensures that updates during simulation (e.g., learned_preferences) affect
    # the same objects the strategy uses for collaborative filtering
    if hasattr(strategy, 'customer_db') and strategy.customer_db is not None:
        # find corresponding customers in the copy by ID
        customer_id_to_copy = {c.customer_id: c for c in customers_copy}
        strategy.customer_db = [customer_id_to_copy.get(c.customer_id, c) 
                               for c in strategy.customer_db]
    
    if hasattr(strategy, 'restaurants_db') and strategy.restaurants_db is not None:
        # find corresponding stores in the copy by ID
        store_id_to_copy = {s.restaurant_id: s for s in stores_copy}
        strategy.restaurants_db = [store_id_to_copy.get(s.restaurant_id, s) 
                                   for s in strategy.restaurants_db]
    
    # generate all arrival times upfront for all days using separate RNGs
    # this ensures fair comparison: all strategies see the same arrival time patterns
    # regardless of how much randomness each strategy uses during processing
    arrival_rng = np.random.RandomState(seed)
    all_days_arrival_times = []
    
    for day in range(num_days):
        # use day number as offset to ensure different but deterministic arrival times per day
        day_rng = np.random.RandomState(seed + day * 10000)
        # calculate day of week for day-specific patterns
        day_of_week = day % 7
        is_weekend = day_of_week >= 5  # Saturday (5) or Sunday (6)
        # get seasonal multiplier for this day
        seasonal_mult = get_seasonal_multiplier(day)
        # generate time-weighted arrival times with realistic peaks (adjusted for weekday/weekend and seasonal effects)
        arrival_times = generate_time_weighted_arrival_times(len(customers_copy), duration, day_rng, is_weekend, seasonal_mult)
        all_days_arrival_times.append(arrival_times)
    
    # reset main RNG to seed for strategy-specific randomness
    # this ensures strategies can use randomness without affecting arrival times
    np.random.seed(seed)
    
    # note: arrival decisions are NOT pre-generated for all days upfront.
    # instead, they are generated at the start of each day based on current customer satisfaction levels. 
    # this allows satisfaction changes during simulation to affect future arrival probabilities, while still maintaining determinism through the use of a deterministic RNG seed per day
    
    # monkey patch select_stores to use the strategy
    original_select_stores = ranking_algorithm.select_stores
    ranking_algorithm.select_stores = strategy.select_stores
    
    # Initialize marketplace with the COPIED stores
    marketplace = Marketplace(stores_copy)
    marketplace.n = n
    
    # Tracking data structures
    customer_daily_activities = []
    daily_kpis_list = []
    
    # Run simulation for specified number of days
    for day in range(1, num_days + 1):
        if verbose and (day % 5 == 0 or day == 1 or day == num_days):
            print(f"  [{strategy_name}] Day {day}/{num_days}...")
        
        # calculate day of week (0=Monday, 6=Sunday) for day-of-week effects
        day_of_week = (day - 1) % 7
        is_weekend = day_of_week >= 5  # Saturday (5) or Sunday (6)
        
        # initialize day for stores (resets counters on the copied stores)
        initialize_day(stores_copy)
        
        # stores may start promotions at beginning of day
        # simulate some stores running promotions (e.g., low inventory stores)
        promotion_rng = np.random.RandomState(seed + day * 20000)
        for store in stores_copy:
            # stores with low inventory more likely to promote
            if store.est_inventory < 5 and promotion_rng.uniform() < 0.3:
                discount = promotion_rng.uniform(10, 25)  # 10-25% discount
                store.start_promotion('discount', discount, duration_hours=6.0, start_time=0.0)
        
        # update churn status: increment days since churn for churned customers
        for customer in customers_copy:
            if customer.is_churned:
                customer.days_since_churn += 1
        
        # reset marketplace counters for the day
        marketplace.total_revenue = 0.0
        marketplace.total_cancellations = 0
        marketplace.total_waste = 0
        marketplace.total_customers_seen = 0
        marketplace.current_time = 0.0
        marketplace.customers = []
        
        # update network effects: propagate social influence from previous day
        # popular stores from previous day influence customer preferences
        if day > 1:
            # decay previous popularity (exponential decay)
            for store_id in marketplace.store_popularity:
                marketplace.store_popularity[store_id] *= 0.7  # 30% decay per day
            
            # update customer social influence scores based on store popularity
            max_popularity = max(marketplace.store_popularity.values()) if marketplace.store_popularity else 1.0
            for customer in customers_copy:
                for store_id, popularity in marketplace.store_popularity.items():
                    # normalize popularity and assign to customer
                    normalized_popularity = popularity / max(1.0, max_popularity)
                    customer.social_influence_score[store_id] = normalized_popularity
        
        # reset store popularity for new day (will be built up during the day)
        marketplace.store_popularity = {}
        
        # reset store recent orders count for network effects
        for store in stores_copy:
            store.recent_orders_count = 0
        
        # use pre-generated arrival times for this day
        arrival_times = all_days_arrival_times[day - 1]
        
        # generate arrival decisions for this day
        # IMPORTANT: use INITIAL satisfaction levels (from day 1) for fair comparison
        # this ensures all strategies see the same customer arrival patterns
        # if we used updated satisfaction, different strategies would have different arrival counts making comparison unfair
        # use a deterministic RNG seed per day to ensure fair comparison across strategies
        base_arrival_probability = 0.7  # base chance customer opens app
        decision_rng = np.random.RandomState(seed + (day - 1) * 10000 + 5000)
        arrival_decision_dict = {}
        
        # create mapping from customer to index for efficient lookup
        customer_to_idx = {id(c): i for i, c in enumerate(customers_copy)}
        
        # store initial satisfaction levels on first day for fair comparison
        if day == 1:
            for customer in customers_copy:
                customer._initial_satisfaction = customer.satisfaction_level
                customer._initial_is_churned = customer.is_churned
        
        for i, customer in enumerate(customers_copy):
            # use INITIAL satisfaction level (from day 1) for fair comparison
            # this ensures all strategies see the same arrival patterns
            initial_satisfaction = getattr(customer, '_initial_satisfaction', customer.satisfaction_level)
            satisfaction_multiplier = 0.5 + initial_satisfaction
            arrival_probability = base_arrival_probability * satisfaction_multiplier
            
            # use initial churn status for fair comparison
            initial_is_churned = getattr(customer, '_initial_is_churned', customer.is_churned)
            if initial_is_churned:
                arrival_probability *= 0.1  # 10% of base probability
            
            arrival_probability = min(1.0, max(0.0, arrival_probability))
            
            # generate the decision using deterministic RNG
            will_arrive = decision_rng.uniform() <= arrival_probability
            arrival_decision_dict[i] = will_arrive
        
        for i, customer in enumerate(customers_copy):
            customer.arrival_time = arrival_times[i]
            customer.decision = None
            customer.chosen_store_id = None
            customer.displayed_stores = []
        
        # sort customers by arrival time to process in order
        customers_sorted = sorted(customers_copy, key=lambda c: c.arrival_time)
        
        # track customer orders for cancellation tracking
        # dictionary preserves insertion order, so order = arrival order
        customer_orders_today = {}  # {customer_id: store_id} - ordered by arrival time
        
        # track peak vs off-peak metrics for supply-demand balance KPI
        peak_reservations = {}  # {store_id: count} - reservations during peak hours
        peak_completed = {}     # {store_id: count} - completed orders during peak hours
        peak_cancellations = {} # {store_id: count} - cancellations during peak hours
        peak_waste = {}         # {store_id: count} - waste during peak hours
        offpeak_reservations = {}
        offpeak_completed = {}
        offpeak_cancellations = {}
        offpeak_waste = {}
        
        # track segment-specific metrics
        segment_orders = {}  # {segment: count} - orders by segment
        segment_revenue = {}  # {segment: float} - revenue by segment
        segment_cancellations = {}  # {segment: count} - cancellations by segment
        segment_customers = {}  # {segment: set} - customers who arrived by segment
        
        # calculate expected sales rate for mid-day inventory updates
        total_expected_sales = sum(s.est_inventory for s in stores_copy)
        expected_sales_rate = total_expected_sales / duration if duration > 0 else 0
        
        # process each customer arrival in order
        # customers cannot cancel their own orders - only restaurant inventory shortage causes cancellations
        customers_processed = 0
        for customer in customers_sorted:
            # check pre-generated arrival decision (use efficient lookup)
            customer_idx = customer_to_idx.get(id(customer), -1)
            will_arrive = arrival_decision_dict.get(customer_idx, True)  # default to True if not found
            
            if not will_arrive:
                # customer doesn't open app today (pre-determined)
                decision = {
                    'action': 'no_arrival',
                    'store_id': None
                }
            else:
                # Customer opens app - proceed with decision
                decision = simulate_customer_arrival(marketplace, customer, skip_arrival_check=True)
            
            # update customer history based on their action
            if decision['action'] == 'buy' and decision.get('store_id'):
                customer.add_to_history(decision['store_id'], 'order')
                # track order for CLV calculation
                store = next((s for s in stores_copy if s.restaurant_id == decision['store_id']), None)
                if store:
                    customer.record_order(day, store.price)
            elif decision['action'] != 'no_arrival' and customer.displayed_stores:
                # customer viewed stores but didn't buy - add viewed stores to history
                for store_id in customer.displayed_stores:
                    customer.add_to_history(store_id, 'view')
            
            # --- STRATEGY LEARNING HOOK ---
            # update strategy if it has learning capabilities (e.g. Anan_Strategy)
            if hasattr(strategy, 'update_learned_preferences'):
                store = None
                if decision['store_id'] is not None:
                     store = next((s for s in marketplace.stores if s.restaurant_id == decision['store_id']), None)
                strategy.update_learned_preferences(customer, decision, store)
            # -----------------------------
            
            # track activity (including no_arrival)
            activity = {
                'day': day,
                'customer_id': customer.customer_id,
                'action': decision['action'],  # 'buy', 'leave', or 'no_arrival'
                'store_id': decision.get('store_id', None),
                'cancelled': False
            }
            customer_daily_activities.append(activity)
            
            # track segment for customers who arrived
            if will_arrive:
                segment = customer.customer_segment
                if segment not in segment_customers:
                    segment_customers[segment] = set()
                segment_customers[segment].add(customer.customer_id)
                # record activity for retention tracking
                customer.record_activity(day)
            
            # only track orders (not no_arrival or leave)
            if decision['action'] == 'buy':
                customer_orders_today[customer.customer_id] = decision['store_id']
                # track peak vs off-peak reservations for supply-demand balance KPI
                store_id = decision['store_id']
                is_peak = is_peak_hour(customer.arrival_time)
                if is_peak:
                    peak_reservations[store_id] = peak_reservations.get(store_id, 0) + 1
                else:
                    offpeak_reservations[store_id] = offpeak_reservations.get(store_id, 0) + 1
                
                # track segment-specific metrics
                segment = customer.customer_segment
                segment_orders[segment] = segment_orders.get(segment, 0) + 1
                store = next((s for s in stores_copy if s.restaurant_id == store_id), None)
                if store:
                    segment_revenue[segment] = segment_revenue.get(segment, 0.0) + store.price
                    # network effects: update store popularity (social proof)
                    marketplace.store_popularity[store_id] = marketplace.store_popularity.get(store_id, 0.0) + 1.0
                    store.recent_orders_count += 1
            
            customers_processed += 1
            
            # update mid-day inventory estimates, dynamic pricing, and promotions periodically (every 10 customers)
            # this allows stores to adjust estimates and prices based on sales velocity and demand
            if customers_processed % 10 == 0:
                time_elapsed = customer.arrival_time
                is_peak = is_peak_hour(time_elapsed)
                for store in stores_copy:
                    store.update_midday_inventory_estimate(expected_sales_rate, time_elapsed, duration)
                    # update promotions (check if expired)
                    store.update_promotion(time_elapsed)
                    # update dynamic pricing (only if no active promotion)
                    if not store.active_promotion:
                        inventory_ratio = (store.est_inventory - store.reservation_count) / max(1, store.est_inventory)
                        demand_pressure = store.reservation_count / max(1, expected_sales_rate * (time_elapsed / duration))
                        store.update_dynamic_price(time_elapsed, is_peak, inventory_ratio, demand_pressure)
        
        # end of day processing
        day_results = process_end_of_day(marketplace)
        
        # update competition dynamics: calculate market share and competitive positioning
        # do this after end-of-day processing so we have final order counts
        total_orders_today = sum(s.completed_order_count for s in stores_copy)
        market_shares = []
        leader_count = 0
        follower_count = 0
        if total_orders_today > 0:
            for store in stores_copy:
                store.market_share = (store.completed_order_count / total_orders_today) * 100
                market_shares.append(store.market_share)
                # update competitive position based on market share
                if store.market_share > 15.0:  # top performer
                    store.competitive_position = 'leader'
                    leader_count += 1
                elif store.market_share < 3.0:  # low performer
                    store.competitive_position = 'follower'
                    follower_count += 1
                else:
                    store.competitive_position = 'neutral'
                # track revenue for competitive analysis
                if not hasattr(store, 'revenue_history'):
                    store.revenue_history = []
                store.revenue_history.append(store.completed_order_count * store.price)
        
        # calculate HHI (Herfindahl-Hirschman Index) for market concentration
        # HHI = sum of (market_share^2) * 10000 (ranges 0-10000)
        hhi = sum((ms / 100.0) ** 2 for ms in market_shares) * 10000 if market_shares else 0.0
        avg_market_share_day = np.mean(market_shares) if market_shares else 0.0
        
        # calculate peak vs off-peak metrics for supply-demand balance KPI
        # we need to determine which orders were completed/cancelled during peak hours
        # by checking customer arrival times for each order
        peak_total_reservations = sum(peak_reservations.values())
        offpeak_total_reservations = sum(offpeak_reservations.values())
        
        # calculate peak/off-peak completed and cancelled based on store results
        # and customer arrival times
        for store in stores_copy:
            store_id = store.restaurant_id
            peak_res = peak_reservations.get(store_id, 0)
            offpeak_res = offpeak_reservations.get(store_id, 0)
            total_res = peak_res + offpeak_res
            
            if total_res > 0:
                # proportionally allocate completed/cancelled to peak vs off-peak
                # ensure we don't allocate more than actual reservations
                peak_ratio = peak_res / total_res if total_res > 0 else 0
                # cap allocations to not exceed reservations
                peak_completed[store_id] = min(int(store.completed_order_count * peak_ratio), peak_res)
                peak_cancellations[store_id] = min(int(store.cancellation_count * peak_ratio), peak_res)
                # ensure off-peak doesn't exceed off-peak reservations
                offpeak_completed[store_id] = min(store.completed_order_count - peak_completed[store_id], offpeak_res)
                offpeak_cancellations[store_id] = min(store.cancellation_count - peak_cancellations[store_id], offpeak_res)
                
                # waste is allocated based on when inventory was available
                # for simplicity, allocate proportionally (could be improved)
                peak_waste[store_id] = int(day_results.get('total_waste', 0) * peak_ratio / len(stores_copy))
                offpeak_waste[store_id] = (day_results.get('total_waste', 0) / len(stores_copy)) - peak_waste[store_id]
        
        peak_total_completed = sum(peak_completed.values())
        peak_total_cancellations = sum(peak_cancellations.values())
        peak_total_waste = sum(peak_waste.values())
        offpeak_total_completed = sum(offpeak_completed.values())
        offpeak_total_cancellations = sum(offpeak_cancellations.values())
        offpeak_total_waste = sum(offpeak_waste.values())
        
        # calculate peak supply-demand balance metrics
        # cap fulfillment rate at 100% (can't fulfill more than reservations)
        peak_fulfillment_rate = min((peak_total_completed / peak_total_reservations * 100) if peak_total_reservations > 0 else 0.0, 100.0)
        peak_cancellation_rate = min((peak_total_cancellations / peak_total_reservations * 100) if peak_total_reservations > 0 else 0.0, 100.0)
        peak_waste_rate = (peak_total_waste / sum(s.actual_inventory for s in stores_copy) * 100) if stores_copy else 0.0
        
        # calculate peak supply-demand ratio (how well supply matched demand)
        # this measures how much demand (reservations) relative to available supply during peaks
        # we approximate by using est_inventory as the supply available
        peak_total_supply = sum(s.est_inventory for s in stores_copy)
        peak_supply_demand_ratio = (peak_total_reservations / peak_total_supply) if peak_total_supply > 0 else 0.0
        
        # off-peak metrics for comparison
        # cap fulfillment rate at 100% (can't fulfill more than reservations)
        offpeak_fulfillment_rate = min((offpeak_total_completed / offpeak_total_reservations * 100) if offpeak_total_reservations > 0 else 0.0, 100.0)
        offpeak_cancellation_rate = min((offpeak_total_cancellations / offpeak_total_reservations * 100) if offpeak_total_reservations > 0 else 0.0, 100.0)
        
        # track segment cancellations
        for activity in customer_daily_activities:
            if activity['day'] == day and activity['action'] == 'buy' and activity.get('cancelled', False):
                customer_id = activity['customer_id']
                customer = customer_id_to_obj.get(customer_id)
                if customer:
                    segment = customer.customer_segment
                    segment_cancellations[segment] = segment_cancellations.get(segment, 0) + 1
        
        # update cancellation status and customer satisfaction
        # cancellations occur when restaurant's actual inventory < reservations
        # the last customers to order from that restaurant get their orders cancelled (LIFO)
        # customers cannot cancel their own orders - only inventory shortage causes cancellations
        cancelled_customers = set()  # track which customers got cancelled
        
        for store in stores_copy:
            if store.cancellation_count > 0:
                # get all customers who ordered from this store today, ordered by arrival time
                store_customer_ids = [cid for cid, sid in customer_orders_today.items() 
                                     if sid == store.restaurant_id]
                # sort by arrival time (last to arrive = last in list if customer_orders_today preserves order)
                # since we process customers in arrival order, the last customers in the list are the last to order
                # these are the ones who get cancelled when actual_inventory < reservation_count
                cancelled_customer_ids = store_customer_ids[-store.cancellation_count:]
                cancelled_customers.update(cancelled_customer_ids)
                
                for activity in customer_daily_activities:
                    if (activity['day'] == day and 
                        activity['customer_id'] in cancelled_customer_ids and
                        activity['store_id'] == store.restaurant_id and
                        activity['action'] == 'buy'):
                        activity['cancelled'] = True
        
        # update individual customer satisfaction levels and churn status based on their experience today
        # find customers who successfully completed orders vs got cancelled
        customer_id_to_obj = {c.customer_id: c for c in customers_copy}
        
        for activity in customer_daily_activities:
            if activity['day'] == day and activity['action'] == 'buy':
                customer_id = activity['customer_id']
                customer = customer_id_to_obj.get(customer_id)
                if customer:
                    if activity['cancelled']:
                        # order was cancelled - decrease satisfaction and track churn
                        customer.satisfaction_level = max(0.0, customer.satisfaction_level - 0.15)
                        customer.consecutive_cancellations += 1
                        
                        # churn logic: 2+ consecutive cancellations increases churn risk
                        if customer.consecutive_cancellations >= 2:
                            customer.is_churned = True
                            customer.days_since_churn = 0
                    else:
                        # order completed successfully - increase satisfaction and reset cancellation streak
                        customer.satisfaction_level = min(1.0, customer.satisfaction_level + 0.05)
                        customer.consecutive_cancellations = 0
                        
                        # win-back: if customer was churned but completed order, reduce churn
                        if customer.is_churned:
                            customer.days_since_churn += 1
                            if customer.days_since_churn >= 7:
                                # win-back after 7 days of good behavior
                                customer.is_churned = False
                                customer.days_since_churn = 0
        
        # store daily KPIs (including enhanced metrics)
        daily_kpis = {
            'day': day,
            'total_customers': day_results['total_customers'],
            'total_completed_orders': day_results['total_completed_orders'],
            'total_cancellations': day_results['total_cancellations'],
            'total_revenue': day_results['total_revenue'],
            'total_waste': day_results['total_waste'],
            'total_waste_monetary': day_results.get('total_waste_monetary', 0.0),
            'customer_satisfaction': day_results.get('customer_satisfaction', 0.0),
            'conversion_rate': day_results.get('conversion_rate', 0.0),
            # enhanced metrics
            'profit_margin': day_results.get('profit_margin_proxy', 0.0),
            'revenue_per_customer': day_results.get('revenue_per_customer', 0.0),
            'avg_store_accuracy': day_results.get('avg_store_accuracy', 0.0),
            # peak supply-demand balance metrics
            'peak_fulfillment_rate': peak_fulfillment_rate,
            'peak_cancellation_rate': peak_cancellation_rate,
            'peak_waste_rate': peak_waste_rate,
            'peak_supply_demand_ratio': peak_supply_demand_ratio,
            'offpeak_fulfillment_rate': offpeak_fulfillment_rate,
            'offpeak_cancellation_rate': offpeak_cancellation_rate,
            'peak_reservations': peak_total_reservations,
            'offpeak_reservations': offpeak_total_reservations,
            # segment-specific metrics
            'segment_orders': segment_orders,
            'segment_revenue': segment_revenue,
            'segment_cancellations': segment_cancellations,
            'segment_customers': {seg: len(customers) for seg, customers in segment_customers.items()},
            # promotion metrics
            'active_promotions': sum(1 for s in stores_copy if hasattr(s, 'active_promotion') and s.active_promotion),
            'promotion_revenue': sum(s.price * s.completed_order_count for s in stores_copy 
                                    if hasattr(s, 'active_promotion') and s.active_promotion),
            # network effects metrics
            'viral_stores': sum(1 for s in stores_copy if hasattr(s, 'recent_orders_count') and s.recent_orders_count > 5),
            'total_social_influence': sum(sum(c.social_influence_score.values()) for c in customers_copy if hasattr(c, 'social_influence_score')),
            # competition dynamics (daily)
            'market_concentration_hhi': hhi,
            'leader_stores': leader_count,
            'follower_stores': follower_count,
            'avg_market_share': avg_market_share_day
        }
        daily_kpis_list.append(daily_kpis)
    
    # Restore original select_stores
    ranking_algorithm.select_stores = original_select_stores
    
    # calculate customer lifetime value metrics
    total_clv = sum(c.calculate_clv(num_days) for c in customers_copy)
    avg_clv = total_clv / len(customers_copy) if customers_copy else 0.0
    
    # calculate CLV by segment
    clv_by_segment = {}
    segment_counts = {}
    for customer in customers_copy:
        segment = customer.customer_segment
        clv = customer.calculate_clv(num_days)
        clv_by_segment[segment] = clv_by_segment.get(segment, 0.0) + clv
        segment_counts[segment] = segment_counts.get(segment, 0) + 1
    
    avg_clv_by_segment = {seg: clv_by_segment[seg] / segment_counts[seg] 
                          for seg in clv_by_segment if segment_counts[seg] > 0}
    
    # calculate retention metrics
    total_customers = len(customers_copy)
    active_customers = sum(1 for c in customers_copy if c.total_orders > 0)
    churned_customers = sum(1 for c in customers_copy if c.is_churned)
    
    # retention rate: customers who made at least one purchase in last 7 days
    # vectorized calculation for performance
    if customers_copy:
        retention_rates = np.array([c.calculate_retention_rate(num_days, 7) for c in customers_copy])
        retention_rate_7d = np.mean(retention_rates)
    else:
        retention_rate_7d = 0.0
    
    # churn rate: % of customers who are churned
    churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0.0
    
    # customer acquisition: new customers (first order in simulation period)
    new_customers = sum(1 for c in customers_copy if c.first_order_day is not None and c.first_order_day <= 3)
    
    # average customer lifespan (days from first to last order)
    customer_lifespans = [c.last_order_day - c.first_order_day + 1 
                          for c in customers_copy if c.first_order_day is not None and c.last_order_day is not None]
    avg_customer_lifespan = np.mean(customer_lifespans) if customer_lifespans else 0.0
    
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
        # enhanced metrics averages
        'profit_margin': np.mean([k.get('profit_margin', 0.0) for k in daily_kpis_list]),
        'revenue_per_customer': np.mean([k.get('revenue_per_customer', 0.0) for k in daily_kpis_list]),
        'avg_store_accuracy': np.mean([k.get('avg_store_accuracy', 0.0) for k in daily_kpis_list]),
        # peak supply-demand balance metrics (averages)
        'avg_peak_fulfillment_rate': np.mean([k.get('peak_fulfillment_rate', 0.0) for k in daily_kpis_list]),
        'avg_peak_cancellation_rate': np.mean([k.get('peak_cancellation_rate', 0.0) for k in daily_kpis_list]),
        'avg_peak_waste_rate': np.mean([k.get('peak_waste_rate', 0.0) for k in daily_kpis_list]),
        'avg_peak_supply_demand_ratio': np.mean([k.get('peak_supply_demand_ratio', 0.0) for k in daily_kpis_list]),
        'avg_offpeak_fulfillment_rate': np.mean([k.get('offpeak_fulfillment_rate', 0.0) for k in daily_kpis_list]),
        'avg_offpeak_cancellation_rate': np.mean([k.get('offpeak_cancellation_rate', 0.0) for k in daily_kpis_list]),
        # customer lifetime value metrics
        'avg_clv': avg_clv,
        'total_clv': total_clv,
        'clv_by_segment': avg_clv_by_segment,
        # price elasticity tracking (aggregate across all customers)
        'price_elasticity_avg': np.mean([PRICE_ELASTICITY.get(c.customer_segment, -1.0) for c in customers_copy]) if customers_copy else 0.0,
        # segment-specific performance metrics (averaged across days)
        'segment_avg_orders': {seg: np.mean([k.get('segment_orders', {}).get(seg, 0) for k in daily_kpis_list]) 
                               for seg in ['PRICE_SENSITIVE', 'QUALITY_FOCUSED', 'LOYAL', 'EXPLORER', 'CONVENIENCE_SEEKER', 'BALANCED']},
        'segment_avg_revenue': {seg: np.mean([k.get('segment_revenue', {}).get(seg, 0.0) for k in daily_kpis_list]) 
                                for seg in ['PRICE_SENSITIVE', 'QUALITY_FOCUSED', 'LOYAL', 'EXPLORER', 'CONVENIENCE_SEEKER', 'BALANCED']},
        'segment_avg_cancellations': {seg: np.mean([k.get('segment_cancellations', {}).get(seg, 0) for k in daily_kpis_list]) 
                                       for seg in ['PRICE_SENSITIVE', 'QUALITY_FOCUSED', 'LOYAL', 'EXPLORER', 'CONVENIENCE_SEEKER', 'BALANCED']},
        'segment_avg_customers': {seg: np.mean([k.get('segment_customers', {}).get(seg, 0) for k in daily_kpis_list]) 
                                  for seg in ['PRICE_SENSITIVE', 'QUALITY_FOCUSED', 'LOYAL', 'EXPLORER', 'CONVENIENCE_SEEKER', 'BALANCED']},
        # retention metrics
        'retention_rate_7d': retention_rate_7d,
        'churn_rate': churn_rate,
        'active_customers': active_customers,
        'churned_customers': churned_customers,
        'new_customers': new_customers,
        'avg_customer_lifespan': avg_customer_lifespan,
        'total_days': num_days
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
                      output_dir: str = "simulation_results", num_days: int = 10) -> Dict:
    """
    Run multi-day simulation with Greedy (baseline) and Near-Optimal strategies and compare KPIs.
    
    Args:
        num_days: Number of days to simulate (default: 10)
        num_stores: Target number of stores (if CSV provided, will generate additional if needed)
        num_customers: Target number of customers (if CSV provided, will generate additional if needed)
    
    Returns:
        Dict with:
        - greedy_results: Results from GreedyStrategy (baseline)
        - near_optimal_results: Results from NearOptimalStrategy
        - comparison: Side-by-side KPI comparison
    """
    import pandas as pd
    import time
    from ranking_algorithm import GreedyStrategy, NearOptimalStrategy, RWES_T_Strategy_Wrapper, Anan_Strategy, Yomna_Strategy
    
    # Use time-based seed if not provided (ensures different results each run)
    if seed is None:
        seed = int(time.time() * 1000) % 1000000  # Use milliseconds for seed
    
    if verbose:
        print("="*90)
        print("COMPARING RANKING STRATEGIES (5-WAY)")
        print(f"Random Seed: {seed}, Days: {num_days}")
        print("="*90)
        print()
    
    # Load or generate stores
    stores = []
    if stores_csv and os.path.exists(stores_csv):
        if verbose:
            print(f"Loading stores from {stores_csv}...")
        stores = load_stores_from_csv(stores_csv)
        csv_num_stores = len(stores)
        
        # Generate additional stores if needed
        if num_stores > csv_num_stores:
            additional_stores_needed = num_stores - csv_num_stores
            if verbose:
                print(f"  Loaded {csv_num_stores} stores from CSV, generating {additional_stores_needed} more...")
            # Get max store ID to avoid conflicts
            max_store_id = max([s.restaurant_id for s in stores]) if stores else 0
            additional_stores = load_store_data(additional_stores_needed, num_customers=num_customers, seed=seed)
            # Reassign IDs to avoid conflicts
            for i, store in enumerate(additional_stores):
                store.restaurant_id = max_store_id + 1 + i
            stores.extend(additional_stores)
        elif verbose:
            print(f"  Loaded {csv_num_stores} stores from CSV")
    else:
        if verbose:
            print(f"Generating {num_stores} stores...")
    stores = load_store_data(num_stores, num_customers=num_customers, seed=seed)
    
    num_stores = len(stores)
    
    # Load or generate customers
    customers = []
    if customers_csv and os.path.exists(customers_csv):
        if verbose:
            print(f"Loading customers from {customers_csv}...")
        # First, read CSV to get actual row count before generating arrival times
        # This ensures we generate enough arrival times for all CSV customers
        import pandas as pd
        df = pd.read_csv(customers_csv)
        csv_actual_count = len(df)
        
        # Generate arrival times based on actual CSV size (not target num_customers)
        # This prevents extra CSV customers from all getting the same arrival time
        arrival_times = sorted(np.random.uniform(0, duration, csv_actual_count))
        customers = load_customers_from_csv(customers_csv, arrival_times, seed=seed)
        csv_num_customers = len(customers)
        
        # Generate additional customers if needed
        if num_customers > csv_num_customers:
            additional_customers_needed = num_customers - csv_num_customers
            if verbose:
                print(f"  Loaded {csv_num_customers} customers from CSV, generating {additional_customers_needed} more...")
            # Get max customer ID to avoid conflicts
            max_customer_id = max([c.customer_id for c in customers]) if customers else 0
            additional_arrival_times = sorted(np.random.uniform(0, duration, additional_customers_needed))
            additional_customers = generate_customer(additional_customers_needed, additional_arrival_times, seed=seed)
            # Reassign IDs to avoid conflicts
            for i, customer in enumerate(additional_customers):
                customer.customer_id = max_customer_id + 1 + i
            customers.extend(additional_customers)
        elif verbose:
            print(f"  Loaded {csv_num_customers} customers from CSV")
        
        # Trim to exact number if we have more than requested
        if len(customers) > num_customers:
            customers = customers[:num_customers]
    else:
        if verbose:
            print(f"Generating {num_customers} customers...")
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
    
    # run strategies (can be parallelized for better performance)
    # note: anan strategy needs original customers/stores, so we handle it separately
    use_parallel = False  # set to True to enable parallel execution (requires picklable strategies)
    
    if use_parallel and cpu_count() > 1:
        # parallel execution (for strategies that can be pickled)
        def run_strategy_wrapper(args):
            strategy, name, stores, customers, n, duration, seed, output_dir, verbose, num_days = args
            return run_single_strategy_simulation(
                strategy, name, stores, customers, n, duration, seed, output_dir, verbose, num_days
            )
        
        strategy_configs = [
            (GreedyStrategy(), "Greedy", stores, customers, n_value, duration, seed, output_dir, verbose, num_days),
            (NearOptimalStrategy(exploration_rate=0.03), "NearOptimal", stores, customers, n_value, duration, seed, output_dir, verbose, num_days),
            (RWES_T_Strategy_Wrapper(), "RWES_T", stores, customers, n_value, duration, seed, output_dir, verbose, num_days),
            (Yomna_Strategy(), "Yomna_St", stores, customers, n_value, duration, seed, output_dir, verbose, num_days),
        ]
        
        with Pool(min(4, cpu_count())) as pool:
            results = pool.map(run_strategy_wrapper, strategy_configs)
        
        greedy_results, near_optimal_results, rwes_t_results, yomna_results = results
    else:
        # sequential execution (default, more reliable)
        if verbose:
            print("Running Greedy Strategy (Baseline)...")
    greedy_strategy = GreedyStrategy()
    greedy_results = run_single_strategy_simulation(
            greedy_strategy, "Greedy", stores, customers, n_value, duration, seed, output_dir, verbose, num_days
    )
    
    if verbose:
        print("\nRunning Near-Optimal Strategy...")
    near_optimal_strategy = NearOptimalStrategy(exploration_rate=0.03)
    near_optimal_results = run_single_strategy_simulation(
            near_optimal_strategy, "NearOptimal", stores, customers, n_value, duration, seed, output_dir, verbose, num_days
    )

    if verbose:
        print("\nRunning RWES_T Strategy...")
    rwes_t_strategy = RWES_T_Strategy_Wrapper()
    rwes_t_results = run_single_strategy_simulation(
            rwes_t_strategy, "RWES_T", stores, customers, n_value, duration, seed, output_dir, verbose, num_days
    )

    if verbose:
        print("\nRunning Yomna Strategy (New)...")
    yomna_strategy = Yomna_Strategy()
    yomna_results = run_single_strategy_simulation(
            yomna_strategy, "Yomna_St", stores, customers, n_value, duration, seed, output_dir, verbose, num_days
        )
    
    # anan strategy needs original customers/stores (not parallelizable easily)
    if verbose:
        print("\nRunning Anan Strategy (New)...")
    anan_strategy = Anan_Strategy(customers, stores)
    anan_results = run_single_strategy_simulation(
        anan_strategy, "Anan_St", stores, customers, n_value, duration, seed, output_dir, verbose, num_days
    )
    
    # Calculate metrics for comparison
    def calc_metrics(results):
        kpis = results['average_kpis']
        return [
            kpis['avg_total_customers'],
            kpis['avg_total_completed_orders'],
            kpis['avg_total_cancellations'],
            kpis['avg_total_revenue'],
            kpis['avg_total_waste'],
            (kpis['avg_total_completed_orders'] / (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) * 100) if (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) > 0 else 0,
            (kpis['avg_total_cancellations'] / (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) * 100) if (kpis['avg_total_completed_orders'] + kpis['avg_total_cancellations']) > 0 else 0,
            # Enhanced Metrics
            kpis.get('profit_margin', 0.0),
            kpis.get('revenue_per_customer', 0.0),
            kpis.get('avg_store_accuracy', 0.0),
            # Peak Supply-Demand Balance Metrics
            kpis.get('avg_peak_fulfillment_rate', 0.0),
            kpis.get('avg_peak_cancellation_rate', 0.0),
            kpis.get('avg_peak_supply_demand_ratio', 0.0),
            kpis.get('avg_offpeak_fulfillment_rate', 0.0)
        ]
    
    comparison = {
        'metric': [
            'Average Customers per Day',
            'Average Completed Orders per Day',
            'Average Cancellations per Day (raw count)',
            'Average Revenue per Day',
            'Average Waste (units) per Day',
            'Fulfillment Rate (%)',
            'Cancellation Rate (%)',
            'Profit Margin (%)',
            'Revenue per Customer',
            'Avg Store Accuracy',
            'Peak Fulfillment Rate (%)',
            'Peak Cancellation Rate (%)',
            'Peak Supply-Demand Ratio',
            'Off-Peak Fulfillment Rate (%)'
        ],
        'greedy': calc_metrics(greedy_results),
        'near_optimal': calc_metrics(near_optimal_results),
        'rwes_t': calc_metrics(rwes_t_results),
        'anan': calc_metrics(anan_results),
        'yomna': calc_metrics(yomna_results)
    }
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save comparison to CSV
    comparison_df = pd.DataFrame(comparison)
    comparison_file = os.path.join(output_dir, "strategy_comparison.csv")
    comparison_df.to_csv(comparison_file, index=False)
    
    # Save individual strategy KPIs
    for name, res in [('greedy', greedy_results), ('near_optimal', near_optimal_results), 
                      ('rwes_t', rwes_t_results), ('anan', anan_results), ('yomna', yomna_results)]:
        df = pd.DataFrame([res['average_kpis']])
        df.to_csv(os.path.join(output_dir, f"{name}_strategy_kpis.csv"), index=False)
    
    greedy_file = os.path.join(output_dir, "greedy_strategy_kpis.csv")
    near_optimal_file = os.path.join(output_dir, "near_optimal_strategy_kpis.csv")
    rwes_t_file = os.path.join(output_dir, "rwes_t_strategy_kpis.csv")
    
    if verbose:
        print(f"\n{'='*130}")
        print("STRATEGY COMPARISON COMPLETE")
        print(f"{'='*130}")
        print(f"\nKPI COMPARISON:")
        
        # Header
        header = f"{'Metric':<40} {'Greedy':<12} {'Near-Opt':<12} {'RWES_T':<12} {'Anan':<12} {'Yomna':<12} {'Best':<10}"
        print(header)
        print("-" * len(header))
        
        for i, metric in enumerate(comparison['metric']):
            vals = {
                'Greedy': comparison['greedy'][i],
                'Near-Opt': comparison['near_optimal'][i],
                'RWES_T': comparison['rwes_t'][i],
                'Anan': comparison['anan'][i],
                'Yomna': comparison['yomna'][i]
            }
            
            # Identify best
            # Lower is better: Cancellations, Waste, Cancellation Rate (including peak cancellation rate)
            if 'Cancellation' in metric or ('Waste' in metric and 'Rate' not in metric):
                best_val = min(vals.values())
            elif 'Supply-Demand Ratio' in metric:
                # For supply-demand ratio, closer to 1.0 is better (balanced supply and demand)
                best_val = min(vals.values(), key=lambda x: abs(x - 1.0))
            else:
                # Higher is better: Revenue, Completed Orders, Fulfillment, Margin, Accuracy (including peak fulfillment)
                best_val = max(vals.values())
            
            best_strategies = [k for k, v in vals.items() if abs(v - best_val) < 0.001]
            best_str = best_strategies[0] if len(best_strategies) == 1 else "Tie"
            
            # Format
            def fmt(v):
                if 'Revenue' in metric or 'Customer' in metric and '$' in metric: return f"${v:.2f}"
                if 'Rate' in metric or 'Margin' in metric or '%' in metric: return f"{v:.2f}%"
                if 'Accuracy' in metric: return f"{v:.3f}"
                if 'Ratio' in metric: return f"{v:.3f}"
                return f"{v:.1f}"
                
            row = f"{metric:<40} {fmt(vals['Greedy']):<12} {fmt(vals['Near-Opt']):<12} {fmt(vals['RWES_T']):<12} {fmt(vals['Anan']):<12} {fmt(vals['Yomna']):<12} {best_str:<10}"
            print(row)
            
        
        print(f"\nOutput Files:")
        print(f"  1. Strategy Comparison: {comparison_file}")
        print(f"  2. Greedy Strategy KPIs: {greedy_file}")
        print(f"  3. Near-Optimal Strategy KPIs: {near_optimal_file}")
        print(f"  4. RWES_T Strategy KPIs: {rwes_t_file}")
        print(f"  5. Anan Strategy KPIs: {os.path.join(output_dir, 'anan_strategy_kpis.csv')}")
        print(f"  6. Yomna Strategy KPIs: {os.path.join(output_dir, 'yomna_strategy_kpis.csv')}")
    
        print(f"{'='*130}\n")
    
    return {
        'greedy_results': greedy_results,
        'near_optimal_results': near_optimal_results,
        'rwes_t_results': rwes_t_results,
        'anan_results': anan_results,
        'yomna_results': yomna_results,
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
        # Dictionary preserves insertion order, so order = arrival order
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

