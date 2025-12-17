# -*- coding: utf-8 -*-
"""
Data Loader Module
Handles loading data from CSV files.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from restaurant_api import Restaurant, _stores
from customer_api import Customer


def load_stores_from_csv(csv_path: str = "stores.csv") -> List[Restaurant]:
    """
    Load stores from CSV file.
    
    Expected columns:
    - store_id: Unique integer ID
    - store_name: Store's name
    - branch: Branch name or area
    - average_bags_at_9AM: Estimated number of surprise bags (int)
    - average_overall_rating: Average store rating (1-5)
    - price: Bag price in minor units (EGP)
    - longitude: Store longitude (float)
    - latitude: Store latitude (float)
    
    Returns:
        List of Restaurant objects
    """
    global _stores
    _stores = {}
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Stores CSV file not found: {csv_path}")
    
    # Validate required columns (handle variations in column names)
    required_base_cols = ['store_id', 'store_name', 'branch', 'average_overall_rating', 
                          'price', 'longitude', 'latitude']
    
    # Check for inventory column (might have different names)
    inventory_col = None
    for col in df.columns:
        if 'bag' in col.lower() and ('9am' in col.lower() or '9 am' in col.lower() or 'average' in col.lower()):
            inventory_col = col
            break
    
    if not inventory_col:
        # Try exact match first
        if 'average_bags_at_9AM' in df.columns:
            inventory_col = 'average_bags_at_9AM'
        elif 'average bags at 9AM' in df.columns:
            inventory_col = 'average bags at 9AM'
        else:
            raise ValueError("Could not find inventory column. Expected something like 'average_bags_at_9AM' or 'average bags at 9AM'")
    
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in stores.csv: {missing_cols}")
    
    restaurants = []
    for _, row in df.iterrows():
        store_id = int(row['store_id'])
        name = str(row['store_name'])
        branch = str(row['branch'])
        
        # Use branch as category, or derive from name if needed
        category = branch  # You can modify this logic if needed
        
        restaurant = Restaurant(store_id, name, category)
        
        # Set values from CSV
        restaurant.price = float(row['price'])
        restaurant.rating = float(row['average_overall_rating'])
        restaurant.est_inventory = int(row[inventory_col])
        restaurant.target_daily_inventory = restaurant.est_inventory
        
        # Store location (for potential future use)
        restaurant.longitude = float(row['longitude'])
        restaurant.latitude = float(row['latitude'])
        
        # Initialize accuracy score (will be updated as simulation runs)
        restaurant.accuracy_score = np.random.uniform(0.7, 1.0)
        
        # Initialize some historical data for accuracy calculation
        for day in range(np.random.randint(5, 20)):
            est = restaurant.est_inventory
            actual = int(est * np.random.uniform(0.7, 1.1))
            restaurant.inventory_history.append((day, est, actual))
        
        restaurant.calculate_accuracy()
        
        _stores[store_id] = restaurant
        restaurants.append(restaurant)
    
    return restaurants


def load_customers_from_csv(csv_path: str = "customers.csv", 
                            arrival_times: Optional[List[float]] = None,
                            seed: int = None) -> List[Customer]:
    """
    Load customers from CSV file.
    
    Expected columns:
    - Customer_ID: Unique integer
    - longitude: Customer longitude (float)
    - latitude: Customer latitude (float)
    - store1_id_valuation, store2_id_valuation, ...: Valuation for each store (float)
    
    The valuation columns indicate how much each customer likes/dislikes each store.
    These will be used in the purchase probability calculation.
    
    Args:
        csv_path: Path to customers CSV file
        arrival_times: Optional list of arrival times (if None, will generate)
        seed: Random seed for reproducibility
    
    Returns:
        List of Customer objects with preference_ratings populated from CSV
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Customers CSV file not found: {csv_path}")
    
    # Validate required columns
    if 'Customer_ID' not in df.columns:
        raise ValueError("Missing required column: Customer_ID")
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        raise ValueError("Missing required columns: longitude and/or latitude")
    
    # Find all valuation columns (store*_id_valuation or store*_valuation)
    # Handle variations: "store1_id_valuation", "store1_valuation", "storel_id_valuation" (typo)
    valuation_cols = [col for col in df.columns if 'valuation' in col.lower()]
    
    if not valuation_cols:
        raise ValueError("No valuation columns found in customers.csv. Expected columns like 'store1_id_valuation', 'store2_id_valuation', etc.")
    
    # Extract store IDs from column names
    # Format: "store1_id_valuation", "store1_valuation", "storel_id_valuation" -> store_id = 1
    store_valuations = {}  # {store_id: column_name}
    import re
    for col in valuation_cols:
        # Try to extract store ID from column name
        # Handle formats like "store1_id_valuation", "store100_valuation", "storel_id_valuation" (typo)
        # Match "store" followed by digits (or 'l' which might be typo for '1')
        match = re.search(r'store([\d]+|l)', col.lower())
        if match:
            store_id_str = match.group(1)
            # Handle typo: 'l' might be '1'
            if store_id_str == 'l':
                store_id = 1
            else:
                store_id = int(store_id_str)
            store_valuations[store_id] = col
    
    if not store_valuations:
        raise ValueError("Could not parse store IDs from valuation column names")
    
    # Generate arrival times if not provided
    num_customers = len(df)
    if arrival_times is None:
        arrival_times = sorted(np.random.uniform(0, 24, num_customers))
    
    customers = []
    for idx, row in df.iterrows():
        customer_id = int(row['Customer_ID'])
        longitude = float(row['longitude'])
        latitude = float(row['latitude'])
        
        # Create customer
        customer = Customer(customer_id)
        customer.longitude = longitude
        customer.latitude = latitude
        
        # Build preference_ratings dictionary from CSV valuations
        # The valuations represent how much this customer likes each store
        preference_ratings = {}
        for store_id, col_name in store_valuations.items():
            if col_name in row and pd.notna(row[col_name]):
                # Normalize valuation to [0, 1] range if needed
                # Assuming valuations are already in reasonable range (e.g., 0-1 or 0-10)
                valuation = float(row[col_name])
                # If valuations are in 0-10 range, normalize to 0-1
                if valuation > 1.0:
                    valuation = valuation / 10.0
                preference_ratings[store_id] = max(0.0, min(1.0, valuation))
            else:
                # Default valuation if missing
                preference_ratings[store_id] = 0.5
        
        # Store preference ratings in customer object
        customer.preference_ratings = preference_ratings
        
        # Set arrival time
        customer.arrival_time = arrival_times[idx] if idx < len(arrival_times) else arrival_times[-1]
        
        customers.append(customer)
    
    return customers


def load_data_from_csv(stores_csv: str = "stores.csv", 
                       customers_csv: str = "customers.csv",
                       arrival_times: Optional[List[float]] = None,
                       seed: int = 42) -> tuple[List[Restaurant], List[Customer]]:
    """
    Load both stores and customers from CSV files.
    
    Returns:
        Tuple of (stores, customers)
    """

def save_stores_to_csv(stores: List[Restaurant], filepath: str = "generated_stores.csv") -> None:
    """
    Save list of stores to CSV in the format expected by load_stores_from_csv.
    """
    print("DEBUG: save_stores_to_csv called with rounding")
    data = []
    for store in stores:
        data.append({
            'store_id': store.restaurant_id,
            'store_name': store.name,
            'branch': store.category, # Using category as branch
            'average_bags_at_9AM': store.est_inventory,
            'average_overall_rating': round(store.rating, 5),
            'price': round(store.price, 5),
            'longitude': round(getattr(store, 'longitude', 0.0) or 0.0, 5),
            'latitude': round(getattr(store, 'latitude', 0.0) or 0.0, 5)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(stores)} stores to {filepath}")


def save_customers_to_csv(customers: List[Customer], stores: List[Restaurant], filepath: str = "generated_customers.csv") -> None:
    """
    Save list of customers to CSV in the format expected by load_customers_from_csv.
    """
    data = []
    for customer in customers:
        row = {
            'Customer_ID': customer.customer_id,
            'longitude': round(getattr(customer, 'longitude', 0.0) or 0.0, 5),
            'latitude': round(getattr(customer, 'latitude', 0.0) or 0.0, 5)
        }
        
        # valid_valuations count check
        valid_valuations = 0
        
        for store in stores:
            col_name = f"store{store.restaurant_id}_valuation"
            
            # Use existing preference if available
            if hasattr(customer, 'preference_ratings') and store.restaurant_id in customer.preference_ratings:
                val = customer.preference_ratings[store.restaurant_id]
            else:
                val = np.random.uniform(0.0, 1.0) 
            
            row[col_name] = round(val, 5)
            valid_valuations += 1
            
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(customers)} customers to {filepath}")

