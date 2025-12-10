# -*- coding: utf-8 -*-
"""
Restaurant API Module
Handles all restaurant/store-related functionality including:
- Restaurant class definition
- Store data loading and management
- Inventory tracking and accuracy calculation
- End-of-day processing
"""

import numpy as np
from typing import List, Dict, Optional

class Restaurant:
    """Represents a restaurant/store in the marketplace"""
    
    def __init__(self, restaurant_id, name, category):
        self.restaurant_id = restaurant_id
        self.name = name
        self.category = category
        self.price = 0.0

        self.rating = 0.0
        self.number_of_ratings = 0

        self.est_inventory = 0
        self.actual_inventory = 0

        self.accuracy_score = 1.0
        self.inventory_history = []  # list of (date, est, actual)

        self.reservation_count = 0
        self.completed_order_count = 0
        self.cancellation_count = 0

        self.exposure_count = 0

    def update_daily_supply(self, new_price, new_inventory):
        """Update daily supply and price"""
        self.price = new_price
        self.est_inventory = new_inventory

    def receive_rating(self, rating):
        """Update rating with new customer rating"""
        n = self.number_of_ratings
        new_rating = self.rating + (rating - self.rating) / (n + 1)
        self.number_of_ratings = n + 1
        self.rating = new_rating

    def reserve_order(self):
        """Reserve an order if inventory available"""
        if self.reservation_count < self.est_inventory:
            self.reservation_count += 1

    def finalize_daily_inventory(self, date, actual_inventory):
        """Finalize actual inventory at end of day"""
        self.actual_inventory = actual_inventory
        self.inventory_history.append((date, self.est_inventory, actual_inventory))
        self.calculate_accuracy()

    def add_daily_summary(self, ordered, received):
        """Add daily summary of orders and cancellations"""
        self.completed_order_count += received
        self.cancellation_count += ordered - received

    def calculate_accuracy(self):
        """Calculate inventory estimation accuracy based on history"""
        if not self.inventory_history:
            return 1.0  # no orders so far. accuracy = 1

        errors = []
        for record in self.inventory_history:
            date, est, actual = record
            if est > 0:
                error = max(est - actual, 0) / est
                errors.append(error)
        if not errors:
            return 1.0

        errors_array = np.array(errors)
        errors_mean = np.mean(errors_array)
        # if we have less than 30 records, do not use window frame.
        if len(errors) <= 30:
            accuracy = 1 - errors_mean
            self.accuracy_score = accuracy
            return accuracy

        window_errors = errors[-30:]
        window_errors_mean = np.mean(window_errors)
        accuracy = 1 - (0.7 * window_errors_mean + 0.3 * errors_mean)
        self.accuracy_score = accuracy
        return accuracy

    def reset_daily_counters(self):
        """Reset daily counters for a new day"""
        self.reservation_count = 0
        self.completed_order_count = 0
        self.cancellation_count = 0
        self.exposure_count = 0
        self.actual_inventory = 0


# Global store registry
_stores: Dict[int, Restaurant] = {}
_actual_inventories_for_day: Optional[Dict[int, int]] = None


def load_store_data(num_stores: int = 10, num_customers: int = 100, seed: int = 42) -> List[Restaurant]:
    """Load or generate store data"""
    global _stores
    _stores = {}
    categories = ["Bakery", "Cafe", "Restaurant", "Grocery", "Deli", "Pizza", "Sushi", "Fast Food"]
    
    np.random.seed(seed)

    # Scale inventory based on customer volume
    expected_purchases = int(num_customers * 0.4)  # Expect 40% conversion
    avg_inventory_per_store = max(3, int(expected_purchases / num_stores * 1.3))  # 30% safety margin
    
    for i in range(num_stores):
        store_id = i + 1
        name = f"Store_{store_id}"
        category = np.random.choice(categories)

        restaurant = Restaurant(store_id, name, category)

        # Initialize with WIDE variation to create distinct stores
        # This ensures strategies have meaningful differences to work with
        restaurant.price = np.random.uniform(3.0, 50.0)  # Much wider price range
        restaurant.rating = np.random.uniform(2.0, 5.0)  # Wider rating range (includes bad stores)
        
        # Scale inventory with customer volume (±50% variation for more diversity)
        min_inv = max(2, int(avg_inventory_per_store * 0.5))
        max_inv = int(avg_inventory_per_store * 1.5)
        restaurant.est_inventory = np.random.randint(min_inv, max_inv + 1)
        
        # Wider accuracy range - some stores are very inaccurate
        restaurant.accuracy_score = np.random.uniform(0.5, 1.0)  # Some stores are only 50% accurate

        # Initialize some historical data
        for day in range(np.random.randint(5, 20)):
            est = np.random.randint(min_inv, max_inv + 1)
            actual = int(est * np.random.uniform(0.7, 1.1))  # actual within 70-110% of estimate
            restaurant.inventory_history.append((day, est, actual))

        restaurant.calculate_accuracy()

        _stores[store_id] = restaurant

    return list(_stores.values())


def get_all_stores() -> List[Restaurant]:
    """Return all stores"""
    return list(_stores.values())


def get_store_info(store_id: int) -> Optional[Restaurant]:
    """Return store info by ID"""
    return _stores.get(store_id)


def update_reservation(store_id: int, count: int = 1) -> bool:
    """
    Update reservation count for a store.
    Only allows reservations up to estimated inventory capacity.
    """
    if store_id in _stores:
        store = _stores[store_id]
        # Only allow reservations up to estimated inventory
        # This prevents overbooking beyond what the store estimated
        max_reservations = store.est_inventory
        if store.reservation_count < max_reservations:
            # Allow reservation if under capacity
            available_slots = max_reservations - store.reservation_count
            actual_count = min(count, available_slots)
            store.reservation_count += actual_count
            return True
        # Store is at capacity, cannot accept more reservations
        return False
    return False


def update_exposure(store_id: int) -> bool:
    """Update exposure count for a store"""
    if store_id in _stores:
        _stores[store_id].exposure_count += 1
        return True
    return False


def get_store_metrics(store_id: int) -> Optional[Dict]:
    """Get store metrics as a dictionary"""
    if store_id not in _stores:
        return None

    store = _stores[store_id]
    return {
        'store_id': store.restaurant_id,
        'name': store.name,
        'category': store.category,
        'rating': store.rating,
        'price': store.price,
        'est_inventory': store.est_inventory,
        'actual_inventory': store.actual_inventory,
        'accuracy_score': store.accuracy_score,
        'reservation_count': store.reservation_count,
        'exposure_count': store.exposure_count,
        'completed_order_count': store.completed_order_count,
        'cancellation_count': store.cancellation_count
    }


def initialize_day(stores: List[Restaurant], actual_inventories: Optional[Dict[int, int]] = None):
    """Initialize a new day"""
    global _actual_inventories_for_day
    _actual_inventories_for_day = actual_inventories

    for store in stores:
        store.reset_daily_counters()
        # actual_inventory remains 0 until end of day when it's revealed

def calculate_gini_coefficient(values: List[float]) -> float:
    """
    Calculate Gini coefficient for inequality measurement.
    Returns value between 0 (perfect equality) and 1 (maximum inequality).
    Used for measuring store exposure and revenue distribution fairness.
    """
    if not values or len(values) == 0:
        return 0.0
    
    # remove zeros and sort
    non_zero_values = [v for v in values if v > 0]
    if len(non_zero_values) < 2:
        return 0.0
    
    n = len(non_zero_values)
    sorted_values = sorted(non_zero_values)
    
    # Cclculate Gini coefficient using the formula
    cumsum = 0
    for i, value in enumerate(sorted_values):
        cumsum += (i + 1) * value
    
    gini = (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def calculate_shannon_entropy(proportions: List[float]) -> float:
    """
    calculate Shannon entropy for diversity measurement.
    Higher values indicate more diversity.
    used for measuring store revenue distribution diversity.
    """
    if not proportions or len(proportions) == 0:
        return 0.0
    
    # Normalize to probabilities
    total = sum(proportions)
    if total == 0:
        return 0.0
    
    probs = [p / total for p in proportions if p > 0]
    if not probs:
        return 0.0
    
    # calculate entropy: -sum(p_i * log2(p_i))
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy

def end_of_day_processing_enhanced(marketplace: Optional["Marketplace"] = None, debug: bool = False) -> Dict:
    """
    enhanced end of day processing with comprehensive KPI calculations.
    
    this is an enhanced version of end_of_day_processing() that calculates
    - Efficiency metrics (revenue per customer, revenue per waste unit, etc.)
    - Customer experience metrics (cancellation cost, net revenue, retention)
    - Store fairness metrics (Gini coefficients, diversity index)
    - Inventory accuracy metrics (accuracy-weighted waste, accuracy tiers)
    - Business health metrics (profit margin, inventory turnover, etc.)
    """
    global _actual_inventories_for_day
    print("DEBUG: ENHANCED PROCESSING CALLED inside restaurant_api")
    total_cancellations = 0
    total_waste_units = 0
    total_waste_monetary = 0.0
    total_revenue = 0.0
    total_completed = 0
    total_customers_who_bought = 0
    
    # new tracking variables for enhanced metrics
    total_cancellation_cost = 0.0
    total_inventory = 0
    store_revenues = []
    store_exposures = []
    accuracy_weighted_waste = 0.0
    accuracy_weighted_cancellations = 0.0
    stores_poor_accuracy = 0
    accuracy_scores = []
    cancellations_by_accuracy_tier = {'high': 0, 'medium': 0, 'low': 0}
    orders_by_accuracy_tier = {'high': 0, 'medium': 0, 'low': 0}
    
    # Determine which stores to process
    if marketplace and hasattr(marketplace, 'stores'):
        stores_to_process = marketplace.stores
    else:
        stores_to_process = list(_stores.values())

    for store in stores_to_process:
        # discover the actual inventory (same as original)
        if _actual_inventories_for_day and store.restaurant_id in _actual_inventories_for_day:
            store.actual_inventory = _actual_inventories_for_day[store.restaurant_id]
        else:
            # generate actual inventory based on accuracy
            accuracy_factor = store.accuracy_score
            max_error = 0.5 * (1.0 - accuracy_factor)
            
            if accuracy_factor > 0.8:
                will_underestimate = np.random.uniform() < 0.5
            else:
                will_underestimate = np.random.uniform() < 0.6
            
            if will_underestimate:
                min_pct = max(0.5, 1.0 - max_error)
                max_pct = 0.95
                factor = np.random.uniform(min_pct, max_pct)
            else:
                min_pct = 1.0
                max_pct = 1.0 + max_error
                factor = np.random.uniform(min_pct, max_pct)
            
            # For simulated stores, est_inventory might be updated by initialize_day
            actual = int(store.est_inventory * factor)
            store.actual_inventory = max(0, actual)
        
        # calculate cancellations
        cancellations = max(0, store.reservation_count - store.actual_inventory)
        store.cancellation_count = cancellations
        total_cancellations += cancellations
        
        # calculate actual sales
        actual_sales = min(store.reservation_count, store.actual_inventory)
        store.completed_order_count = actual_sales
        total_completed += actual_sales
        total_customers_who_bought += actual_sales
        
        # calculate waste
        waste_units = max(0, store.actual_inventory - store.reservation_count)
        waste_monetary = waste_units * store.price
        total_waste_units += waste_units
        total_waste_monetary += waste_monetary
        
        # calculate cancellation cost (revenue lost)
        cancellation_cost = cancellations * store.price
        total_cancellation_cost += cancellation_cost
        
        # track accuracy metrics
        accuracy_scores.append(store.accuracy_score)
        if store.accuracy_score < 0.7:
            stores_poor_accuracy += 1
        
        # categorize by accuracy tier
        if store.accuracy_score > 0.8:
            accuracy_tier = 'high'
        elif store.accuracy_score >= 0.6:
            accuracy_tier = 'medium'
        else:
            accuracy_tier = 'low'
        
        cancellations_by_accuracy_tier[accuracy_tier] += cancellations
        orders_by_accuracy_tier[accuracy_tier] += store.completed_order_count
        
        # calculate accuracy-weighted metrics
        inaccuracy = 1.0 - store.accuracy_score
        accuracy_weighted_waste += waste_units * inaccuracy
        accuracy_weighted_cancellations += cancellations * inaccuracy
        
        # calculate revenue
        revenue = actual_sales * store.price
        total_revenue += revenue
        store_revenues.append(revenue)
        store_exposures.append(store.exposure_count)
        total_inventory += store.actual_inventory
        
        # update inventory history
        store.inventory_history.append((
            len(store.inventory_history),
            store.est_inventory,
            store.actual_inventory
        ))
        
        # recalculate accuracy
        store.calculate_accuracy()
        store.add_daily_summary(store.reservation_count, actual_sales)
    
    # calculate customer satisfaction (improved formula)
    total_customers = marketplace.total_customers_seen if marketplace else 0
    customers_who_left = total_customers - total_customers_who_bought
    no_purchase_count = customers_who_left - total_cancellations
    
    # improved satisfaction: weighted formula
    satisfaction_score = (total_completed * 1.0 - total_cancellations * 2.0 - no_purchase_count * 0.3)
    true_satisfaction = satisfaction_score / total_customers if total_customers > 0 else 0
    true_satisfaction = max(0.0, min(1.0, true_satisfaction))
    
    # conversion rate
    conversion_rate = (total_customers_who_bought / total_customers * 100) if total_customers > 0 else 0
    conversion_rate = min(100.0, conversion_rate)
    
    # ===== EFFICIENCY METRICS =====
    revenue_per_customer = total_revenue / total_customers if total_customers > 0 else 0.0
    revenue_per_waste_unit = total_revenue / total_waste_units if total_waste_units > 0 else 0.0
    revenue_per_order = total_revenue / total_completed if total_completed > 0 else 0.0
    waste_efficiency_ratio = total_waste_units / total_inventory if total_inventory > 0 else 0.0
    orders_per_customer = total_completed / total_customers if total_customers > 0 else 0.0
    
    # ===== CUSTOMER EXPERIENCE METRICS =====
    net_revenue = total_revenue - total_waste_monetary - total_cancellation_cost
    avg_cancellation_cost = total_cancellation_cost / total_cancellations if total_cancellations > 0 else 0.0
    cancellation_impact_ratio = total_cancellation_cost / total_revenue if total_revenue > 0 else 0.0
    
    # ===== INVENTORY ACCURACY METRICS =====
    avg_store_accuracy = np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    # cancellation rate by accuracy tier
    cancellation_rate_by_tier = {}
    for tier in ['high', 'medium', 'low']:
        total_orders_tier = orders_by_accuracy_tier[tier] + cancellations_by_accuracy_tier[tier]
        cancellation_rate_by_tier[tier] = (cancellations_by_accuracy_tier[tier] / total_orders_tier * 100) if total_orders_tier > 0 else 0.0
    
    # ===== BUSINESS HEALTH METRICS =====
    profit_margin_proxy = (net_revenue / total_revenue * 100) if total_revenue > 0 else 0.0
    inventory_turnover = total_completed / (total_inventory / len(_stores)) if len(_stores) > 0 and total_inventory > 0 else 0.0
    revenue_per_inventory_unit = total_revenue / total_inventory if total_inventory > 0 else 0.0
    
    return {
        # basic metrics (same as original)
        'total_cancellations': total_cancellations,
        'total_waste': total_waste_units,
        'total_waste_monetary': total_waste_monetary,
        'total_revenue': total_revenue,
        'total_completed_orders': total_completed,
        'total_customers': total_customers,
        'total_customers_who_bought': total_customers_who_bought,
        'customer_satisfaction': true_satisfaction,
        'conversion_rate': conversion_rate,
        # efficiency metrics
        'revenue_per_customer': revenue_per_customer,
        'revenue_per_waste_unit': revenue_per_waste_unit,
        'revenue_per_order': revenue_per_order,
        'waste_efficiency_ratio': waste_efficiency_ratio,
        'orders_per_customer': orders_per_customer,
        # customer experience metrics
        'cancellation_cost': total_cancellation_cost,
        'net_revenue': net_revenue,
        'avg_cancellation_cost': avg_cancellation_cost,
        'cancellation_impact_ratio': cancellation_impact_ratio,
        # inventory accuracy metrics
        'accuracy_weighted_waste': accuracy_weighted_waste,
        'accuracy_weighted_cancellations': accuracy_weighted_cancellations,
        'stores_poor_accuracy': stores_poor_accuracy,
        'avg_store_accuracy': avg_store_accuracy,
        'cancellation_rate_by_accuracy_tier': cancellation_rate_by_tier,
        # business health metrics
        'profit_margin_proxy': profit_margin_proxy,
        'inventory_turnover': inventory_turnover,
        'revenue_per_inventory_unit': revenue_per_inventory_unit,
        # store-level data for fairness calculations
        'store_revenues': store_revenues,
        'store_exposures': store_exposures,
        'total_inventory': total_inventory,
        'stores': {store_id: get_store_metrics(store_id) for store_id in _stores.keys()}
    }

    
def end_of_day_processing(marketplace, debug: bool = False) -> Dict:
    """Process end of day: calculate cancellations, waste, and update accuracy"""
    global _actual_inventories_for_day
    total_cancellations = 0
    total_waste_units = 0
    total_waste_monetary = 0.0
    total_revenue = 0.0
    total_completed = 0
    # Track unique customers who successfully completed orders
    # This is needed for accurate conversion rate calculation
    total_customers_who_bought = 0
    # Track unique customers who successfully completed orders
    # This is needed for accurate conversion rate calculation
    total_customers_who_bought = 0
    
    for store in _stores.values():
        # Discover the actual inventory
        if _actual_inventories_for_day and store.restaurant_id in _actual_inventories_for_day:
            store.actual_inventory = _actual_inventories_for_day[store.restaurant_id]
        else:
            # Generate actual inventory based on accuracy
            # More accurate stores have actual inventory closer to estimate
            accuracy_factor = store.accuracy_score
            
            # Base error range: stores with perfect accuracy (1.0) have ±5% error
            # Stores with poor accuracy (0.0) have ±50% error
            max_error = 0.5 * (1.0 - accuracy_factor)  # 0% to 50% error range
            
            # Decide if store overestimated or underestimated
            # More accurate stores are more likely to be close to estimate
            # Less accurate stores can go either way
            if accuracy_factor > 0.8:
                # High accuracy: 50/50 chance of over/under estimate
                will_underestimate = np.random.uniform() < 0.5
            else:
                # Lower accuracy: slight bias toward underestimation (60% chance)
                # This reflects real-world tendency to be conservative
                will_underestimate = np.random.uniform() < 0.6
            
            if will_underestimate:
                # Underestimation: actual < estimate (causes cancellations)
                # Range: (1 - max_error) to 0.95 of estimate
                min_pct = max(0.5, 1.0 - max_error)
                max_pct = 0.95
                factor = np.random.uniform(min_pct, max_pct)
            else:
                # Overestimation: actual > estimate (causes waste)
                # Range: 1.0 to (1 + max_error) of estimate
                min_pct = 1.0
                max_pct = 1.0 + max_error
                factor = np.random.uniform(min_pct, max_pct)
            
            actual = int(store.est_inventory * factor)
            store.actual_inventory = max(0, actual)

        # Calculate cancellations
        # Cancellations occur ONLY when actual inventory is less than reservations
        # This happens when estimated bags > actual bags available
        # Customers cannot cancel their own orders - only the restaurant's inventory shortage causes cancellations
        cancellations = max(0, store.reservation_count - store.actual_inventory)
        store.cancellation_count = cancellations
        total_cancellations += cancellations
        
        if debug and (cancellations > 0 or store.reservation_count > store.est_inventory):
            print(f"  {store.name}: Est={store.est_inventory}, Actual={store.actual_inventory}, Reserved={store.reservation_count}, Cancelled={cancellations}")

        # Calculate actual sales with bag doubling logic
        # If actual inventory > estimated, restaurant can double bag quantity
        if store.actual_inventory > store.est_inventory:
            # Restaurant can double the bag capacity (2x estimated inventory)
            doubled_capacity = 2 * store.est_inventory
            # Can fulfill up to doubled capacity or reservation count, whichever is smaller
            max_fulfillable = min(store.reservation_count, doubled_capacity)
            actual_sales = min(max_fulfillable, store.actual_inventory)
        else:
            # Normal case: actual <= estimated
            actual_sales = min(store.reservation_count, store.actual_inventory)
        
        store.completed_order_count = actual_sales
        total_completed += actual_sales
        # Each completed order represents one customer who successfully bought
        total_customers_who_bought += actual_sales

        # Calculate waste with bag doubling logic
        # If actual > estimated, restaurant can double bag capacity
        if store.actual_inventory > store.est_inventory:
            # Doubled capacity = 2 * estimated inventory
            doubled_capacity = 2 * store.est_inventory
            # Maximum that can be used (fulfilled orders)
            max_used = min(store.reservation_count, doubled_capacity, store.actual_inventory)
            # Waste = anything left after using up to doubled capacity
            waste_units = max(0, store.actual_inventory - max_used)
        else:
            # Normal case: waste = actual - reservations (if positive)
            waste_units = max(0, store.actual_inventory - store.reservation_count)
        
        waste_monetary = waste_units * store.price
        total_waste_units += waste_units
        total_waste_monetary += waste_monetary

        # Calculate revenue
        revenue = actual_sales * store.price
        total_revenue += revenue

        # Update inventory history
        store.inventory_history.append((
            len(store.inventory_history),
            store.est_inventory,
            store.actual_inventory
        ))

        # Recalculate accuracy
        store.calculate_accuracy()

        store.add_daily_summary(store.reservation_count, actual_sales)
  
    # Calculate customer satisfaction
    total_customers = marketplace.total_customers_seen
    true_satisfaction = total_completed / total_customers if total_customers > 0 else 0
    
    # Conversion rate: customers who bought / customers who arrived
    # Use total_customers_who_bought (unique customers) instead of total_completed (which might be > customers if there's a bug)
    conversion_rate = (total_customers_who_bought / total_customers * 100) if total_customers > 0 else 0
    conversion_rate = min(100.0, conversion_rate)  # Cap at 100% - cannot exceed

    return {
        'total_cancellations': total_cancellations,
        'total_waste': total_waste_units,
        'total_waste_monetary': total_waste_monetary,
        'total_revenue': total_revenue,
        'total_completed_orders': total_completed,
        'total_customers': total_customers,
        'total_customers_who_bought': total_customers_who_bought,  # New: unique customers who bought
        'customer_satisfaction': true_satisfaction,
        'conversion_rate': conversion_rate,  # Pre-calculated conversion rate
        'stores': {store_id: get_store_metrics(store_id) for store_id in _stores.keys()}
    }