# -*- coding: utf-8 -*-
"""
Customer API Module
Handles all customer-related functionality including:
- Customer class definition
- Customer generation
- Purchase probability calculation
- Store selection and decision making
"""

import numpy as np
from typing import List, Dict, Optional
from scipy.special import softmax
from restaurant_api import Restaurant


class Customer:
    """Represents a customer in the marketplace"""
    
    def __init__(self, customer_id: int, name: str = None):
        self.customer_id = customer_id
        self.name = name or f"Customer_{customer_id}"

        # Preferences
        self.preferences = {
            'max_price': np.random.uniform(15.0, 30.0),
            'min_rating': np.random.uniform(2.5, 4.0),
            'preferred_categories': np.random.choice(
                ["Bakery", "Cafe", "Restaurant", "Grocery", "Deli", "Pizza", "Sushi", "Fast Food"],
                size=np.random.randint(1, 4),
                replace=False
            ).tolist()
        }

        # Behavioral scores
        self.bias_score = np.random.uniform(0.0, 1.0)  # preference for familiar stores
        self.neophila_score = np.random.uniform(0.0, 1.0)  # likelihood to try new restaurants
        self.satisfaction_level = np.random.uniform(0.5, 1.0)  # current satisfaction

        # History tracking
        self.rating_history = []  # list of (store_id, rating)
        self.history = {
            "viewedRestaurants": [],
            "orders": []
        }

        # Optional location
        self.location = None
        self.longitude = None
        self.latitude = None
        
        # Store-specific preference ratings (from CSV: store_id -> valuation)
        # This represents how much the customer likes each store
        self.preference_ratings: Dict[int, float] = {}

        # Decision tracking
        self.arrival_time = 0.0
        self.decision = None  # 'buy' or 'leave'
        self.chosen_store_id = None
        self.displayed_stores = []

    def update_preferences(self, **kwargs):
        """Update customer preferences"""
        self.preferences.update(kwargs)

    def add_to_history(self, store_id: int, action: str):
        """Add store interaction to history"""
        if action == 'view':
            if store_id not in self.history['viewedRestaurants']:
                self.history['viewedRestaurants'].append(store_id)
        elif action == 'order':
            self.history['orders'].append(store_id)


# Global customer counter
_customer_counter = 0


def generate_customer(k: int, arrival_times: Optional[List[float]] = None, seed: int = None) -> List[Customer]:
    """
    Generate k customers with WIDE diversity in preferences.
    This ensures strategies have meaningful differences to work with.
    """
    global _customer_counter

    customers = []
    if seed is not None:
        np.random.seed(seed)

    if arrival_times is None:
        arrival_times = sorted(np.random.uniform(0, 24, k))

    for i in range(k):
        customer = Customer(_customer_counter + 1)
        customer.arrival_time = arrival_times[i]
        
        # Create MORE DIVERSE customers with stronger preferences
        # Price sensitivity: some very price-sensitive, others not
        customer.preferences['max_price'] = np.random.uniform(10.0, 60.0)  # Wider range
        
        # Rating preferences: some very picky, others not
        customer.preferences['min_rating'] = np.random.uniform(1.5, 4.5)  # Wider range
        
        # Category preferences: some have 1 category, others have many
        num_categories = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.3, 0.1])  # More variation
        
        # Bias and neophilia: wider ranges for more distinct customer types
        customer.bias_score = np.random.uniform(0.0, 1.0)  # Some very loyal, others not
        customer.neophila_score = np.random.uniform(0.0, 1.0)  # Some very adventurous, others not
        
        customers.append(customer)
        _customer_counter += 1
    return customers


def customer_arrives(customer: Customer) -> None:
    """Mark customer as arrived"""
    pass


def display_stores_to_customer(customer: Customer, store_list: List[Restaurant]) -> None:
    """Display stores to customer and update viewing history"""
    customer.displayed_stores = [store.restaurant_id for store in store_list]
    for store in store_list:
        customer.add_to_history(store.restaurant_id, 'view')


def probability_of_purchase(store: Restaurant, customer: Customer) -> float:
    """
    Calculate probability that customer will purchase from this store.
    
    Uses customer's preference_ratings (from CSV) if available, otherwise
    falls back to calculated factors based on price, rating, etc.
    """
    prob = 1.0

    # Use CSV valuation if available
    if customer.preference_ratings and store.restaurant_id in customer.preference_ratings:
        # The CSV valuation directly represents how much customer likes this store
        csv_valuation = customer.preference_ratings[store.restaurant_id]
        # Use valuation as base probability (0-1 range)
        prob = csv_valuation
        
        # Still apply price and rating filters (but less harshly)
        # Price factor: moderate penalty if too expensive
        if store.price > customer.preferences['max_price']:
            price_factor = 0.5  # moderate penalty instead of 0.2
        else:
            price_factor = 1.0 - (store.price / customer.preferences['max_price']) * 0.3
            price_factor = max(0.3, price_factor)  # less harsh minimum
        prob *= price_factor

        # Rating factor: moderate penalty if rating too low
        if store.rating < customer.preferences['min_rating']:
            rating_factor = 0.6  # moderate penalty instead of 0.3
        else:
            rating_factor = 0.7 + (store.rating / 5.0) * 0.3  # boost for good ratings
        prob *= rating_factor
        
    else:
        # Fallback: Use original calculation if no CSV valuation
        # Price factor: lower prices = higher probability
        if store.price > customer.preferences['max_price']:
            price_factor = 0.2  # very low probability if not fitting expected food prices
        else:
            price_factor = 1.0 - (store.price / customer.preferences['max_price']) * 0.5
            price_factor = max(0.1, price_factor)  # minimum
        prob *= price_factor

        # Rating factor: higher rating = higher probability
        if store.rating < customer.preferences['min_rating']:
            rating_factor = 0.3  # very low probability if not fitting expected
        else:
            rating_factor = store.rating / 5.0
        prob *= rating_factor

        # CATEGORY PREFERENCES (REALISTIC BEHAVIOR) - Strong effect
        # Users strongly prefer stores in their preferred categories
        if store.category in customer.preferences['preferred_categories']:
            category_factor = 1.6  # Strong boost for preferred category (increased from 1.2)
        else:
            category_factor = 0.6  # Strong penalty for non-preferred (decreased from 0.8)
        prob *= category_factor

    # FAMILIARITY BIAS (REALISTIC BEHAVIOR) - Strong effect
    # Users who have ordered from a store before are much more likely to order again
    if store.restaurant_id in customer.history["orders"]:
        # Strong familiarity boost (customers are loyal to familiar stores)
        prob *= (1.0 + customer.bias_score * 0.8)  # Increased from 0.3

    # NEOPHILIA (REALISTIC BEHAVIOR) - Strong effect for adventurous users
    # Users who haven't seen this store before
    if store.restaurant_id not in customer.history["viewedRestaurants"]:
        # Strong neophilia boost for adventurous customers
        # High neophilia score = loves trying new things
        prob *= (1.0 + customer.neophila_score * 0.6)  # Increased from 0.2

    # Base purchase probability
    base_prob = 0.7
    final_prob = base_prob * prob

    # Clip to [0, 1]
    return max(0.0, min(1.0, final_prob))


class CustomerChoiceMNL:
    """
    Proper Multinomial Logit Model (MNL) for customer choice.
    Uses softmax to convert utilities to probabilities.
    """
    
    def __init__(self):
        self.params = {
            'base_utility': 1.0,
            'price_sensitivity': -0.8,  # STRONGER price sensitivity (negative = lower price preferred)
            'rating_importance': 1.2,  # STRONGER rating importance (positive = higher rating preferred)
            'category_match': 1.5,  # STRONGER category preference boost
            'familiarity_boost': 1.0,  # STRONGER familiarity boost
            'neophilia_boost': 0.8,  # STRONGER neophilia boost
            'leave_utility': 0.5  # LOWER leave utility (customers more likely to buy)
        }
    
    def calculate_utility(self, store: Restaurant, customer: Customer, current_hour: float) -> float:
        """
        Calculate utility using additive model (proper MNL).
        Utility = sum of attribute values × coefficients
        """
        utility = self.params['base_utility']
        
        # Price (continuous, negative coefficient)
        utility += self.params['price_sensitivity'] * (store.price / 10.0)
        
        # Rating (continuous, positive coefficient)
        utility += self.params['rating_importance'] * (store.rating / 5.0)
        
        # Category match (realistic behavior - users prefer same category)
        if store.category in customer.preferences.get('preferred_categories', []):
            utility += self.params['category_match']
        
        # Familiarity bias (realistic behavior - loyal customers)
        if store.restaurant_id in customer.history.get("orders", []):
            utility += self.params['familiarity_boost'] * customer.bias_score
        
        # Neophilia (realistic behavior - adventurous users)
        if store.restaurant_id not in customer.history.get("viewedRestaurants", []):
            utility += self.params['neophilia_boost'] * customer.neophila_score
        
        # Inventory availability (prefer stores with stock)
        if store.est_inventory > store.reservation_count:
            safe_capacity = store.est_inventory - store.reservation_count
            utility += 0.2 * min(1.0, safe_capacity / 10.0)
        
        return utility
    
    def make_choice(self, displayed_stores: List[Restaurant], customer: Customer, current_hour: float) -> Optional[Restaurant]:
        """
        MNL choice with proper softmax and leave option.
        
        Returns:
            Restaurant if customer chooses to buy, None if customer leaves
        """
        if not displayed_stores:
            return None
        
        # Calculate utilities for all displayed stores
        utilities = [self.calculate_utility(store, customer, current_hour) 
                    for store in displayed_stores]
        
        # Add leave option utility (outside option)
        utilities.append(self.params['leave_utility'])
        
        # Convert to probabilities using softmax (proper MNL)
        # softmax(x_i) = exp(x_i) / sum(exp(x_j))
        probabilities = softmax(utilities)
        
        # Make choice based on probabilities
        choice_idx = np.random.choice(len(probabilities), p=probabilities)
        
        # Last index is "leave" option
        if choice_idx == len(probabilities) - 1:
            return None  # Customer leaves without buying
        else:
            return displayed_stores[choice_idx]


# Global MNL instance
_mnl_model = CustomerChoiceMNL()


def choose_store(stores: List[Restaurant], customer: Customer, current_hour: float = 12.0) -> Optional[Restaurant]:
    """
    Customer chooses a store using proper Multinomial Logit Model.
    
    Args:
        stores: List of displayed stores
        customer: Customer making the choice
        current_hour: Current hour of day (0-24) for time-based adjustments
    
    Returns:
        Restaurant if customer buys, None if customer leaves
    """
    return _mnl_model.make_choice(stores, customer, current_hour)


def customer_makes_decision(customer: Customer, displayed_stores: List[Restaurant], current_hour: float = 12.0) -> Dict:
    """
    Customer makes a decision using proper Multinomial Logit Model.
    
    Args:
        customer: Customer making the decision
        displayed_stores: Stores displayed to customer
        current_hour: Current hour of day (0-24) for time-based utility adjustments
    
    Returns:
        Dict with 'action' ('buy' or 'leave') and 'store_id' (if buy)
    """
    customer.displayed_stores = [store.restaurant_id for store in displayed_stores]
    
    # Use proper MNL model with current hour
    chosen_store = choose_store(displayed_stores, customer, current_hour)

    if chosen_store is None:
        customer.decision = 'leave'
        customer.chosen_store_id = None
        return {
            'action': customer.decision,
            'store_id': customer.chosen_store_id
        }
    else:
        customer.decision = 'buy'
        customer.chosen_store_id = chosen_store.restaurant_id
        customer.add_to_history(chosen_store.restaurant_id, 'order')
        return {
            'action': customer.decision,
            'store_id': customer.chosen_store_id
        }


def get_customer_preferences(customer: Customer) -> Dict:
    """Return customer preferences as a dictionary"""
    return customer.preferences.copy()

