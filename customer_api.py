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
import functools


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

        # Learned preferences for Anan_Strategy (evolves over time)
        self.learned_preference_ratings: Dict[int, float] = {}

        # Decision tracking
        self.arrival_time = 0.0
        self.decision = None  # 'buy' or 'leave'
        self.chosen_store_id = None
        self.displayed_stores = []
        
        # network effects tracking
        self.social_influence_score = {}  # {store_id: influence_score} - influenced by other customers
        
        # churn tracking (dynamic with running averages)
        self.consecutive_cancellations = 0
        self.is_churned = False
        self.days_since_churn = 0
        self.cancellation_history = []  # list of (day, was_cancelled: bool) for running average
        self.cancellation_rate_window = 7  # days to look back for running average
        self.running_cancellation_rate = 0.0  # running average cancellation rate
        
        # customer lifetime value (CLV) tracking
        self.total_orders = 0
        self.total_spent = 0.0
        self.order_history_dates = []  # list of (day, order_value)
        self.first_order_day = None
        self.last_order_day = None
        
        # retention tracking
        self.days_active = set()  # set of days customer made a purchase
        self.first_seen_day = None
        self.last_seen_day = None
    
    @functools.lru_cache(maxsize=1)
    def _get_segment_cache_key(self) -> tuple:
        """create cache key for segment calculation"""
        max_price = self.preferences.get('max_price', 25.0)
        min_rating = self.preferences.get('min_rating', 3.0)
        has_orders = len(self.history.get('orders', [])) > 0
        return (max_price, min_rating, has_orders, self.bias_score, self.neophila_score)
    
    def determine_segment(self) -> str:
        """
        derive customer segment from existing characteristics.
        segments emerge naturally from customer traits - no random assignment needed.
        cached for performance.
        """
        # use cached values for performance
        max_price = self.preferences.get('max_price', 25.0)
        min_rating = self.preferences.get('min_rating', 3.0)
        has_orders = len(self.history.get('orders', [])) > 0
        
        # price sensitive: low price tolerance
        if max_price < 20.0:
            return 'PRICE_SENSITIVE'
        
        # quality focused: high rating requirements, not price sensitive
        if min_rating > 3.5 and max_price > 25.0:
            return 'QUALITY_FOCUSED'
        
        # loyal: high bias score and has order history
        if self.bias_score > 0.7 and has_orders:
            return 'LOYAL'
        
        # explorer: high neophilia, low bias
        if self.neophila_score > 0.7 and self.bias_score < 0.4:
            return 'EXPLORER'
        
        # convenience seeker: moderate bias, moderate neophilia (balanced but prefers familiar)
        if 0.4 <= self.bias_score <= 0.7 and 0.3 <= self.neophila_score <= 0.6:
            return 'CONVENIENCE_SEEKER'
        
        # balanced: default for everyone else
        return 'BALANCED'
    
    @property
    def customer_segment(self) -> str:
        """get customer segment (computed property)"""
        return self.determine_segment()
    
    def _clear_segment_cache(self):
        """clear segment cache when customer traits change"""
        if hasattr(self.determine_segment, 'cache_clear'):
            self.determine_segment.cache_clear()

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
    
    def record_order(self, day: int, order_value: float):
        """record an order for CLV calculation"""
        self.total_orders += 1
        self.total_spent += order_value
        self.order_history_dates.append((day, order_value))
        if self.first_order_day is None:
            self.first_order_day = day
        self.last_order_day = day
        self.days_active.add(day)
    
    def record_activity(self, day: int):
        """record customer activity (view or arrival) for retention tracking"""
        if self.first_seen_day is None:
            self.first_seen_day = day
        self.last_seen_day = day
    
    def calculate_retention_rate(self, current_day: int, lookback_days: int = 7) -> float:
        """calculate retention rate: % of recent days customer was active"""
        if current_day <= lookback_days:
            return 1.0 if self.total_orders > 0 else 0.0
        
        recent_days = [d for d in self.days_active if d > (current_day - lookback_days)]
        return len(recent_days) / lookback_days if lookback_days > 0 else 0.0
    
    def calculate_clv(self, current_day: int, discount_rate: float = 0.1) -> float:
        """
        calculate customer lifetime value.
        CLV = (average order value * purchase frequency * customer lifespan) / (1 + discount_rate)
        
        simplified formula for simulation:
        CLV = total_spent * retention_multiplier
        where retention_multiplier accounts for expected future value
        """
        if self.total_orders == 0:
            return 0.0
        
        # calculate average order value
        avg_order_value = self.total_spent / self.total_orders
        
        # calculate purchase frequency (orders per day active)
        if self.first_order_day is not None:
            days_active = max(1, current_day - self.first_order_day + 1)
            purchase_frequency = self.total_orders / days_active
        else:
            purchase_frequency = 0.0
        
        # estimate customer lifespan (based on retention)
        # if customer has been active, estimate future value
        retention_probability = min(1.0, self.satisfaction_level)
        if self.is_churned:
            retention_probability *= 0.3  # churned customers have low retention
        
        # expected future orders (simplified: based on current frequency and retention)
        expected_future_orders = purchase_frequency * 30 * retention_probability  # next 30 days
        
        # CLV = past value + expected future value (discounted)
        past_value = self.total_spent
        future_value = avg_order_value * expected_future_orders / (1 + discount_rate)
        
        return past_value + future_value


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
        
        # Generate realistic coordinates (e.g., roughly 30.0-30.1 range for a city)
        customer.latitude = np.random.uniform(30.00000, 30.10000)
        customer.longitude = np.random.uniform(31.20000, 31.30000)

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


# segment-specific utility multipliers
# applied to base utility components based on customer segment
SEGMENT_MULTIPLIERS = {
    'PRICE_SENSITIVE': {'price': 2.0, 'rating': 0.5, 'familiarity': 0.5, 'neophilia': 0.5},
    'QUALITY_FOCUSED': {'price': 0.5, 'rating': 2.0, 'familiarity': 0.5, 'neophilia': 0.5},
    'LOYAL': {'price': 1.0, 'rating': 1.0, 'familiarity': 3.0, 'neophilia': 0.2},
    'EXPLORER': {'price': 1.0, 'rating': 1.0, 'familiarity': 0.3, 'neophilia': 2.0},
    'CONVENIENCE_SEEKER': {'price': 1.0, 'rating': 1.0, 'familiarity': 1.5, 'neophilia': 0.3},
    'BALANCED': {'price': 1.0, 'rating': 1.0, 'familiarity': 1.0, 'neophilia': 1.0}
}

# price elasticity coefficients by segment
# elasticity = % change in demand / % change in price
# negative values mean demand decreases as price increases
# more negative = more price sensitive
PRICE_ELASTICITY = {
    'PRICE_SENSITIVE': -2.5,  # very elastic: 10% price increase = 25% demand decrease
    'QUALITY_FOCUSED': -0.5,  # inelastic: 10% price increase = 5% demand decrease
    'LOYAL': -1.0,            # moderate elasticity
    'EXPLORER': -1.2,         # moderate-high elasticity
    'CONVENIENCE_SEEKER': -1.5,  # high elasticity
    'BALANCED': -1.0          # moderate elasticity
}


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
        # cache for utility calculations (store_id, customer_id) -> utility
        self._utility_cache = {}
    
    def calculate_utility(self, store: Restaurant, customer: Customer, current_hour: float) -> float:
        """
        Calculate utility using additive model (proper MNL).
        Utility = sum of attribute values × coefficients
        Segment-specific multipliers adjust weights based on customer type.
        Price elasticity modeling: demand response to price changes varies by segment.
        Cached for performance when same store/customer combination is evaluated multiple times.
        """
        # use cache key (store and customer don't change frequently during a single decision)
        cache_key = (id(store), id(customer))
        if cache_key in self._utility_cache:
            return self._utility_cache[cache_key]
        
        utility = self.params['base_utility']
        
        # get customer segment and multipliers
        segment = customer.customer_segment
        multipliers = SEGMENT_MULTIPLIERS.get(segment, SEGMENT_MULTIPLIERS['BALANCED'])
        elasticity = PRICE_ELASTICITY.get(segment, PRICE_ELASTICITY['BALANCED'])
        
        # price (continuous, negative coefficient) - adjusted by segment and elasticity
        # use reference price (customer's max_price) for elasticity calculation
        reference_price = customer.preferences.get('max_price', 25.0)
        effective_price = store.get_effective_price() if hasattr(store, 'get_effective_price') else store.price
        price_ratio = effective_price / reference_price if reference_price > 0 else 1.0
        
        # apply price elasticity: more elastic segments respond more strongly to price changes
        # elasticity effect: (price_ratio - 1) * elasticity
        # e.g., if price is 20% above reference and elasticity is -2.0, utility decreases by 40%
        price_elasticity_effect = (price_ratio - 1.0) * abs(elasticity)
        price_component = self.params['price_sensitivity'] * (effective_price / 10.0) * (1.0 + price_elasticity_effect)
        utility += price_component * multipliers['price']
        
        # promotion boost: customers respond positively to active promotions
        if hasattr(store, 'active_promotion') and store.active_promotion:
            promotion_boost = 0.3 * (store.active_promotion.get('discount_pct', 0) / 100.0)
            utility += promotion_boost  # boost utility for promotions
        
        # network effects: social proof from other customers
        # high-rated stores with recent orders get boost (word-of-mouth effect)
        if hasattr(store, 'recent_orders_count') and store.recent_orders_count > 0:
            # social proof: more recent orders = more popular = higher utility
            social_proof_boost = 0.15 * min(1.0, store.recent_orders_count / 10.0)
            utility += social_proof_boost
        
        # social influence: if store is popular in marketplace, boost utility
        # this simulates word-of-mouth and social proof
        if hasattr(customer, 'social_influence_score') and store.restaurant_id in customer.social_influence_score:
            influence_boost = customer.social_influence_score[store.restaurant_id] * 0.2
            utility += influence_boost
        
        # rating (continuous, positive coefficient) - adjusted by segment
        rating_component = self.params['rating_importance'] * (store.rating / 5.0)
        utility += rating_component * multipliers['rating']
        
        # category match (realistic behavior - users prefer same category)
        if store.category in customer.preferences.get('preferred_categories', []):
            utility += self.params['category_match']
        
        # familiarity bias (realistic behavior - loyal customers) - adjusted by segment
        if store.restaurant_id in customer.history.get("orders", []):
            familiarity_component = self.params['familiarity_boost'] * customer.bias_score
            utility += familiarity_component * multipliers['familiarity']
        
        # neophilia (realistic behavior - adventurous users) - adjusted by segment
        if store.restaurant_id not in customer.history.get("viewedRestaurants", []):
            neophilia_component = self.params['neophilia_boost'] * customer.neophila_score
            utility += neophilia_component * multipliers['neophilia']
        
        # inventory availability (prefer stores with stock)
        if store.est_inventory > store.reservation_count:
            safe_capacity = store.est_inventory - store.reservation_count
            utility += 0.2 * min(1.0, safe_capacity / 10.0)
        
        # cache result
        self._utility_cache[cache_key] = utility
        return utility
    
    def clear_cache(self):
        """clear utility cache (call when stores/customers change)"""
        self._utility_cache.clear()
    
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

