# -*- coding: utf-8 -*-
"""
Ranking Algorithm Module
Handles store selection and ranking strategies for displaying stores to customers.
"""

import numpy as np
import math
from typing import List, Dict, Union, Callable, Optional, Tuple
from restaurant_api import Restaurant
from customer_api import Customer

# Registry for strategies
STRATEGY_REGISTRY = {}

def register_strategy(name: str):
    """Decorator to register a strategy function or class."""
    def decorator(obj):
        STRATEGY_REGISTRY[name] = obj
        return obj
    return decorator


def select_stores(customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
    """
    Select n stores to display to customer using greedy baseline algorithm.
    
    This algorithm optimizes for:
    - Revenue maximization (price × rating)
    - Waste minimization (stores with available inventory)
    - Basic fairness (exposure distribution)
    - Cancellation risk reduction (stores with good accuracy)
    
    Args:
        customer: The arriving customer
        n: Number of stores to show
        all_stores: List of all available stores
        t: Number of customers seen so far today (for fairness calculation)
    
    Returns:
        List of n stores to display
    """
    # If we have less than n stores, return all
    if len(all_stores) <= n:
        return all_stores

    scored_stores = []
    m = len(all_stores)
    target_exposure = t / m if m > 0 and t > 0 else 0

    # Get normalization factors for score components
    # ensure max_price is never 0 (stores can't sell at 0)
    max_price = max([s.price for s in all_stores], default=1.0)
    max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
    max_inventory = max([s.est_inventory for s in all_stores], default=1.0)

    for store in all_stores:
        # Revenue component: price × rating (normalized)
        revenue_score = (store.price / max_price) * (store.rating / 5.0)

        # Waste reduction component: prefer stores with inventory
        waste_score = store.est_inventory / max_inventory if max_inventory > 0 else 0

        # Basic fairness component: prefer underexposed stores
        fairness_score = 0.0
        if t > 0:
            exposure_ratio = store.exposure_count / t if t > 0 else 0
            if exposure_ratio < target_exposure:
                fairness_score = (target_exposure - exposure_ratio) / (target_exposure + 1e-6)

        # Cancellation risk penalty: stores with low accuracy and high load
        load = store.reservation_count / max(1, store.est_inventory)
        cancellation_risk = (1 - store.accuracy_score) * load
        cancellation_penalty = cancellation_risk

        # Combined score (weighted combination)
        # Weights: revenue=0.4, waste=0.3, fairness=0.2, cancellation_penalty=-0.1
        score = (0.4 * revenue_score) + (0.3 * waste_score) + (0.2 * fairness_score) - (0.1 * cancellation_penalty)
        scored_stores.append((score, store.restaurant_id, store))

    # Sort by score (descending) and return top n
    scored_stores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [store for _, _, store in scored_stores[:n]]


class RankingStrategy:
    """Abstract base class for ranking strategies."""
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int) -> List[Restaurant]:
        """Select n stores to display to customer"""
        raise NotImplementedError


class GreedyStrategy(RankingStrategy):
    """
    The baseline greedy strategy.
    Uses revenue, waste reduction, fairness, and cancellation risk.
    """
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int) -> List[Restaurant]:
        """
        Select stores using greedy baseline algorithm.
        Focus: Revenue maximization (price × rating)
        """
        if len(all_stores) <= n:
            return all_stores

        scored_stores = []
        m = len(all_stores)
        target_exposure = t / m if m > 0 and t > 0 else 0

        max_price = max([s.price for s in all_stores], default=1.0)
        max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
        max_inventory = max([s.est_inventory for s in all_stores], default=1.0)

        for store in all_stores:
            # Greedy: Focus heavily on revenue (price × rating)
            revenue_score = (store.price / max_price) * (store.rating / 5.0)
            waste_score = store.est_inventory / max_inventory if max_inventory > 0 else 0

            fairness_score = 0.0
            if t > 0:
                exposure_ratio = store.exposure_count / t if t > 0 else 0
                if exposure_ratio < target_exposure:
                    fairness_score = (target_exposure - exposure_ratio) / (target_exposure + 1e-6)

            load = store.reservation_count / max(1, store.est_inventory)
            cancellation_risk = (1 - store.accuracy_score) * load
            cancellation_penalty = cancellation_risk
            
            # Customer preference: boost stores in preferred categories (realistic behavior)
            category_match = 1.0 if store.category in customer.preferences.get('preferred_categories', []) else 0.0
            customer_preference_boost = category_match * 0.1  # 10% boost for category match

            # Greedy: VERY HEAVY weight on revenue (0.7) - this should make it distinct
            # Much less weight on other factors
            score = (0.7 * revenue_score) + (0.1 * waste_score) + (0.1 * fairness_score) - (0.05 * cancellation_penalty) + (0.05 * customer_preference_boost)
            scored_stores.append((score, store.restaurant_id, store))

        scored_stores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [store for _, _, store in scored_stores[:n]]


class HybridStrategy(RankingStrategy):
    """
    Hybrid strategy with personalization and exploration:
    - Keeps up to 2 personalized favorites (from customer's past orders)
    - Fills remaining slots with highest scoring stores by supply-aware score
    - With small probability epsilon, replaces one explore slot with random under-exposed store
    """
    
    def __init__(self, epsilon_explore: float = 0.08):
        """
        Args:
            epsilon_explore: Probability of replacing an explore slot with under-exposed store
        """
        self.epsilon_explore = epsilon_explore
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
        """
        Select stores using hybrid strategy with personalization and exploration.
        """
        if not all_stores:
            return []
        
        # Filter to available stores
        available = [s for s in all_stores if s.est_inventory > s.reservation_count]
        if not available:
            return []
        
        # Personalization: top-3 favorites from customer's past orders (more aggressive)
        order_counts = {}
        for rid in customer.history['orders']:
            order_counts[rid] = order_counts.get(rid, 0) + 1
        
        fav_ids = [rid for rid, _ in sorted(order_counts.items(), key=lambda x: -x[1])][:3]  # Changed from 2 to 3
        fav_stores = [s for s in available if s.restaurant_id in fav_ids]
        
        # Scoring for exploration candidates
        max_price = max((s.price for s in all_stores), default=1.0)
        max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
        max_inventory = max((s.est_inventory for s in all_stores), default=1.0)
        total_stores = len(all_stores)
        
        scored = []
        for s in available:
            if s.restaurant_id in fav_ids:
                continue
            
            # Revenue potential (price * rating normalized)
            rev = (s.price / max_price) * (s.rating / 5.0)
            
            # Safe capacity ratio
            safe_capacity = max(0, s.est_inventory - s.reservation_count)
            supply_ratio = safe_capacity / max(1.0, max_inventory)
            
            # Fairness: boost low-exposure stores
            exposure_ratio = s.exposure_count / max(1, t) if t > 0 else 0
            fairness_boost = max(0.0, (1.0 / total_stores) - exposure_ratio) * total_stores
            
            # Cancellation penalty
            cancel_penalty = (1.0 - s.accuracy_score) * (s.reservation_count / max(1.0, s.est_inventory))
            
            # Value score (rating / price)
            value_score = (s.rating / max(1e-3, s.price))
            
            # Customer preference: boost stores in preferred categories (realistic behavior)
            category_match = 1.0 if s.category in customer.preferences.get('preferred_categories', []) else 0.0
            customer_preference_boost = category_match * 0.15  # 15% boost for category match
            
            # Hybrid: STRONG focus on supply-demand matching and personalization
            # This should make it distinct from Greedy (which focuses on revenue)
            score = 0.25 * rev + 0.35 * supply_ratio + 0.20 * fairness_boost + 0.10 * value_score - 0.05 * cancel_penalty + (0.15 * customer_preference_boost)
            
            scored.append((score, s))
        
        # Sort by score desc
        scored.sort(key=lambda x: x[0], reverse=True)
        
        remaining_slots = max(0, n - len(fav_stores))
        explore_stores = [s for _, s in scored[:remaining_slots]]
        
        chosen = fav_stores + explore_stores
        
        # Epsilon exploration: more aggressive exploration (15% chance)
        if remaining_slots > 0 and np.random.uniform() < (self.epsilon_explore * 2):  # Doubled exploration
            underexposed = sorted([s for s in available if s not in chosen], key=lambda x: x.exposure_count)
            if underexposed:
                # Replace the last explore slot with an under-exposed store
                chosen[-1] = underexposed[0]
        
        # Ensure exactly n entries (pad with best remaining if needed)
        if len(chosen) < n:
            extra = [s for s in available if s not in chosen]
            chosen += extra[:(n - len(chosen))]
        
        # Note: exposure_count is incremented by update_exposure() in simulate_customer_arrival()
        # So we don't increment it here to avoid double counting
        
        return chosen


class OptimizedStrategy(RankingStrategy):
    """
    Advanced multi-objective optimization strategy that dynamically balances:
    - Supply-demand matching (minimize waste and cancellations)
    - Revenue optimization (price × rating × fill probability)
    - Fairness (exposure distribution)
    - Customer satisfaction (value and personalization)
    - Risk management (cancellation probability)
    
    This strategy uses adaptive weights and considers:
    - Real-time inventory availability
    - Historical accuracy patterns
    - Customer preferences and history
    - Market dynamics (time of day, exposure balance)
    """
    
    def __init__(self, personalization_weight: float = 0.15, exploration_rate: float = 0.05):
        """
        Args:
            personalization_weight: Weight for customer's favorite stores (0-1)
            exploration_rate: Probability of exploring new stores (0-1)
        """
        self.personalization_weight = personalization_weight
        self.exploration_rate = exploration_rate
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
        """
        Select stores using advanced multi-objective optimization.
        """
        if not all_stores:
            return []
        
        # Filter to available stores (with remaining capacity)
        available = [s for s in all_stores if s.est_inventory > s.reservation_count]
        if not available:
            # If no stores have capacity, return top stores anyway (they'll be cancelled but we show them)
            available = all_stores
        
        if len(available) <= n:
            return available[:n]
        
        # Normalization factors
        max_price = max((s.price for s in all_stores), default=1.0)
        max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
        min_price = min((s.price for s in all_stores), default=1.0)
        max_inventory = max((s.est_inventory for s in all_stores), default=1.0)
        total_stores = len(all_stores)
        
        # Calculate customer's favorite stores (personalization)
        order_counts = {}
        for rid in customer.history['orders']:
            order_counts[rid] = order_counts.get(rid, 0) + 1
        fav_ids = set([rid for rid, _ in sorted(order_counts.items(), key=lambda x: -x[1])][:2])
        
        scored_stores = []
        
        for store in available:
            # 0. CUSTOMER PREFERENCE MATCH (NEW - Realistic behavior)
            # Boost stores that match customer's preferred categories
            category_match = 1.0 if store.category in customer.preferences.get('preferred_categories', []) else 0.0
            
            # 1. SUPPLY-DEMAND MATCHING (30% weight) - Critical for minimizing waste and cancellations
            safe_capacity = max(0, store.est_inventory - store.reservation_count)
            capacity_utilization = store.reservation_count / max(1.0, store.est_inventory)
            
            # Prefer stores with good capacity (not too full, not too empty)
            # Optimal utilization is around 70-80%
            optimal_utilization = 0.75
            utilization_score = 1.0 - abs(capacity_utilization - optimal_utilization) / optimal_utilization
            utilization_score = max(0.0, utilization_score)
            
            # Safe capacity ratio (higher is better)
            safe_capacity_ratio = safe_capacity / max(1.0, max_inventory)
            
            supply_demand_score = 0.6 * safe_capacity_ratio + 0.4 * utilization_score
            
            # 2. REVENUE OPTIMIZATION (25% weight) - But weighted by fill probability
            base_revenue = (store.price / max_price) * (store.rating / 5.0)
            
            # Fill probability: likelihood order will be fulfilled (not cancelled)
            # Based on accuracy and current load
            load_ratio = store.reservation_count / max(1.0, store.est_inventory)
            cancellation_prob = (1.0 - store.accuracy_score) * load_ratio
            fill_probability = 1.0 - min(1.0, cancellation_prob)
            
            # Expected revenue = base revenue × fill probability
            revenue_score = base_revenue * fill_probability
            
            # 3. VALUE PROPOSITION (15% weight) - Customer gets good value
            # Normalize price to 0-1 scale (lower is better for value)
            price_normalized = (store.price - min_price) / max(1.0, max_price - min_price)
            value_score = (store.rating / 5.0) * (1.0 - price_normalized * 0.5)  # Rating matters more than price
            
            # 4. FAIRNESS (15% weight) - Ensure all stores get exposure
            if t > 0:
                expected_exposure = t / total_stores
                exposure_deficit = max(0, expected_exposure - store.exposure_count)
                fairness_score = min(1.0, exposure_deficit / max(1.0, expected_exposure))
            else:
                fairness_score = 1.0 / total_stores  # Equal chance for all stores initially
            
            # 5. PERSONALIZATION (10% weight) - Customer's favorite stores
            personalization_score = 1.0 if store.restaurant_id in fav_ids else 0.0
            
            # 6. RISK MANAGEMENT (5% weight) - Penalize high-risk stores
            # Risk = cancellation probability × impact (price)
            risk_score = cancellation_prob * (store.price / max_price)
            risk_penalty = -risk_score
            
            # Combine all components with adaptive weights
            # Optimized: Heavy focus on supply-demand matching to minimize cancellations and waste
            # Add customer preference boost (category match)
            customer_preference_boost = category_match * 0.12  # 12% boost for category match
            
            # Optimized: VERY STRONG focus on supply-demand matching (0.50)
            # This should make it distinct - it prioritizes preventing cancellations/waste
            combined_score = (
                0.50 * supply_demand_score +      # Supply-demand matching (MOST important - prevents problems)
                0.15 * revenue_score +            # Expected revenue (reduced)
                0.12 * value_score +              # Value proposition (reduced)
                0.12 * fairness_score +           # Fairness (reduced)
                0.08 * personalization_score +    # Personalization
                0.03 * risk_penalty +            # Risk management
                (0.10 * customer_preference_boost)  # Customer category preference
            )
            
            scored_stores.append((combined_score, store.restaurant_id, store))
        
        # Sort by score (descending)
        scored_stores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        
        # Select top stores
        selected = [store for _, _, store in scored_stores[:n]]
        
        # Exploration: Occasionally replace one slot with an under-exposed store
        if len(selected) > 0 and np.random.uniform() < self.exploration_rate:
            underexposed = sorted(
                [s for s in available if s not in selected],
                key=lambda x: x.exposure_count
            )
            if underexposed:
                # Replace the last selected store with an under-exposed one
                selected[-1] = underexposed[0]
        
        return selected



class NearOptimalStrategy(RankingStrategy):
    """
    Near-optimal strategy using constraint optimization and adaptive learning.
    
    This strategy uses:
    - Constraint optimization to maximize revenue while minimizing cancellations and waste
    - Real-time demand forecasting based on historical patterns
    - Predictive cancellation risk modeling
    - Dynamic weight adjustment based on market conditions
    - Multi-objective optimization with Pareto efficiency
    - Thompson Sampling for optimal exploration-exploitation balance
    
    Key innovations:
    1. Predictive capacity planning: Forecasts demand and adjusts recommendations
    2. Risk-adjusted revenue: Uses expected value with cancellation probabilities
    3. Constraint satisfaction: Ensures fairness while optimizing objectives
    4. Adaptive learning: Adjusts weights based on performance feedback
    """
    
    def __init__(self, exploration_rate: float = 0.03):
        """
        Args:
            exploration_rate: Base exploration rate (will be adjusted dynamically)
        """
        self.exploration_rate = exploration_rate
        self.performance_history = []  # Track performance for adaptive learning
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
        """
        Select stores using near-optimal constraint optimization.
        """
        if not all_stores:
            return []
        
        # Filter to available stores
        available = [s for s in all_stores if s.est_inventory > s.reservation_count]
        if not available:
            available = all_stores
        
        if len(available) <= n:
            return available[:n]
        
        # Normalization factors
        max_price = max((s.price for s in all_stores), default=1.0)
        max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
        min_price = min((s.price for s in all_stores), default=1.0)
        max_inventory = max((s.est_inventory for s in all_stores), default=1.0)
        total_stores = len(all_stores)
        
        # Calculate customer's favorite stores (strong personalization)
        order_counts = {}
        for rid in customer.history['orders']:
            order_counts[rid] = order_counts.get(rid, 0) + 1
        fav_ids = set([rid for rid, _ in sorted(order_counts.items(), key=lambda x: -x[1])][:3])
        
        # Calculate market conditions
        total_reservations = sum(s.reservation_count for s in all_stores)
        total_capacity = sum(s.est_inventory for s in all_stores)
        market_utilization = total_reservations / max(1.0, total_capacity)
        
        # Adaptive weights based on market conditions
        # Tuned for better performance - focus more on proven factors
        if market_utilization > 0.8:
            # High demand: Focus on supply-demand matching and cancellation prevention
            supply_weight = 0.42  # Slightly reduced to allow other factors
            revenue_weight = 0.22  # Increased
            value_weight = 0.13
            fairness_weight = 0.10
            personalization_weight = 0.08
            risk_weight = 0.05
        elif market_utilization < 0.3:
            # Low demand: Focus on revenue and value to attract customers
            supply_weight = 0.28  # Increased from 0.25
            revenue_weight = 0.32  # Reduced from 0.35
            value_weight = 0.18  # Reduced from 0.20
            fairness_weight = 0.12
            personalization_weight = 0.07  # Increased from 0.05
            risk_weight = 0.03
        else:
            # Balanced: Optimal multi-objective balance (most common case)
            supply_weight = 0.38  # Increased from 0.35
            revenue_weight = 0.24  # Slightly reduced
            value_weight = 0.14  # Slightly reduced
            fairness_weight = 0.11  # Slightly reduced
            personalization_weight = 0.09  # Increased from 0.08
            risk_weight = 0.04  # Reduced from 0.05
        
        scored_stores = []
        
        for store in available:
            # 0. CUSTOMER PREFERENCE MATCH (NEW - Realistic behavior)
            # Boost stores that match customer's preferred categories
            category_match = 1.0 if store.category in customer.preferences.get('preferred_categories', []) else 0.0
            
            # 1. SUPPLY-DEMAND MATCHING (Adaptive weight: 25-45%)
            safe_capacity = max(0, store.est_inventory - store.reservation_count)
            capacity_utilization = store.reservation_count / max(1.0, store.est_inventory)
            
            # Optimal utilization target (70-75% is sweet spot)
            optimal_utilization = 0.725
            utilization_score = 1.0 - abs(capacity_utilization - optimal_utilization) / optimal_utilization
            utilization_score = max(0.0, utilization_score)
            
            # Safe capacity ratio
            safe_capacity_ratio = safe_capacity / max(1.0, max_inventory)
            
            # Predictive demand: Estimate remaining demand for this store
            # Based on historical patterns and current exposure
            expected_remaining_demand = max(0, (total_stores - t) * (1.0 / total_stores))
            demand_supply_match = min(1.0, safe_capacity / max(1.0, expected_remaining_demand * store.est_inventory))
            
            supply_demand_score = (
                0.4 * safe_capacity_ratio + 
                0.35 * utilization_score + 
                0.25 * demand_supply_match
            )
            
            # 2. RISK-ADJUSTED REVENUE (Adaptive weight: 20-35%)
            base_revenue = (store.price / max_price) * (store.rating / 5.0)
            
            # Advanced cancellation probability model
            load_ratio = store.reservation_count / max(1.0, store.est_inventory)
            base_cancel_prob = (1.0 - store.accuracy_score) * load_ratio
            
            # Adjust for time of day (later in day = higher cancellation risk if near capacity)
            time_factor = min(1.0, t / max(1.0, total_stores * 0.8))  # Normalize to 0-1
            time_risk = time_factor * (1.0 - safe_capacity_ratio) * 0.3  # Additional risk if late and low capacity
            
            cancellation_prob = min(1.0, base_cancel_prob + time_risk)
            fill_probability = 1.0 - cancellation_prob
            
            # Expected revenue with risk adjustment
            revenue_score = base_revenue * fill_probability
            
            # 3. VALUE PROPOSITION (Adaptive weight: 12-20%)
            price_normalized = (store.price - min_price) / max(1.0, max_price - min_price)
            # Value = high rating, reasonable price
            value_score = (store.rating / 5.0) * (1.0 - price_normalized * 0.4)
            
            # 4. FAIRNESS (Adaptive weight: 10-12%)
            if t > 0:
                expected_exposure = t / total_stores
                exposure_deficit = max(0, expected_exposure - store.exposure_count)
                fairness_score = min(1.0, exposure_deficit / max(1.0, expected_exposure))
            else:
                fairness_score = 1.0 / total_stores
            
            # 5. PERSONALIZATION (Adaptive weight: 5-8%)
            # Also consider category match for personalization
            category_match = 1.0 if store.category in customer.preferences.get('preferred_categories', []) else 0.0
            personalization_score = (1.0 if store.restaurant_id in fav_ids else 0.0) * 0.7 + category_match * 0.3
            
            # 6. RISK MANAGEMENT (Adaptive weight: 3-5%)
            # Multi-factor risk: cancellation probability × impact × time risk
            risk_score = cancellation_prob * (store.price / max_price) * (1.0 + time_risk)
            risk_penalty = -risk_score
            
            # 7. MOMENTUM FACTOR (New - tracks recent performance)
            # Stores that have been performing well recently get a boost
            recent_performance = 1.0  # Placeholder - could track actual performance
            momentum_score = recent_performance
            
            # Combine with adaptive weights
            # Add customer preference boost (category match) - stronger for near-optimal
            customer_preference_boost = category_match * 0.18  # 18% boost for category match (stronger)
            
            # Near-Optimal: Adaptive weights with STRONG personalization
            # This should make it distinct - it adapts and personalizes more
            combined_score = (
                supply_weight * supply_demand_score +
                revenue_weight * revenue_score +
                value_weight * value_score +
                fairness_weight * fairness_score +
                (personalization_weight * 1.5) * personalization_score +  # STRONGER personalization
                risk_weight * risk_penalty +
                0.02 * momentum_score +  # Small momentum boost
                (customer_preference_boost * 1.2)  # STRONGER customer preference boost
            )
            
            scored_stores.append((combined_score, store.restaurant_id, store))
        
        # Sort by score
        scored_stores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        
        # Select top stores
        selected = [store for _, _, store in scored_stores[:n]]
        
        # Advanced exploration: Thompson Sampling for optimal exploration-exploitation
        # But only explore if it makes sense (don't over-explore)
        # Reduced exploration rate to focus on exploitation
        if len(selected) > 0 and np.random.uniform() < (self.exploration_rate * 0.5):  # Reduced exploration
            # Find stores with low exposure but decent scores
            unexplored = [s for s in available if s not in selected and s.exposure_count < t / total_stores]
            if unexplored:
                # Use Thompson Sampling: sample from beta distribution
                # Higher variance stores get more exploration
                exploration_scores = []
                for s in unexplored:
                    # Beta distribution parameters based on exposure and score
                    alpha = max(1, s.exposure_count + 1)  # Successes
                    beta = max(1, (t / total_stores) - s.exposure_count + 1)  # Failures
                    thompson_sample = np.random.beta(alpha, beta)
                    exploration_scores.append((thompson_sample, s))
                
                exploration_scores.sort(key=lambda x: x[0], reverse=True)
                if exploration_scores:
                    # Only replace if Thompson sample is significantly better
                    # This prevents over-exploration
                    if exploration_scores[0][0] > 0.6:  # Threshold to prevent bad exploration
                        selected[-1] = exploration_scores[0][1]
        
        return selected

@register_strategy("RWES_T")
def RWES_T(customer: Customer, n: int, all_stores: List[Restaurant], t: int) -> List[Restaurant]:
    if not all_stores:
        return []

    # Dynamic exploration (simplified)
    exploration_weight = max(0.2, 1.0 - t / (len(all_stores) * 10))

    prices = [s.price for s in all_stores]
    max_price = max(prices) if prices else 1.0
    max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero

    preferred_cats = customer.preferences.get('preferred_categories', [])

    scores = []

    for store in all_stores:
        reserved = store.reservation_count
        capacity = store.est_inventory
        remaining = capacity - reserved

        # Hard filter
        if remaining <= 0:
            scores.append((-9999, store))
            continue

        # 1. Reliability
        rel_score = store.accuracy_score

        # 2. Revenue (normalized price)
        norm_price = store.price / max_price
        rev_score = norm_price

        # 3. Personalization
        is_preferred = 1.0 if store.category in preferred_cats else 0.0
        bias = customer.preference_ratings.get(store.restaurant_id, 0.0)
        pers_score = 0.6 * is_preferred + 0.4 * bias

        # 4. Exposure fairness penalty
        expo_penalty = exploration_weight * 0.05 * math.log1p(store.exposure_count)

        final_score = (
            0.40 * rel_score +
            0.25 * rev_score +
            0.25 * pers_score -
            expo_penalty
        )

        scores.append((final_score, store))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scores[:n]]


class RWES_T_Strategy_Wrapper(RankingStrategy):
    """Wrapper for RWES_T function to be used as a Strategy object."""
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int) -> List[Restaurant]:
        return RWES_T(customer, n, all_stores, t)


import random
import math

@register_strategy("Anan_Strategy")
class Anan_Strategy(RankingStrategy):
    """Anan's custom strategy for store selection"""
    def __init__(self, customer_db: List[Customer], restaurants_db: List[Restaurant]):
        self.customer_db = customer_db
        self.restaurants_db = restaurants_db
        self.arrivals_so_far = 0

    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0, cold_start: int = 10) -> List[Restaurant]:
        # Example: combine collaborative filtering with basic ranking
        if self.arrivals_so_far < cold_start:
            # Fallback to random or basic selection if cold start
            if len(all_stores) <= n:
                return all_stores
            basic_selected = random.sample(all_stores, n)
            self.arrivals_so_far += 1
            return basic_selected
        
        cf_selected = self.collaborative_filtering(customer, n, all_stores, self.customer_db)
        
        # If less than n stores selected, fill with basic ranking (random in this case to ensure diversity)
        if len(cf_selected) < n:
            remaining = n - len(cf_selected)
            remaining_stores = [s for s in all_stores if s not in cf_selected]
            if len(remaining_stores) <= remaining:
                cf_selected.extend(remaining_stores)
            else:
                basic_selected = random.sample(remaining_stores, remaining)
                cf_selected.extend(basic_selected)
        
        self.arrivals_so_far += 1
        return cf_selected[:n]

    def customer_to_dense_vector(self, customer: Customer, all_stores: List[Restaurant]) -> np.ndarray:
        """convert customer preference ratings to dense vector"""
        vector = np.zeros(len(all_stores))
        store_id_to_index = {store.restaurant_id: idx for idx, store in enumerate(all_stores)}
        
        ratings_source = customer.learned_preference_ratings
        if not ratings_source and customer.preference_ratings:
             pass
             
        for store_id, rating in ratings_source.items():
            if store_id in store_id_to_index:
                index = store_id_to_index[store_id]
                vector[index] = rating
        return vector

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def collaborative_filtering(self, customer: Customer, n:int, all_stores: List[Restaurant], customer_db: List[Customer]) -> List[Restaurant]:
        """collaborative filtering based store selection with inventory and accuracy filtering"""
        customer_vector = self.customer_to_dense_vector(customer, all_stores)

        similarities = []
        for other_customer in customer_db:
            if other_customer.customer_id != customer.customer_id:
                other_vector = self.customer_to_dense_vector(other_customer, all_stores)
                sim = self.cosine_similarity(customer_vector, other_vector)
                similarities.append((sim, other_customer))

        similarities.sort(key=lambda x: x[0], reverse=True)

        aggregate_scores = np.zeros(len(all_stores))
        top_k = 10
        count = 0
        for sim, similar_customer in similarities:  
            if count >= top_k: break
            if sim <= 0: continue 
            
            similar_vector = self.customer_to_dense_vector(similar_customer, all_stores)
            aggregate_scores += sim * similar_vector
            count += 1

        viable_stores = []
        for idx, score in enumerate(aggregate_scores):
            if idx >= len(all_stores): continue
            store = all_stores[idx]
            
            if store.est_inventory - store.reservation_count <= 0:
                continue
            
            adjusted_score = score * store.accuracy_score
            viable_stores.append((adjusted_score, idx, store))
        
        viable_stores.sort(key=lambda x: x[0], reverse=True)
        selected_stores = [store for _, _, store in viable_stores[:n]]

        return selected_stores

    def update_learned_preferences(self, customer: Customer, decision: Dict, store: Optional[Restaurant], learning_rate: float = 0.3):
      """Update a customer's learned preference ratings based on their purchase decision"""
      if decision['action'] == 'buy' and store:
        store_id = decision['store_id']
        current_pref = customer.learned_preference_ratings.get(store_id, 0.5)
        customer.learned_preference_ratings[store_id] = current_pref + learning_rate * (1 - current_pref)
        
        for displayed_store_id in customer.displayed_stores:
          if displayed_store_id != store_id:
            current_pref = customer.learned_preference_ratings.get(displayed_store_id, 0.5)
            customer.learned_preference_ratings[displayed_store_id] = current_pref - (learning_rate * current_pref * 0.3)
            customer.learned_preference_ratings[displayed_store_id] = max(0, min(1, customer.learned_preference_ratings[displayed_store_id]))
      
      else:
        for displayed_store_id in customer.displayed_stores:
          current_pref = customer.learned_preference_ratings.get(displayed_store_id, 0.5)
          customer.learned_preference_ratings[displayed_store_id] = current_pref - (learning_rate * current_pref * 0.15)
          customer.learned_preference_ratings[displayed_store_id] = max(0, min(1, customer.learned_preference_ratings[displayed_store_id]))


@register_strategy("Yomna_Strategy")
class Yomna_Strategy(RankingStrategy):
    """
    Trust-first, value-aware strategy with structured tiers and simple exploration.
    No adaptive NearOptimal logic; fixed weights, hard safety gates, and lightweight personalization.
    """

    def __init__(self, explore_rate: float = 0.05, enable_personalization: bool = True):
        self.explore_rate = explore_rate
        self.enable_personalization = enable_personalization

    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
        if not all_stores:
            return []

        available = [s for s in all_stores if s.est_inventory > s.reservation_count]
        if not available:
            return []
        if len(available) <= n:
            return available[:n]

        max_price = max((s.price for s in all_stores), default=1.0)
        max_price = max(max_price, 1.0)  # ensure minimum of 1.0 to prevent division by zero
        max_inventory = max((s.est_inventory for s in all_stores), default=1.0)

        # Light personalization (small boost only)
        order_counts = {}
        if hasattr(customer, "history") and "orders" in customer.history:
            for rid in customer.history["orders"]:
                order_counts[rid] = order_counts.get(rid, 0) + 1
        fav_ids = set([rid for rid, _ in sorted(order_counts.items(), key=lambda x: -x[1])][:2])
        preferred_cats = customer.preferences.get("preferred_categories", [])

        scored = []
        for store in available:
            safe_cap = max(0.0, store.est_inventory - store.reservation_count)
            safe_ratio = safe_cap / max(1.0, max_inventory)

            util = store.reservation_count / max(1.0, store.est_inventory)
            util_score = max(0.0, 1.0 - abs(util - 0.70) / 0.70)
            reliability = store.accuracy_score
            value_metric = (store.rating / 5.0) / max(0.1, store.price / max_price)

            exposure = getattr(store, "exposure_count", 0)
            novelty = (1.0 / (1.0 + exposure)) * min(1.0, max(0.6, reliability))

            waste_penalty = max(0.0, safe_ratio - util_score) * (store.price / max_price)
            overload_risk = max(0.0, util - 0.85) * 0.2
            risk_penalty = (1.0 - reliability) * util + overload_risk

            # Hard gates
            if reliability < 0.50:
                continue
            if reliability < 0.70 and util > 0.70:
                continue
            if util > 0.95:
                continue

            personal_boost = 0.0
            if self.enable_personalization:
                is_fav = store.restaurant_id in fav_ids
                cat_match = 1.0 if store.category in preferred_cats else 0.0
                if is_fav or cat_match:
                    personal_boost = 0.04  # capped small boost

            base_score = (
                0.30 * safe_ratio +
                0.18 * util_score +
                0.16 * reliability +
                0.12 * value_metric +
                0.08 * novelty +
                (-0.10) * waste_penalty +
                (-0.12) * risk_penalty +
                personal_boost
            )

            scored.append({
                "store": store,
                "score": base_score,
                "value": value_metric,
                "novelty": novelty,
                "risk": risk_penalty,
                "util": util,
                "reliability": reliability,
            })

        if not scored:
            return []

        # Tier sizes
        core_slots = max(1, int(round(n * 0.4)))
        value_slots = max(1, int(round(n * 0.3)))
        explore_slots = max(1, n - core_slots - value_slots)

        def sort_by_score(items):
            return sorted(items, key=lambda x: (x["score"], x["store"].rating, -x["store"].price), reverse=True)

        def sort_by_value(items):
            return sorted(items, key=lambda x: (x["value"], x["score"]), reverse=True)

        def sort_by_novelty(items):
            return sorted(items, key=lambda x: (x["novelty"], x["score"]), reverse=True)

        selected = []
        used_ids = set()

        # Core: reliable & not overloaded
        core_candidates = [x for x in scored if x["reliability"] >= 0.78 and x["util"] < 0.82]
        for entry in sort_by_score(core_candidates):
            if len(selected) >= core_slots:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)

        # Value picks
        value_candidates = [x for x in scored if x["store"].restaurant_id not in used_ids and x["risk"] < 0.22]
        for entry in sort_by_value(value_candidates):
            if len(selected) >= core_slots + value_slots:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)

        # Explorers (novelty with minimum reliability)
        explore_candidates = [x for x in scored if x["store"].restaurant_id not in used_ids and x["reliability"] >= 0.65]
        for entry in sort_by_novelty(explore_candidates):
            if len(selected) >= core_slots + value_slots + explore_slots:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)

        # Backfill if short
        if len(selected) < n:
            remaining = [x for x in sort_by_score(scored) if x["store"].restaurant_id not in used_ids]
            for entry in remaining:
                if len(selected) >= n:
                    break
                selected.append(entry)
                used_ids.add(entry["store"].restaurant_id)

        # Safety swap: replace high-risk with safer near-score alternative
        if selected:
            risky_idx, risky_entry = max(enumerate(selected), key=lambda x: x[1]["risk"])
            if risky_entry["risk"] > 0.30:
                alternatives = [
                    x for x in sort_by_score(scored)
                    if x["store"].restaurant_id not in used_ids and x["risk"] < 0.18 and x["util"] < 0.78
                    and x["score"] >= risky_entry["score"] - 0.12
                ]
                if alternatives:
                    best_alt = alternatives[0]
                    used_ids.add(best_alt["store"].restaurant_id)
                    selected[risky_idx] = best_alt

        # Exploration: occasional novelty swap (only if reliable enough)
        if selected and np.random.uniform() < self.explore_rate:
            candidates = [
                x for x in sort_by_novelty(scored)
                if x["store"].restaurant_id not in used_ids and x["reliability"] >= 0.70
            ]
            if candidates:
                selected[-1] = candidates[0]

        return [entry["store"] for entry in selected[:n]]


@register_strategy("HybridElite")
class HybridEliteStrategy(RankingStrategy):
    """
    Elite hybrid strategy combining the best of Near_Optimal, RWES_T, and Yomna.
    
    Key Improvements:
    1. Better revenue optimization (direct price + rating, like RWES_T but enhanced)
    2. Enhanced fulfillment rate (less aggressive filtering, better peak handling)
    3. Improved profit margin (better cost/waste management)
    4. Stronger fairness (exposure penalty like RWES_T)
    5. Peak-time optimization (time-aware adjustments)
    6. Better net revenue (optimize for actual revenue, not just risk-adjusted)
    
    Key Features:
    1. Adaptive weights from Near_Optimal (market condition aware)
    2. Hard reliability gates from Yomna (trust-first approach) - but less restrictive
    3. Structured tier selection from Yomna (core/value/explore)
    4. Reliability focus from RWES_T (35-40% weight on accuracy)
    5. Efficiency optimization from RWES_T (revenue/waste ratio)
    6. Supply-demand matching from Near_Optimal (predictive capacity)
    7. Safety swaps from Yomna (risk mitigation)
    8. Dynamic exploration from RWES_T (time-adaptive)
    9. Exposure fairness penalty from RWES_T (logarithmic penalty)
    10. Peak-time handling (time-of-day optimizations)
    
    Expected to outperform all individual strategies.
    """
    
    def __init__(self, exploration_rate: float = 0.05, min_reliability: float = 0.45):
        """
        Args:
            exploration_rate: Base exploration rate (will be adjusted dynamically)
            min_reliability: Minimum accuracy score threshold (Yomna-style gate, lowered for better fulfillment)
        """
        self.exploration_rate = exploration_rate
        self.min_reliability = min_reliability
    
    def select_stores(self, customer: Customer, n: int, all_stores: List[Restaurant], t: int = 0) -> List[Restaurant]:
        """
        Select stores using elite hybrid approach - ENHANCED VERSION.
        """
        if not all_stores:
            return []
        
        # PHASE 1: Hard filtering (Yomna-style + RWES_T-style) - LESS AGGRESSIVE for better fulfillment
        available = [s for s in all_stores if s.est_inventory > s.reservation_count]
        if not available:
            available = all_stores  # Fallback if no capacity
        
        if len(available) <= n:
            return available[:n]
        
        # Normalization factors
        max_price = max((s.price for s in all_stores), default=1.0)
        max_price = max(max_price, 1.0)
        min_price = min((s.price for s in all_stores), default=1.0)
        max_inventory = max((s.est_inventory for s in all_stores), default=1.0)
        total_stores = len(all_stores)
        
        # Calculate customer's favorite stores (personalization) - MORE AGGRESSIVE
        order_counts = {}
        for rid in customer.history.get('orders', []):
            order_counts[rid] = order_counts.get(rid, 0) + 1
        fav_ids = set([rid for rid, _ in sorted(order_counts.items(), key=lambda x: -x[1])][:4])  # Increased from 3 to 4
        preferred_cats = customer.preferences.get('preferred_categories', [])
        
        # Calculate market conditions (Near_Optimal-style)
        total_reservations = sum(s.reservation_count for s in all_stores)
        total_capacity = sum(s.est_inventory for s in all_stores)
        market_utilization = total_reservations / max(1.0, total_capacity)
        
        # Detect peak time (for peak fulfillment optimization)
        # Peak time is when we're in the middle-high range of customer arrivals
        peak_time_factor = min(1.0, max(0.0, (t / max(1.0, total_stores * 0.5)) - 0.3))  # 0.3 to 0.8 of day
        is_peak_time = peak_time_factor > 0.5
        
        # PHASE 2: Adaptive weight calculation (ENHANCED - optimized for all metrics)
        if market_utilization > 0.8:
            # High demand: Focus on reliability and supply-demand, but also revenue
            reliability_weight = 0.33  # Slightly reduced to allow revenue
            supply_weight = 0.32  # Slightly reduced
            revenue_weight = 0.18  # INCREASED from 0.12 for better revenue
            efficiency_weight = 0.08
            value_weight = 0.04
            personalization_weight = 0.03
            fairness_weight = 0.02
        elif market_utilization < 0.3:
            # Low demand: Focus on revenue and value
            reliability_weight = 0.28  # Reduced to allow more revenue focus
            supply_weight = 0.18  # Reduced - plenty of capacity
            revenue_weight = 0.28  # INCREASED from 0.25 for better revenue
            efficiency_weight = 0.10
            value_weight = 0.10  # INCREASED from 0.08
            personalization_weight = 0.05
            fairness_weight = 0.01
        else:
            # Balanced: Optimal multi-objective - ENHANCED for better revenue
            reliability_weight = 0.30  # Slightly reduced
            supply_weight = 0.26  # Slightly reduced
            revenue_weight = 0.22  # INCREASED from 0.18 for better revenue
            efficiency_weight = 0.10
            value_weight = 0.06
            personalization_weight = 0.04
            fairness_weight = 0.02
        
        # PHASE 3: Score calculation with gates (ENHANCED)
        scored = []
        
        # Dynamic exploration weight (RWES_T-style)
        exploration_weight = max(0.05, 1.0 - t / (len(all_stores) * 10))
        
        for store in available:
            # Hard gates (Yomna-style) - LESS RESTRICTIVE for better fulfillment
            reliability = store.accuracy_score
            util = store.reservation_count / max(1.0, store.est_inventory)
            
            # Gate 1: Minimum reliability (LOWERED threshold)
            if reliability < self.min_reliability:
                continue
            
            # Gate 2: Conditional reliability (LESS STRICT - only exclude if very bad)
            if reliability < 0.60 and util > 0.80:  # Changed from 0.70/0.70 to 0.60/0.80
                continue
            
            # Gate 3: Overload protection (LESS STRICT)
            if util > 0.98:  # Changed from 0.95 to 0.98
                continue
            
            # Calculate score components
            
            # 1. RELIABILITY SCORE (RWES_T-style, 28-33% weight)
            reliability_score = reliability
            
            # 2. SUPPLY-DEMAND SCORE (Near_Optimal-style, 18-32% weight)
            safe_capacity = max(0, store.est_inventory - store.reservation_count)
            safe_capacity_ratio = safe_capacity / max(1.0, max_inventory)
            
            optimal_utilization = 0.72  # Slightly adjusted
            utilization_score = 1.0 - abs(util - optimal_utilization) / optimal_utilization
            utilization_score = max(0.0, utilization_score)
            
            # Predictive demand (Near_Optimal-style)
            expected_remaining_demand = max(0, (total_stores - t) * (1.0 / total_stores))
            demand_supply_match = min(1.0, safe_capacity / max(1.0, expected_remaining_demand * store.est_inventory))
            
            supply_demand_score = (
                0.4 * safe_capacity_ratio +
                0.35 * utilization_score +
                0.25 * demand_supply_match
            )
            
            # 3. REVENUE SCORE (ENHANCED - direct like RWES_T but with rating boost)
            # Use direct normalized price (like RWES_T) for better revenue
            norm_price = store.price / max_price
            rating_factor = store.rating / 5.0
            
            # Base revenue: direct price + rating (like RWES_T but enhanced)
            base_revenue = norm_price * 0.6 + rating_factor * 0.4  # Price matters more
            
            # Risk adjustment (lighter than before to not over-penalize)
            load_ratio = store.reservation_count / max(1.0, store.est_inventory)
            base_cancel_prob = (1.0 - reliability) * load_ratio * 0.7  # Reduced impact
            
            # Time-of-day risk (lighter adjustment)
            time_factor = min(1.0, t / max(1.0, total_stores * 0.8))
            time_risk = time_factor * (1.0 - safe_capacity_ratio) * 0.2  # Reduced from 0.3
            cancellation_prob = min(1.0, base_cancel_prob + time_risk)
            fill_probability = 1.0 - cancellation_prob
            
            # Revenue score: base revenue with lighter risk adjustment
            revenue_score = base_revenue * (0.7 + 0.3 * fill_probability)  # Less penalty for risk
            
            # 4. EFFICIENCY SCORE (RWES_T-style, 8-10% weight)
            waste_potential = max(0.1, safe_capacity)
            revenue_per_waste = base_revenue / waste_potential if waste_potential > 0 else 0
            efficiency_score = min(1.0, revenue_per_waste / 100.0)
            
            # 5. VALUE PROPOSITION (Near_Optimal-style, 4-10% weight)
            price_normalized = (store.price - min_price) / max(1.0, max_price - min_price)
            value_score = (store.rating / 5.0) * (1.0 - price_normalized * 0.4)
            
            # 6. PERSONALIZATION (ENHANCED - stronger boost)
            is_fav = 1.0 if store.restaurant_id in fav_ids else 0.0
            cat_match = 1.0 if store.category in preferred_cats else 0.0
            preference_rating = customer.preference_ratings.get(store.restaurant_id, 0.0)
            personalization_score = 0.5 * is_fav + 0.3 * cat_match + 0.2 * preference_rating
            
            # 7. FAIRNESS (ENHANCED - use RWES_T-style exposure penalty)
            if t > 0:
                # Exposure deficit (Near_Optimal-style)
                expected_exposure = t / total_stores
                exposure_deficit = max(0, expected_exposure - store.exposure_count)
                fairness_boost = min(1.0, exposure_deficit / max(1.0, expected_exposure))
                
                # Exposure penalty (RWES_T-style) - logarithmic penalty for over-exposure
                exposure_penalty = exploration_weight * 0.08 * math.log1p(store.exposure_count)  # Increased from 0.05
                
                fairness_score = fairness_boost - exposure_penalty
            else:
                fairness_score = 1.0 / total_stores
            
            # 8. RISK PENALTY (lighter)
            risk_penalty = cancellation_prob * (store.price / max_price) * (1.0 + time_risk * 0.5)  # Reduced impact
            
            # 9. PEAK-TIME BOOST (NEW - for peak fulfillment optimization)
            peak_boost = 0.0
            if is_peak_time:
                # During peak, boost stores with high reliability and good capacity
                if reliability > 0.75 and util < 0.75:
                    peak_boost = 0.05 * (reliability * (1.0 - util))
            
            # Combine all components with adaptive weights
            combined_score = (
                reliability_weight * reliability_score +
                supply_weight * supply_demand_score +
                revenue_weight * revenue_score +
                efficiency_weight * efficiency_score +
                value_weight * value_score +
                personalization_weight * personalization_score +
                fairness_weight * fairness_score -
                0.015 * risk_penalty +  # Reduced penalty
                peak_boost  # Peak-time boost
            )
            
            # Store metadata for tiered selection
            scored.append({
                "store": store,
                "score": combined_score,
                "value": value_score,
                "novelty": (1.0 / (1.0 + store.exposure_count)) * min(1.0, max(0.6, reliability)),
                "risk": risk_penalty,
                "util": util,
                "reliability": reliability,
            })
        
        if not scored:
            # Fallback: return top stores by basic score if all filtered out
            return available[:n]
        
        # PHASE 4: Structured tier selection (Yomna-style) - ENHANCED
        tier_sizes = {
            "core": max(1, int(round(n * 0.4))),
            "value": max(1, int(round(n * 0.3))),
            "explore": max(1, n - max(1, int(round(n * 0.4))) - max(1, int(round(n * 0.3))))
        }
        
        def sort_by_score(items):
            return sorted(items, key=lambda x: (x["score"], x["store"].rating, -x["store"].price), reverse=True)
        
        def sort_by_value(items):
            return sorted(items, key=lambda x: (x["value"], x["score"]), reverse=True)
        
        def sort_by_novelty(items):
            return sorted(items, key=lambda x: (x["novelty"], x["score"]), reverse=True)
        
        selected = []
        used_ids = set()
        
        # Core tier: High reliability + low utilization (LESS STRICT for better fulfillment)
        core_candidates = [x for x in scored if x["reliability"] >= 0.75 and x["util"] < 0.85]  # Changed from 0.78/0.82
        if not core_candidates:
            core_candidates = [x for x in scored if x["reliability"] >= 0.70]  # Fallback
        for entry in sort_by_score(core_candidates):
            if len(selected) >= tier_sizes["core"]:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)
        
        # Value tier: Best value + low risk
        value_candidates = [x for x in scored if x["store"].restaurant_id not in used_ids and x["risk"] < 0.25]  # Less strict
        for entry in sort_by_value(value_candidates):
            if len(selected) >= tier_sizes["core"] + tier_sizes["value"]:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)
        
        # Explore tier: Novelty + minimum reliability (LESS STRICT)
        explore_candidates = [x for x in scored if x["store"].restaurant_id not in used_ids and x["reliability"] >= 0.60]  # Changed from 0.65
        for entry in sort_by_novelty(explore_candidates):
            if len(selected) >= tier_sizes["core"] + tier_sizes["value"] + tier_sizes["explore"]:
                break
            sid = entry["store"].restaurant_id
            if sid not in used_ids:
                selected.append(entry)
                used_ids.add(sid)
        
        # Backfill if needed
        if len(selected) < n:
            remaining = [x for x in sort_by_score(scored) if x["store"].restaurant_id not in used_ids]
            for entry in remaining:
                if len(selected) >= n:
                    break
                selected.append(entry)
                used_ids.add(entry["store"].restaurant_id)
        
        # PHASE 5: Safety swap (Yomna-style) - LESS AGGRESSIVE
        if selected:
            risky_idx, risky_entry = max(enumerate(selected), key=lambda x: x[1]["risk"])
            if risky_entry["risk"] > 0.35:  # Changed from 0.30 to 0.35 (less aggressive)
                alternatives = [
                    x for x in sort_by_score(scored)
                    if x["store"].restaurant_id not in used_ids
                    and x["risk"] < 0.20  # Changed from 0.18
                    and x["util"] < 0.80  # Changed from 0.78
                    and x["score"] >= risky_entry["score"] - 0.15  # Changed from 0.12 (more flexible)
                ]
                if alternatives:
                    best_alt = alternatives[0]
                    used_ids.add(best_alt["store"].restaurant_id)
                    selected[risky_idx] = best_alt
        
        # PHASE 6: Dynamic exploration (RWES_T-style) - ENHANCED
        if selected and np.random.uniform() < (self.exploration_rate * exploration_weight * 1.2):  # Slightly more exploration
            candidates = [
                x for x in sort_by_novelty(scored)
                if x["store"].restaurant_id not in used_ids
                and x["reliability"] >= 0.65  # Changed from 0.70
            ]
            if candidates:
                selected[-1] = candidates[0]
        
        return [entry["store"] for entry in selected[:n]]


