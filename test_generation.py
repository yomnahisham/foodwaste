
print("DEBUG: Starting Test Script")
try:
    from restaurant_api import load_store_data
    from data_loader import save_stores_to_csv
    from customer_api import generate_customer
    from data_loader import save_customers_to_csv
    import numpy as np
    
    print("DEBUG: Imports complete")

    # Test Store Generation
    print("DEBUG: Generating Stores...")
    stores = load_store_data(num_stores=2, num_customers=10, seed=42)
    save_stores_to_csv(stores, "test_generated_stores.csv")
    
    # Check if coords are present
    s = stores[0]
    print(f"DEBUG: Store 1 Coords: Lat={getattr(s, 'latitude', 'N/A')}, Long={getattr(s, 'longitude', 'N/A')}")

    # Test Customer Generation
    print("DEBUG: Generating Customers...")
    customers = generate_customer(2, seed=42)
    save_customers_to_csv(customers, stores, "test_generated_customers.csv")
    
    c = customers[0]
    print(f"DEBUG: Customer 1 Coords: Lat={getattr(c, 'latitude', 'N/A')}, Long={getattr(c, 'longitude', 'N/A')}")
    
    print("DEBUG: Test Complete")

except Exception as e:
    print(f"DEBUG: Error: {e}")
    import traceback
    traceback.print_exc()
