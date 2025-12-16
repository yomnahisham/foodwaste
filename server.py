from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import os
import uuid
import json
import numpy as np

# Import simulation logic
# We need to make sure we can import these. 
# Assuming server.py is in the same directory as simulation.py
from simulation import compare_strategies
from data_loader import (
    load_stores_from_csv, 
    load_customers_from_csv,
    save_stores_to_csv,
    save_customers_to_csv
)

app = FastAPI(title="Food Waste Simulation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for temp storage
UPLOAD_DIR = "temp_uploads"
RESULTS_DIR = "simulation_results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class SimulationConfig(BaseModel):
    num_days: int = 10
    num_stores: Optional[int] = 15
    num_customers: Optional[int] = 70
    use_uploaded_data: bool = False
    stores_filename: Optional[str] = None
    customers_filename: Optional[str] = None
    skip_anan: bool = True

# Store latest results in memory for quick access (or could rely on file system)
latest_results = {}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Food Waste Simulation API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file (stores or customers).
    Returns the saved filename to be used in simulation config.
    """
    try:
        # Generate unique filename to prevent collisions/caching issues
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "filename": unique_filename, 
            "original_name": file.filename,
            "path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SimulationResponse(BaseModel):
    status: str
    message: str
    results_id: str

def run_simulation_task(config: SimulationConfig, results_id: str):
    """
    Background task to run the simulation
    """
    global latest_results
    
    try:
        print(f"Starting simulation {results_id} with config: {config}")
        
        stores_csv_path = None
        customers_csv_path = None
        
        if config.use_uploaded_data:
            if config.stores_filename:
                stores_csv_path = os.path.join(UPLOAD_DIR, config.stores_filename)
            if config.customers_filename:
                customers_csv_path = os.path.join(UPLOAD_DIR, config.customers_filename)
        
        # Run the comparison
        results = compare_strategies(
            num_stores=config.num_stores,
            num_customers=config.num_customers,
            stores_csv=stores_csv_path,
            customers_csv=customers_csv_path,
            num_days=config.num_days,
            verbose=True,
            output_dir=os.path.join(RESULTS_DIR, results_id),
            skip_anan=config.skip_anan
        )
        
        # Process results for JSON response (handle numpy types)
        processed_results = sanitize_for_json(results)
        
        latest_results[results_id] = {
            "status": "completed",
            "data": processed_results
        }
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        latest_results[results_id] = {
            "status": "failed",
            "error": str(e)
        }

def sanitize_for_json(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    else:
        return obj

@app.post("/run", response_model=SimulationResponse)
async def run_simulation(config: SimulationConfig, background_tasks: BackgroundTasks):
    """
    Trigger a new simulation run.
    """
    print(f"DEBUG: Endpoint /run called with config: {config}")
    results_id = str(uuid.uuid4())
    latest_results[results_id] = {"status": "running"}
    
    background_tasks.add_task(run_simulation_task, config, results_id)
    
    return {
        "status": "submitted", 
        "message": "Simulation started in background",
        "results_id": results_id
    }

@app.get("/results/{results_id}")
async def get_results(results_id: str):
    """
    Get results for a specific simulation run.
    """
    if results_id not in latest_results:
        # Check if it exists on disk (if server restarted)
        results_path = os.path.join(RESULTS_DIR, results_id, "strategy_comparison.csv")
        if os.path.exists(results_path):
             return {"status": "completed", "message": "Results on disk (TODO: load from disk)"}
        raise HTTPException(status_code=404, detail="Results not found")
        
    return latest_results[results_id]

if __name__ == "__main__":
    import uvicorn
    # Verify we can import critical modules before starting
    try:
        from simulation import compare_strategies
        print("Simulation module loaded successfully")
    except ImportError as e:
        print(f"Error loading simulation module: {e}")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
