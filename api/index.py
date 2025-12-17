from fastapi import FastAPI
import sys
import os

# Add the parent directory to sys.path so we can import server.py
# Vercel places the project root in /var/task usually, but relative pathing is safer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

# Vercel expects a variable named 'app' (or 'handler') to be the entry point
# We already imported 'app' from server.py
