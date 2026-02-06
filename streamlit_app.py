"""
Streamlit Cloud Entry Point for Oral Cancer Detection AI
This file serves as the entry point for Streamlit Cloud deployment.
"""
import sys
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Import and run the main application
from main import main

if __name__ == "__main__":
    main()
