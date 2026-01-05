#!/usr/bin/env python3
"""
Sensor Fusion Dashboard - Main Entry Point
Launches the GUI application from project root
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and run GUI
from gui.gui_app import main

if __name__ == "__main__":
    main()
