import sys
import os
import threading

# Add current directory to path so src can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dashboard.app import app
from src.core.config_manager import ConfigManager

if __name__ == "__main__":
    config = ConfigManager()
    port = config.get("flask_port", 5000)
    print(f"Starting Flask Dashboard on port {port}...")
    print(f"Open http://127.0.0.1:{port} in your browser to view the dashboard.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
