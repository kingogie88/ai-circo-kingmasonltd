"""
Main application entry point for Circo AI Recycling System
"""

import os
import logging
import json
import asyncio
from typing import Dict

from orchestration.coordinator import Coordinator
from dashboard.app import DashboardApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """Load system configuration."""
    config_path = os.getenv('CIRCO_CONFIG', 'config/config.json')
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {}

async def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            raise ValueError("Failed to load configuration")
        
        # Initialize components
        coordinator = Coordinator(config)
        dashboard = DashboardApp()
        
        # Start dashboard in background
        dashboard_task = asyncio.create_task(
            dashboard.app.serve(host="0.0.0.0", port=8000)
        )
        
        # Start coordinator
        await coordinator.start()
        
        # Wait for shutdown signal
        try:
            await asyncio.gather(dashboard_task)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await coordinator.shutdown()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise 