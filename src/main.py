"""
Main entry point for the AI-Powered Plastic Recycling System.
"""

import asyncio
import logging
from pathlib import Path

import yaml
from fastapi import FastAPI
from loguru import logger
from prometheus_client import start_http_server

from src.api.router import api_router
from src.dashboard.app import create_dashboard
from src.monitoring.system_monitor import SystemMonitor
from src.robotics.controller import RobotController
from src.safety_monitoring.safety_system import SafetySystem
from src.vision.plastic_detector import PlasticDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("logs/system.log", rotation="1 day")

class RecyclingSystem:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.app = FastAPI(title="Plastic Recycling System API")
        self.setup_components()

    def _load_config(self, config_path: str) -> dict:
        """Load system configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file) as f:
            return yaml.safe_load(f)

    def setup_components(self):
        """Initialize all system components."""
        # Initialize vision system
        self.vision_system = PlasticDetector(
            model_path=self.config["vision"]["model_path"],
            confidence_threshold=self.config["vision"]["confidence_threshold"]
        )

        # Initialize robotics
        self.robot_controller = RobotController(
            arm_config=self.config["robotics"]["arm_config"],
            conveyor_config=self.config["robotics"]["conveyor"]
        )

        # Initialize safety system
        self.safety_system = SafetySystem(
            timeout=self.config["safety"]["emergency_stop_timeout"],
            check_interval=self.config["safety"]["sensor_check_interval"]
        )

        # Initialize monitoring
        self.system_monitor = SystemMonitor(
            update_interval=self.config["monitoring"]["update_interval"],
            alert_thresholds=self.config["monitoring"]["alert_thresholds"]
        )

        # Setup API routes
        self.app.include_router(api_router)

    async def startup(self):
        """Start all system components."""
        try:
            # Start Prometheus metrics server
            start_http_server(8000)
            logger.info("Started Prometheus metrics server")

            # Initialize vision system
            await self.vision_system.initialize()
            logger.info("Vision system initialized")

            # Initialize robot controller
            await self.robot_controller.initialize()
            logger.info("Robot controller initialized")

            # Start safety monitoring
            await self.safety_system.start_monitoring()
            logger.info("Safety system active")

            # Start system monitoring
            await self.system_monitor.start()
            logger.info("System monitoring active")

            # Start dashboard
            dashboard_port = self.config["dashboard"]["port"]
            create_dashboard(port=dashboard_port)
            logger.info(f"Dashboard available at http://localhost:{dashboard_port}")

        except Exception as e:
            logger.error(f"Error during system startup: {e}")
            raise

    async def shutdown(self):
        """Gracefully shut down all system components."""
        try:
            await self.vision_system.shutdown()
            await self.robot_controller.shutdown()
            await self.safety_system.shutdown()
            await self.system_monitor.shutdown()
            logger.info("System shutdown complete")
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            raise

async def main():
    """Main entry point for the recycling system."""
    system = RecyclingSystem()
    
    try:
        await system.startup()
        
        # Keep the system running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        await system.shutdown()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await system.shutdown()
        raise

if __name__ == "__main__":
    asyncio.run(main()) 