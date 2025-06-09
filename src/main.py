"""
Main entry point for the AI-Powered Plastic Recycling System.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI
from loguru import logger
from prometheus_client import start_http_server
import click

from src.api.router import api_router
from src.dashboard.app import create_dashboard
from src.monitoring.system_monitor import SystemMonitor
from src.robotics.controller import RobotController
from src.safety_monitoring.safety_system import SafetySystem
from src.vision.plastic_detector import PlasticDetector
from .vision import ImageProcessor

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

def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

async def initialize_system(
    model_path: Optional[Path] = None,
    arm_config: Optional[dict] = None,
    conveyor_config: Optional[dict] = None
):
    """Initialize the recycling system components."""
    # Set default configurations if not provided
    if arm_config is None:
        arm_config = {"base": 0, "elbow": 0, "wrist": 0}
    if conveyor_config is None:
        conveyor_config = {"speed": 0.5}
    
    # Initialize components
    safety_system = SafetySystem()
    robot_controller = RobotController(arm_config, conveyor_config)
    
    # Initialize safety system first
    await safety_system.initialize()
    await robot_controller.initialize()
    
    return safety_system, robot_controller

@click.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    help="Path to the YOLOv8 model file"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def main(model_path: Optional[str] = None, log_level: str = "INFO"):
    """Run the plastic recycling system."""
    # Set up logging
    setup_logging(log_level)
    logger.info("Starting plastic recycling system")
    
    try:
        # Run the async initialization
        loop = asyncio.get_event_loop()
        safety_system, robot_controller = loop.run_until_complete(
            initialize_system(
                model_path=Path(model_path) if model_path else None
            )
        )
        
        # Start safety monitoring
        loop.run_until_complete(safety_system.start_monitoring())
        
        # Keep the system running
        loop.run_forever()
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        
    except Exception as e:
        logger.error(f"Error during system operation: {e}")
        raise
        
    finally:
        # Clean up
        if 'safety_system' in locals():
            loop.run_until_complete(safety_system.shutdown())
        if 'robot_controller' in locals():
            loop.run_until_complete(robot_controller.shutdown())
        loop.close()

if __name__ == "__main__":
    main() 