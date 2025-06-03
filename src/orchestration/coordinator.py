"""
Central Orchestration Hub for coordinating all system components
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import json

from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import pandas as pd
import numpy as np

from ..vision.model import PlasticClassifier
from ..robotics.controller import RoboticController
from ..maintenance.predictor import MaintenancePredictor
from ..energy.optimizer import EnergyOptimizer
from ..blockchain.logger import BlockchainLogger

logger = logging.getLogger(__name__)

class SystemState(BaseModel):
    """System state model."""
    facility_id: str
    operational_status: str
    current_batch: Optional[Dict] = None
    maintenance_status: Dict
    energy_status: Dict
    robot_status: Dict
    vision_status: Dict
    blockchain_status: Dict

class Coordinator:
    def __init__(self, config: Dict):
        """Initialize the orchestration coordinator."""
        self.config = config
        self.facility_id = config["facility_id"]
        
        # Initialize components
        self.vision_system = PlasticClassifier(
            model_path=config["vision"]["model_path"]
        )
        
        self.robot_controller = RoboticController(
            robot_type=config["robotics"]["robot_type"],
            conveyor_ip=config["robotics"]["conveyor_ip"]
        )
        
        self.maintenance_system = MaintenancePredictor(
            model_path=config["maintenance"]["model_path"]
        )
        
        self.energy_optimizer = EnergyOptimizer(
            config=config["energy"]
        )
        
        self.blockchain_logger = BlockchainLogger(
            network_url=config["blockchain"]["network_url"],
            contract_address=config["blockchain"]["contract_address"],
            private_key=config["blockchain"]["private_key"]
        )
        
        # Initialize FastAPI app
        self.app = FastAPI(title="Circo AI Orchestrator")
        self.setup_routes()
        
        # Initialize state
        self.system_state = SystemState(
            facility_id=self.facility_id,
            operational_status="initializing",
            maintenance_status={},
            energy_status={},
            robot_status={},
            vision_status={},
            blockchain_status={}
        )
        
        # Active connections
        self.active_connections: List[WebSocket] = []
        
        logger.info(f"Initialized Coordinator for facility {self.facility_id}")

    def setup_routes(self):
        """Set up API routes."""
        @self.app.get("/status")
        async def get_status():
            return self.system_state.dict()

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.process_websocket_message(websocket, data)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.remove(websocket)

    async def start(self):
        """Start the orchestration system."""
        try:
            # Initialize connections
            if not await self.initialize_connections():
                raise Exception("Failed to initialize connections")
            
            # Start main control loop
            self.system_state.operational_status = "running"
            await self.broadcast_state()
            
            # Start background tasks
            asyncio.create_task(self.monitoring_loop())
            asyncio.create_task(self.optimization_loop())
            
            logger.info("System started successfully")
            
        except Exception as e:
            logger.error(f"System startup failed: {e}")
            self.system_state.operational_status = "error"
            await self.broadcast_state()

    async def initialize_connections(self) -> bool:
        """Initialize connections to all subsystems."""
        try:
            # Connect to robot
            if not self.robot_controller.connect():
                raise Exception("Robot connection failed")
            
            # Test vision system
            test_image = self.get_test_image()
            if not self.vision_system.predict(test_image):
                raise Exception("Vision system test failed")
            
            # Initialize other subsystems
            # (These don't require active connections)
            
            return True
            
        except Exception as e:
            logger.error(f"Connection initialization failed: {e}")
            return False

    async def monitoring_loop(self):
        """Background task for system monitoring."""
        while True:
            try:
                # Update component status
                self.system_state.maintenance_status = (
                    self.maintenance_system.predict_failures(
                        self.get_current_sensor_data()
                    )
                )
                
                self.system_state.energy_status = (
                    self.energy_optimizer.get_energy_metrics()
                )
                
                self.system_state.robot_status = {
                    "connected": self.robot_controller.robot_connected,
                    "position": "home",  # This would come from actual robot
                    "status": "idle"
                }
                
                # Broadcast updated state
                await self.broadcast_state()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def optimization_loop(self):
        """Background task for system optimization."""
        while True:
            try:
                # Get current system state
                current_state = self.get_current_sensor_data()
                
                # Optimize energy consumption
                energy_params = self.energy_optimizer.optimize_energy_consumption(
                    current_state,
                    production_target=1000  # kg/hour
                )
                
                # Apply optimization parameters
                self.robot_controller.control_conveyor(
                    energy_params["conveyor_speed"]
                )
                
                # Log optimization results
                logger.info(f"Applied optimization parameters: {energy_params}")
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(120)  # Wait longer on error

    async def process_batch(self, image: np.ndarray) -> Dict:
        """
        Process a single batch of plastic waste.
        
        Args:
            image: Image of the plastic waste
            
        Returns:
            Dict with processing results
        """
        try:
            # Detect plastic type
            detections = self.vision_system.predict(image)
            if not detections:
                raise Exception("No plastic detected")
            
            # Get primary detection
            detection = detections[0]
            
            # Analyze quality
            quality_metrics = self.vision_system.analyze_quality(
                image,
                detection
            )
            
            # Move robot to pick position
            success = self.robot_controller.move_to_position(
                self._calculate_pick_pose(detection["bbox"])
            )
            if not success:
                raise Exception("Robot movement failed")
            
            # Log to blockchain
            batch_data = {
                "material_type": detection["class"],
                "quantity": quality_metrics["size"]["area_pixels"] / 1000,
                "quality": 1 - quality_metrics["contamination_level"],
                "facility_id": self.facility_id,
                "energy_used": self.energy_optimizer.get_energy_metrics()["current_consumption_kw"],
                "carbon_credits": self._calculate_carbon_credits(detection["class"])
            }
            
            tx_hash = self.blockchain_logger.log_recycling_batch(batch_data)
            
            return {
                "status": "success",
                "detection": detection,
                "quality": quality_metrics,
                "batch_data": batch_data,
                "tx_hash": tx_hash
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {"status": "error", "message": str(e)}

    async def broadcast_state(self):
        """Broadcast system state to all connected clients."""
        if not self.active_connections:
            return
            
        state_data = self.system_state.dict()
        for connection in self.active_connections:
            try:
                await connection.send_json(state_data)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")

    def get_current_sensor_data(self) -> pd.DataFrame:
        """Get current sensor readings from all systems."""
        # This would typically read from actual sensors
        # For now, return dummy data
        return pd.DataFrame({
            "temperature": [25.5],
            "humidity": [45.0],
            "vibration": [0.15],
            "power_consumption": [75.5],
            "conveyor_speed": [0.8],
            "robot_position": [0.0],
            "air_quality": [95.0],
            "noise_level": [65.0]
        })

    def get_test_image(self) -> np.ndarray:
        """Get test image for vision system."""
        # This would typically load a real test image
        return np.zeros((640, 640, 3), dtype=np.uint8)

    def _calculate_pick_pose(self, bbox: List[float]) -> Dict:
        """Calculate robot pick pose from detection bbox."""
        # This would typically do proper coordinate transformation
        x1, y1, x2, y2 = bbox
        return {
            "position": {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "z": 0.1
            },
            "orientation": {
                "r": 0.0,
                "p": 3.14,
                "y": 0.0
            }
        }

    def _calculate_carbon_credits(self, material_type: str) -> float:
        """Calculate carbon credits for recycled material."""
        # This would typically use real carbon credit calculations
        credits_per_kg = {
            "PET": 2.5,
            "HDPE": 2.0,
            "PVC": 1.5,
            "LDPE": 1.8,
            "PP": 2.2,
            "PS": 1.7,
            "OTHER": 1.0
        }
        return credits_per_kg.get(material_type, 1.0)

    async def shutdown(self):
        """Gracefully shut down the system."""
        try:
            # Stop robot
            self.robot_controller.emergency_stop()
            
            # Close connections
            for connection in self.active_connections:
                await connection.close()
            
            self.system_state.operational_status = "shutdown"
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        asyncio.run(self.shutdown()) 