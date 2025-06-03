"""
Real-time Analytics Dashboard for monitoring system performance
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DashboardMetrics(BaseModel):
    """Dashboard metrics model."""
    timestamp: str
    throughput: Dict[str, float]
    quality_metrics: Dict[str, float]
    energy_metrics: Dict[str, float]
    maintenance_metrics: Dict[str, float]
    blockchain_metrics: Dict[str, float]

class DashboardApp:
    def __init__(self):
        """Initialize the dashboard application."""
        self.app = FastAPI(title="Circo AI Dashboard")
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files (React build)
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Initialize routes
        self.setup_routes()
        
        logger.info("Initialized DashboardApp")

    def setup_routes(self):
        """Set up API routes."""
        @self.app.get("/")
        async def read_root():
            return {"status": "ok"}

        @self.app.get("/metrics")
        async def get_metrics():
            return await self.get_current_metrics()

        @self.app.get("/metrics/history")
        async def get_metrics_history(hours: int = 24):
            return await self.get_historical_metrics(hours)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Send real-time updates
                    metrics = await self.get_current_metrics()
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.remove(websocket)

    async def get_current_metrics(self) -> Dict:
        """Get current system metrics."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "throughput": {
                    "total_processed": 45670,  # kg
                    "current_rate": 156.7,     # kg/hour
                    "target_rate": 200.0,      # kg/hour
                    "efficiency": 78.35        # percent
                },
                "quality_metrics": {
                    "average_purity": 97.8,    # percent
                    "contamination": 2.2,      # percent
                    "rejection_rate": 1.5,     # percent
                    "accuracy": 99.2           # percent
                },
                "energy_metrics": {
                    "current_consumption": 75.5,  # kW
                    "daily_usage": 1816.0,       # kWh
                    "efficiency": 85.0,          # percent
                    "cost_savings": 22.5         # percent
                },
                "maintenance_metrics": {
                    "system_health": 92.0,     # percent
                    "next_maintenance": "2024-03-15T08:00:00",
                    "predicted_issues": 1,
                    "uptime": 99.8             # percent
                },
                "blockchain_metrics": {
                    "batches_logged": 1567,
                    "carbon_credits": 156.7,    # tons
                    "credit_value": 3134.00,    # USD
                    "verification_rate": 100.0  # percent
                }
            }
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_historical_metrics(self, hours: int) -> Dict:
        """Get historical metrics for specified time period."""
        try:
            # Generate time series data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            timestamps = pd.date_range(start_time, end_time, freq='5min')
            
            return {
                "timestamps": timestamps.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
                "throughput": {
                    "processed": self._generate_timeseries(150, 200, len(timestamps)),
                    "target": [200] * len(timestamps)
                },
                "quality": {
                    "purity": self._generate_timeseries(95, 99, len(timestamps)),
                    "contamination": self._generate_timeseries(1, 5, len(timestamps))
                },
                "energy": {
                    "consumption": self._generate_timeseries(70, 80, len(timestamps)),
                    "efficiency": self._generate_timeseries(80, 90, len(timestamps))
                },
                "maintenance": {
                    "health": self._generate_timeseries(90, 95, len(timestamps))
                }
            }
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _generate_timeseries(self, 
                           min_val: float,
                           max_val: float,
                           length: int) -> List[float]:
        """Generate smooth time series data."""
        base = np.linspace(min_val, max_val, length)
        noise = np.random.normal(0, (max_val - min_val) * 0.05, length)
        return (base + noise).tolist()

    async def broadcast_metrics(self):
        """Broadcast metrics to all connected clients."""
        if not self.active_connections:
            return
            
        metrics = await self.get_current_metrics()
        for connection in self.active_connections:
            try:
                await connection.send_json(metrics)
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")

    def start(self):
        """Start the dashboard application."""
        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    dashboard = DashboardApp()
    dashboard.start() 