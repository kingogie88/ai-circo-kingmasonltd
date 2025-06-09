"""
API router for the recycling system.
"""

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class PlasticItem(BaseModel):
    """Model for plastic item detection."""
    type: str
    confidence: float
    location: Dict[str, float]

class SystemMetrics(BaseModel):
    """Model for system metrics."""
    cpu_usage: float
    memory_usage: float
    items_processed: int
    errors: int

class SafetyStatus(BaseModel):
    """Model for safety status."""
    monitoring_active: bool
    violations: List[str]
    sensors: Dict[str, bool]

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics():
    """Get system metrics."""
    try:
        from src.main import system
        return system.system_monitor.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/safety", response_model=SafetyStatus)
async def get_safety_status():
    """Get safety system status."""
    try:
        from src.main import system
        return system.safety_system.get_safety_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect")
async def detect_plastic(frame: bytes):
    """Detect plastic in image frame."""
    try:
        from src.main import system
        import numpy as np
        import cv2
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run detection
        result = await system.vision_system.detect_plastic(img)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sort")
async def sort_item(item: PlasticItem):
    """Sort a detected plastic item."""
    try:
        from src.main import system
        await system.robot_controller.sort_item(
            plastic_type=item.type,
            position=item.location
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emergency-stop")
async def trigger_emergency_stop():
    """Trigger emergency stop."""
    try:
        from src.main import system
        await system.safety_system.emergency_stop()
        return {"status": "emergency stop triggered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conveyor/speed")
async def set_conveyor_speed(speed: float):
    """Set conveyor speed."""
    try:
        if not 0 <= speed <= 1:
            raise ValueError("Speed must be between 0 and 1")
            
        from src.main import system
        await system.robot_controller.control_conveyor(speed)
        return {"status": "success", "speed": speed}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 