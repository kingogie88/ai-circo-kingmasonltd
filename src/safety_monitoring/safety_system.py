"""
Safety monitoring system for the plastic recycling system.
"""

import asyncio
from typing import Dict, List

import RPi.GPIO as GPIO
from loguru import logger

class SafetySystem:
    """Class for monitoring system safety."""
    
    def __init__(self, timeout: float = 0.1, check_interval: float = 0.5):
        """Initialize the safety system.
        
        Args:
            timeout: Timeout for safety checks in seconds
            check_interval: Interval between safety checks in seconds
        """
        self.timeout = timeout
        self.check_interval = check_interval
        self.emergency_stop_pin = 21  # GPIO21
        self.safety_sensors = {
            'proximity': 22,  # GPIO22
            'light_curtain': 23,  # GPIO23
            'door_switch': 24,  # GPIO24
            'pressure_mat': 25  # GPIO25
        }
        self.monitoring = False
        self.safety_violations: List[str] = []
        
    async def initialize(self):
        """Initialize the safety system."""
        try:
            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            
            # Setup emergency stop
            GPIO.setup(self.emergency_stop_pin, GPIO.OUT)
            GPIO.output(self.emergency_stop_pin, GPIO.HIGH)  # Normal state
            
            # Setup safety sensors
            for pin in self.safety_sensors.values():
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            logger.info("Safety system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize safety system: {e}")
            raise
            
    async def start_monitoring(self):
        """Start safety monitoring loop."""
        self.monitoring = True
        try:
            while self.monitoring:
                await self._check_safety()
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"Error in safety monitoring: {e}")
            await self.emergency_stop()
            raise
            
    async def _check_safety(self):
        """Perform safety checks."""
        violations = []
        
        try:
            # Check proximity sensor
            if not GPIO.input(self.safety_sensors['proximity']):
                violations.append("Proximity violation detected")
            
            # Check light curtain
            if not GPIO.input(self.safety_sensors['light_curtain']):
                violations.append("Light curtain breach detected")
            
            # Check door switch
            if not GPIO.input(self.safety_sensors['door_switch']):
                violations.append("Safety door open")
            
            # Check pressure mat
            if not GPIO.input(self.safety_sensors['pressure_mat']):
                violations.append("Unauthorized access detected")
            
            # Update violations list
            self.safety_violations = violations
            
            # Trigger emergency stop if violations found
            if violations:
                logger.warning(f"Safety violations detected: {violations}")
                await self.emergency_stop()
                
        except Exception as e:
            logger.error(f"Error during safety check: {e}")
            await self.emergency_stop()
            raise
            
    async def emergency_stop(self):
        """Trigger emergency stop."""
        try:
            # Activate emergency stop
            GPIO.output(self.emergency_stop_pin, GPIO.LOW)
            
            # Wait for timeout
            await asyncio.sleep(self.timeout)
            
            # Log the event
            logger.warning("Emergency stop triggered")
            
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")
            raise
            
    def get_safety_status(self) -> Dict:
        """Get current safety status.
        
        Returns:
            Dictionary containing safety status information
        """
        return {
            'monitoring_active': self.monitoring,
            'violations': self.safety_violations,
            'sensors': {
                name: GPIO.input(pin)
                for name, pin in self.safety_sensors.items()
            }
        }
            
    async def shutdown(self):
        """Shutdown the safety system."""
        try:
            # Stop monitoring
            self.monitoring = False
            
            # Trigger emergency stop
            await self.emergency_stop()
            
            # Cleanup GPIO
            GPIO.cleanup([
                self.emergency_stop_pin,
                *self.safety_sensors.values()
            ])
            
            logger.info("Safety system shut down")
            
        except Exception as e:
            logger.error(f"Error during safety system shutdown: {e}")
            raise 