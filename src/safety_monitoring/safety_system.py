"""
Safety monitoring system for the plastic recycling system.
"""

import asyncio
from typing import Dict, List, Optional

import RPi.GPIO as GPIO
from loguru import logger
from .safety_monitor import SafetyMonitor, SafetyMetrics

class SafetySystem:
    """Overall safety management system."""
    
    def __init__(self):
        """Initialize safety system."""
        self.monitors: Dict[str, SafetyMonitor] = {}
        self.global_metrics: List[SafetyMetrics] = []
        self.timeout: float = 0.1
        self.check_interval: float = 0.5
        self.emergency_stop_pin = 21  # GPIO21
        self.safety_sensors = {
            'proximity': 22,  # GPIO22
            'light_curtain': 23,  # GPIO23
            'door_switch': 24,  # GPIO24
            'pressure_mat': 25  # GPIO25
        }
        self.monitoring = False
        self.safety_violations: List[str] = []
        
    def add_monitor(self, name: str, monitor: SafetyMonitor):
        """Add a safety monitor.
        
        Args:
            name: Monitor name
            monitor: SafetyMonitor instance
        """
        self.monitors[name] = monitor
        
    def remove_monitor(self, name: str):
        """Remove a safety monitor.
        
        Args:
            name: Name of monitor to remove
        """
        self.monitors.pop(name, None)
        
    def get_global_safety_score(self) -> float:
        """Calculate global safety score.
        
        Returns:
            Float between 0 and 1 representing overall safety
        """
        if not self.global_metrics:
            return 1.0
            
        scores = [m.safety_score for m in self.global_metrics]
        return sum(scores) / len(scores)
        
    def get_system_report(self) -> str:
        """Generate system-wide safety report.
        
        Returns:
            Formatted report string
        """
        report = ["System Safety Report", "==================="]
        
        report.append(f"\nGlobal Safety Score: {self.get_global_safety_score():.2f}")
        
        for name, monitor in self.monitors.items():
            report.append(f"\nMonitor: {name}")
            report.append("-" * (len(name) + 9))
            report.append(monitor.get_safety_report())
            
        return "\n".join(report)
        
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