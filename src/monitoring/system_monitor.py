"""
System monitoring module for the recycling system.
"""

import asyncio
import time
from typing import Dict

import psutil
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram

class SystemMonitor:
    """Class for monitoring system metrics and performance."""
    
    def __init__(self, update_interval: float, alert_thresholds: Dict):
        """Initialize the system monitor.
        
        Args:
            update_interval: Interval between metric updates in seconds
            alert_thresholds: Dictionary of alert thresholds for various metrics
        """
        self.update_interval = update_interval
        self.alert_thresholds = alert_thresholds
        self.monitoring = False
        
        # Initialize Prometheus metrics
        self.metrics = {
            'cpu_usage': Gauge('recycling_cpu_usage', 'CPU usage percentage'),
            'memory_usage': Gauge('recycling_memory_usage', 'Memory usage percentage'),
            'temperature': Gauge('recycling_temperature', 'System temperature'),
            'items_processed': Counter('recycling_items_processed', 'Number of items processed'),
            'processing_time': Histogram('recycling_processing_time', 'Item processing time',
                                      buckets=[0.1, 0.5, 1.0, 2.0, 5.0]),
            'errors': Counter('recycling_errors', 'Number of system errors',
                            labelnames=['component', 'error_type'])
        }
        
    async def start(self):
        """Start the monitoring system."""
        self.monitoring = True
        try:
            logger.info("Starting system monitoring")
            while self.monitoring:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in monitoring system: {e}")
            raise
            
    async def _update_metrics(self):
        """Update system metrics."""
        try:
            # Update CPU usage
            cpu_percent = psutil.cpu_percent()
            self.metrics['cpu_usage'].set(cpu_percent)
            
            # Update memory usage
            memory = psutil.virtual_memory()
            self.metrics['memory_usage'].set(memory.percent)
            
            # Update temperature (if available)
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and 'coretemp' in temps:
                    temp = temps['coretemp'][0].current
                    self.metrics['temperature'].set(temp)
                    
                    # Check temperature threshold
                    if temp > self.alert_thresholds['temperature']:
                        logger.warning(f"High temperature detected: {temp}°C")
            
            # Check other thresholds
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent}%")
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            self.metrics['errors'].labels(component='monitoring', error_type='metric_update').inc()
            raise
            
    def record_item_processed(self, processing_time: float):
        """Record a processed item.
        
        Args:
            processing_time: Time taken to process the item in seconds
        """
        try:
            self.metrics['items_processed'].inc()
            self.metrics['processing_time'].observe(processing_time)
            
        except Exception as e:
            logger.error(f"Error recording processed item: {e}")
            self.metrics['errors'].labels(component='monitoring', error_type='record_item').inc()
            
    def record_error(self, component: str, error_type: str):
        """Record a system error.
        
        Args:
            component: Component where the error occurred
            error_type: Type of error
        """
        try:
            self.metrics['errors'].labels(component=component, error_type=error_type).inc()
            logger.error(f"Error in {component}: {error_type}")
            
        except Exception as e:
            logger.error(f"Error recording error metric: {e}")
            
    def get_metrics(self) -> Dict:
        """Get current system metrics.
        
        Returns:
            Dictionary containing current metric values
        """
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'items_processed': self.metrics['items_processed']._value.get(),
            'errors': sum(self.metrics['errors']._metrics.values())
        }
            
    async def shutdown(self):
        """Shutdown the monitoring system."""
        try:
            self.monitoring = False
            logger.info("System monitoring shut down")
            
        except Exception as e:
            logger.error(f"Error during monitoring system shutdown: {e}")
            raise 