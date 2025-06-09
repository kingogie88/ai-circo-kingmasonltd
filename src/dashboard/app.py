"""
Dashboard application for the recycling system.
"""

import time
from typing import Dict, List

import plotly.graph_objs as go
import streamlit as st
from loguru import logger

def create_dashboard(port: int = 8501):
    """Create and run the dashboard application.
    
    Args:
        port: Port number for the dashboard
    """
    try:
        st.set_page_config(
            page_title="Plastic Recycling System",
            page_icon="♻️",
            layout="wide"
        )
        
        st.title("♻️ Plastic Recycling System Dashboard")
        
        # Create main sections
        metrics_col, safety_col = st.columns(2)
        
        with metrics_col:
            display_system_metrics()
            
        with safety_col:
            display_safety_status()
            
        # Create processing section
        st.header("Processing Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            display_processing_stats()
            
        with stats_col2:
            display_plastic_distribution()
            
        with stats_col3:
            display_error_rates()
            
        # Create controls section
        st.header("System Controls")
        control_col1, control_col2 = st.columns(2)
        
        with control_col1:
            display_conveyor_controls()
            
        with control_col2:
            display_emergency_controls()
            
        # Auto-refresh
        if st.button("Refresh Data"):
            st.experimental_rerun()
            
        # Add auto-refresh using JavaScript
        st.markdown(
            """
            <script>
                var timeout = setTimeout(function() {
                    window.location.reload();
                }, 5000);
            </script>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        st.error(f"Dashboard error: {str(e)}")

def display_system_metrics():
    """Display system metrics."""
    try:
        from src.main import system
        metrics = system.system_monitor.get_metrics()
        
        st.subheader("System Metrics")
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU Usage",
                f"{metrics['cpu_usage']:.1f}%",
                delta=f"{metrics['cpu_usage'] - 50:.1f}%"
            )
            
        with col2:
            st.metric(
                "Memory Usage",
                f"{metrics['memory_usage']:.1f}%",
                delta=f"{metrics['memory_usage'] - 50:.1f}%"
            )
            
        with col3:
            st.metric(
                "Items Processed",
                metrics['items_processed']
            )
            
        with col4:
            st.metric(
                "Errors",
                metrics['errors'],
                delta=-metrics['errors'],
                delta_color="inverse"
            )
            
        # Create CPU/Memory chart
        chart_data = {
            'Time': [time.strftime('%H:%M:%S')],
            'CPU': [metrics['cpu_usage']],
            'Memory': [metrics['memory_usage']]
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_data['Time'],
            y=chart_data['CPU'],
            name='CPU'
        ))
        fig.add_trace(go.Scatter(
            x=chart_data['Time'],
            y=chart_data['Memory'],
            name='Memory'
        ))
        
        fig.update_layout(
            title='System Resource Usage',
            xaxis_title='Time',
            yaxis_title='Usage %'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")
        st.error("Failed to load system metrics")

def display_safety_status():
    """Display safety system status."""
    try:
        from src.main import system
        status = system.safety_system.get_safety_status()
        
        st.subheader("Safety Status")
        
        # Display monitoring status
        st.write("Monitoring Status:", 
                "🟢 Active" if status['monitoring_active'] else "🔴 Inactive")
        
        # Display violations
        if status['violations']:
            st.error("Safety Violations Detected:")
            for violation in status['violations']:
                st.write(f"⚠️ {violation}")
        else:
            st.success("No Safety Violations")
            
        # Display sensor status
        st.write("Sensor Status:")
        for sensor, active in status['sensors'].items():
            st.write(f"{'🟢' if active else '🔴'} {sensor.replace('_', ' ').title()}")
            
    except Exception as e:
        logger.error(f"Error displaying safety status: {e}")
        st.error("Failed to load safety status")

def display_processing_stats():
    """Display processing statistics."""
    try:
        st.subheader("Processing Performance")
        
        # Create sample processing time chart
        times = list(range(10))
        processing_times = [0.5 + time * 0.05 for time in times]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=processing_times,
            name='Processing Time'
        ))
        
        fig.update_layout(
            title='Item Processing Time',
            xaxis_title='Item Number',
            yaxis_title='Time (s)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying processing stats: {e}")
        st.error("Failed to load processing statistics")

def display_plastic_distribution():
    """Display plastic type distribution."""
    try:
        st.subheader("Plastic Type Distribution")
        
        # Create sample distribution
        plastic_types = ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'OTHER']
        counts = [30, 25, 10, 15, 10, 5, 5]
        
        fig = go.Figure(data=[go.Pie(
            labels=plastic_types,
            values=counts,
            hole=.3
        )])
        
        fig.update_layout(title='Plastic Types Processed')
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying plastic distribution: {e}")
        st.error("Failed to load plastic distribution")

def display_error_rates():
    """Display error rates."""
    try:
        st.subheader("Error Rates")
        
        # Create sample error rate chart
        components = ['Vision', 'Robotics', 'Conveyor', 'Safety']
        error_rates = [2, 1, 3, 0]
        
        fig = go.Figure(data=[go.Bar(
            x=components,
            y=error_rates
        )])
        
        fig.update_layout(
            title='Errors by Component',
            xaxis_title='Component',
            yaxis_title='Error Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error displaying error rates: {e}")
        st.error("Failed to load error rates")

def display_conveyor_controls():
    """Display conveyor control interface."""
    try:
        st.subheader("Conveyor Control")
        
        speed = st.slider(
            "Conveyor Speed",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        if st.button("Set Speed"):
            from src.main import system
            system.robot_controller.control_conveyor(speed)
            st.success(f"Conveyor speed set to {speed}")
            
    except Exception as e:
        logger.error(f"Error displaying conveyor controls: {e}")
        st.error("Failed to load conveyor controls")

def display_emergency_controls():
    """Display emergency control interface."""
    try:
        st.subheader("Emergency Controls")
        
        if st.button("🔴 EMERGENCY STOP", key="emergency_stop"):
            from src.main import system
            system.safety_system.emergency_stop()
            st.error("Emergency stop triggered!")
            
    except Exception as e:
        logger.error(f"Error displaying emergency controls: {e}")
        st.error("Failed to load emergency controls") 