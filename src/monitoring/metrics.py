"""
Real-time monitoring system for LLM deployments.
"""
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import time
import threading
import queue
from enum import Enum

from ..core.providers import CloudProvider

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOKEN_USAGE = "token_usage"
    ERROR_RATE = "error_rate"
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_USAGE = "memory_usage"
    QUEUE_LENGTH = "queue_length"
    DRIFT = "drift"
    CUSTOM = "custom"


class Alert:
    """Represents an alert triggered by a metric threshold."""
    
    def __init__(
        self,
        metric_name: str,
        threshold: float,
        current_value: float,
        deployment_id: str,
        severity: str = "warning",
    ):
        """
        Initialize an alert.
        
        Args:
            metric_name: Name of the metric that triggered the alert
            threshold: Threshold value that was exceeded
            current_value: Current value of the metric
            deployment_id: ID of the deployment
            severity: Alert severity (info, warning, error, critical)
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.current_value = current_value
        self.deployment_id = deployment_id
        self.severity = severity
        self.timestamp = datetime.now()
        self.resolved = False
        self.resolved_timestamp = None
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolved_timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "deployment_id": self.deployment_id,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp.isoformat() if self.resolved_timestamp else None,
        }


class DriftDetector:
    """Detects drift in model inputs and outputs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the drift detector.
        
        Args:
            config: Drift detector configuration
        """
        self.config = config
        self.reference_data = None
        self.current_window = []
        self.window_size = config.get("window_size", 1000)
        self.drift_threshold = config.get("drift_threshold", 0.05)
        logger.info("Drift detector initialized")
    
    def set_reference_data(self, data: List[Dict[str, Any]]) -> None:
        """
        Set the reference data for drift detection.
        
        Args:
            data: Reference data samples
        """
        self.reference_data = data
        logger.info(f"Reference data set with {len(data)} samples")
    
    def add_sample(self, sample: Dict[str, Any]) -> None:
        """
        Add a sample to the current window.
        
        Args:
            sample: Sample data
        """
        self.current_window.append(sample)
        
        # If window is full, remove oldest sample
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
    
    def detect_drift(self) -> Dict[str, float]:
        """
        Detect drift between reference data and current window.
        
        Returns:
            Dictionary of drift metrics
        """
        if not self.reference_data or len(self.current_window) < self.window_size * 0.5:
            return {"input_drift": 0.0, "output_drift": 0.0}
        
        # In a real implementation, this would use statistical tests
        # to detect drift in the distribution of inputs and outputs
        
        # Simplified example
        input_drift = 0.02  # Simulated value
        output_drift = 0.01  # Simulated value
        
        return {
            "input_drift": input_drift,
            "output_drift": output_drift,
        }


class MetricsCollector:
    """Collects metrics from a deployment."""
    
    def __init__(
        self,
        deployment_id: str,
        provider: CloudProvider,
        config: Dict[str, Any],
    ):
        """
        Initialize the metrics collector.
        
        Args:
            deployment_id: ID of the deployment
            provider: Cloud provider instance
            config: Metrics collector configuration
        """
        self.deployment_id = deployment_id
        self.provider = provider
        self.config = config
        self.metrics = {}
        self.collection_interval = config.get("collection_interval_seconds", 60)
        self.enabled_metrics = config.get("enabled_metrics", [
            MetricType.LATENCY.value,
            MetricType.THROUGHPUT.value,
            MetricType.TOKEN_USAGE.value,
            MetricType.ERROR_RATE.value,
            MetricType.GPU_UTILIZATION.value,
        ])
        
        # Initialize drift detector if enabled
        if MetricType.DRIFT.value in self.enabled_metrics:
            self.drift_detector = DriftDetector(config.get("drift_detector", {}))
        else:
            self.drift_detector = None
        
        # Initialize custom metrics
        self.custom_metrics = config.get("custom_metrics", {})
        
        # Thread for collecting metrics
        self.running = False
        self.collection_thread = None
        
        logger.info(f"Metrics collector initialized for deployment {deployment_id}")
    
    def start(self) -> None:
        """Start collecting metrics."""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection for deployment {self.deployment_id}")
    
    def stop(self) -> None:
        """Stop collecting metrics."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info(f"Stopped metrics collection for deployment {self.deployment_id}")
    
    def _collect_metrics_loop(self) -> None:
        """Background thread for collecting metrics."""
        while self.running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}", exc_info=True)
            
            # Sleep until next collection interval
            time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> None:
        """Collect metrics from the provider."""
        now = datetime.now()
        start_time = (now - timedelta(seconds=self.collection_interval)).isoformat()
        end_time = now.isoformat()
        
        # Collect standard metrics from the provider
        provider_metrics = self.provider.get_metrics(
            deployment_id=self.deployment_id,
            metric_names=self.enabled_metrics,
            start_time=start_time,
            end_time=end_time,
        )
        
        # Update metrics store
        for metric_name, metric_data in provider_metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].extend(metric_data)
            
            # Keep only recent metrics (last 24 hours)
            cutoff = now - timedelta(hours=24)
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]
        
        # Collect drift metrics if enabled
        if self.drift_detector:
            drift_metrics = self.drift_detector.detect_drift()
            
            for metric_name, value in drift_metrics.items():
                drift_metric_name = f"drift_{metric_name}"
                
                if drift_metric_name not in self.metrics:
                    self.metrics[drift_metric_name] = []
                
                self.metrics[drift_metric_name].append({
                    "timestamp": now.isoformat(),
                    "value": value,
                })
        
        logger.debug(f"Collected metrics for deployment {self.deployment_id}")
    
    def get_metrics(
        self,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get collected metrics.
        
        Args:
            metric_names: List of metrics to retrieve (None for all)
            start_time: Start time for metrics query (ISO format)
            end_time: End time for metrics query (ISO format)
            
        Returns:
            Dictionary of metrics data
        """
        result = {}
        
        # Filter metrics by name
        metrics_to_return = metric_names or list(self.metrics.keys())
        
        # Parse time bounds if provided
        start_datetime = datetime.fromisoformat(start_time) if start_time else None
        end_datetime = datetime.fromisoformat(end_time) if end_time else None
        
        # Filter and return metrics
        for metric_name in metrics_to_return:
            if metric_name not in self.metrics:
                result[metric_name] = []
                continue
            
            # Filter by time bounds
            filtered_metrics = self.metrics[metric_name]
            
            if start_datetime:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if datetime.fromisoformat(m["timestamp"]) >= start_datetime
                ]
            
            if end_datetime:
                filtered_metrics = [
                    m for m in filtered_metrics
                    if datetime.fromisoformat(m["timestamp"]) <= end_datetime
                ]
            
            result[metric_name] = filtered_metrics
        
        return result
    
    def add_sample_for_drift_detection(self, sample: Dict[str, Any]) -> None:
        """
        Add a sample for drift detection.
        
        Args:
            sample: Sample data (input and output)
        """
        if self.drift_detector:
            self.drift_detector.add_sample(sample)


class AlertManager:
    """Manages alerts for metric thresholds."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config: Alert manager configuration
        """
        self.config = config
        self.alerts = []
        self.alert_handlers = []
        self.thresholds = config.get("thresholds", {})
        logger.info("Alert manager initialized")
    
    def register_alert_handler(self, handler: callable) -> None:
        """
        Register a handler function for alerts.
        
        Args:
            handler: Function that takes an Alert object
        """
        self.alert_handlers.append(handler)
    
    def check_thresholds(
        self,
        metrics: Dict[str, List[Dict[str, Any]]],
        deployment_id: str,
    ) -> List[Alert]:
        """
        Check if any metrics exceed thresholds.
        
        Args:
            metrics: Dictionary of metrics data
            deployment_id: ID of the deployment
            
        Returns:
            List of triggered alerts
        """
        new_alerts = []
        
        for metric_name, threshold_config in self.thresholds.items():
            if metric_name not in metrics:
                continue
            
            # Get the most recent metric value
            metric_data = metrics[metric_name]
            if not metric_data:
                continue
            
            latest_metric = max(metric_data, key=lambda m: m["timestamp"])
            current_value = latest_metric["value"]
            
            # Check warning threshold
            warning_threshold = threshold_config.get("warning")
            if warning_threshold is not None:
                if (threshold_config.get("direction", "above") == "above" and current_value > warning_threshold) or \
                   (threshold_config.get("direction") == "below" and current_value < warning_threshold):
                    alert = Alert(
                        metric_name=metric_name,
                        threshold=warning_threshold,
                        current_value=current_value,
                        deployment_id=deployment_id,
                        severity="warning",
                    )
                    new_alerts.append(alert)
                    self._handle_alert(alert)
            
            # Check error threshold
            error_threshold = threshold_config.get("error")
            if error_threshold is not None:
                if (threshold_config.get("direction", "above") == "above" and current_value > error_threshold) or \
                   (threshold_config.get("direction") == "below" and current_value < error_threshold):
                    alert = Alert(
                        metric_name=metric_name,
                        threshold=error_threshold,
                        current_value=current_value,
                        deployment_id=deployment_id,
                        severity="error",
                    )
                    new_alerts.append(alert)
                    self._handle_alert(alert)
            
            # Check critical threshold
            critical_threshold = threshold_config.get("critical")
            if critical_threshold is not None:
                if (threshold_config.get("direction", "above") == "above" and current_value > critical_threshold) or \
                   (threshold_config.get("direction") == "below" and current_value < critical_threshold):
                    alert = Alert(
                        metric_name=metric_name,
                        threshold=critical_threshold,
                        current_value=current_value,
                        deployment_id=deployment_id,
                        severity="critical",
                    )
                    new_alerts.append(alert)
                    self._handle_alert(alert)
        
        # Add new alerts to the list
        self.alerts.extend(new_alerts)
        
        return new_alerts
    
    def _handle_alert(self, alert: Alert) -> None:
        """
        Handle a new alert by calling registered handlers.
        
        Args:
            alert: The triggered alert
        """
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {str(e)}", exc_info=True)
    
    def get_active_alerts(
        self,
        deployment_id: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            deployment_id: Filter by deployment ID
            severity: Filter by severity
            
        Returns:
            List of active alerts
        """
        active_alerts = [a for a in self.alerts if not a.resolved]
        
        if deployment_id:
            active_alerts = [a for a in active_alerts if a.deployment_id == deployment_id]
        
        if severity:
            active_alerts = [a for a in active_alerts if a.severity == severity]
        
        return active_alerts
    
    def resolve_alert(self, alert_id: int) -> None:
        """
        Resolve an alert by its index.
        
        Args:
            alert_id: Index of the alert in the alerts list
        """
        if 0 <= alert_id < len(self.alerts):
            self.alerts[alert_id].resolve()
            logger.info(f"Resolved alert {alert_id}")
        else:
            raise ValueError(f"Invalid alert ID: {alert_id}")


class MonitoringSystem:
    """
    Real-time monitoring system for LLM deployments.
    """
    
    def __init__(self, provider: CloudProvider, config: Dict[str, Any]):
        """
        Initialize the monitoring system.
        
        Args:
            provider: Cloud provider instance
            config: Monitoring system configuration
        """
        self.provider = provider
        self.config = config
        self.collectors = {}  # Map of deployment_id to MetricsCollector
        self.alert_manager = AlertManager(config.get("alerts", {}))
        
        # Set up alert handlers
        self._setup_alert_handlers()
        
        # Thread for checking thresholds
        self.threshold_check_interval = config.get("threshold_check_interval_seconds", 300)
        self.running = False
        self.threshold_thread = None
        
        logger.info("Monitoring system initialized")
    
    def _setup_alert_handlers(self) -> None:
        """Set up handlers for alerts."""
        # Log handler
        self.alert_manager.register_alert_handler(
            lambda alert: logger.warning(f"Alert triggered: {alert.metric_name} = {alert.current_value} "
                                        f"(threshold: {alert.threshold}) for deployment {alert.deployment_id}")
        )
        
        # Email handler (would be implemented in a real system)
        if self.config.get("email_alerts", {}).get("enabled", False):
            self.alert_manager.register_alert_handler(self._send_email_alert)
        
        # Webhook handler (would be implemented in a real system)
        if self.config.get("webhook_alerts", {}).get("enabled", False):
            self.alert_manager.register_alert_handler(self._send_webhook_alert)
    
    def _send_email_alert(self, alert: Alert) -> None:
        """
        Send an email alert (placeholder).
        
        Args:
            alert: The triggered alert
        """
        # In a real implementation, this would send an email
        logger.info(f"Would send email alert for {alert.metric_name}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """
        Send a webhook alert (placeholder).
        
        Args:
            alert: The triggered alert
        """
        # In a real implementation, this would send a webhook request
        logger.info(f"Would send webhook alert for {alert.metric_name}")
    
    def start(self) -> None:
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        
        # Start all collectors
        for collector in self.collectors.values():
            collector.start()
        
        # Start threshold checking thread
        self.threshold_thread = threading.Thread(target=self._check_thresholds_loop)
        self.threshold_thread.daemon = True
        self.threshold_thread.start()
        
        logger.info("Monitoring system started")
    
    def stop(self) -> None:
        """Stop the monitoring system."""
        self.running = False
        
        # Stop all collectors
        for collector in self.collectors.values():
            collector.stop()
        
        # Stop threshold checking thread
        if self.threshold_thread:
            self.threshold_thread.join(timeout=5)
        
        logger.info("Monitoring system stopped")
    
    def register_deployment(
        self,
        deployment_id: str,
        model_id: str,
        model_version: str,
    ) -> None:
        """
        Register a deployment for monitoring.
        
        Args:
            deployment_id: ID of the deployment
            model_id: ID of the model
            model_version: Version of the model
        """
        if deployment_id in self.collectors:
            logger.warning(f"Deployment {deployment_id} already registered for monitoring")
            return
        
        # Create a metrics collector for the deployment
        collector_config = self.config.get("collectors", {}).copy()
        collector_config.update({
            "model_id": model_id,
            "model_version": model_version,
        })
        
        collector = MetricsCollector(
            deployment_id=deployment_id,
            provider=self.provider,
            config=collector_config,
        )
        
        self.collectors[deployment_id] = collector
        
        # Start the collector if the monitoring system is running
        if self.running:
            collector.start()
        
        logger.info(f"Registered deployment {deployment_id} for monitoring")
    
    def unregister_deployment(self, deployment_id: str) -> None:
        """
        Unregister a deployment from monitoring.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id not in self.collectors:
            logger.warning(f"Deployment {deployment_id} not registered for monitoring")
            return
        
        # Stop the collector
        self.collectors[deployment_id].stop()
        
        # Remove the collector
        del self.collectors[deployment_id]
        
        logger.info(f"Unregistered deployment {deployment_id} from monitoring")
    
    def update_deployment(self, deployment_id: str) -> None:
        """
        Update a deployment's monitoring configuration.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id not in self.collectors:
            logger.warning(f"Deployment {deployment_id} not registered for monitoring")
            return
        
        # Restart the collector to pick up any configuration changes
        self.collectors[deployment_id].stop()
        self.collectors[deployment_id].start()
        
        logger.info(f"Updated monitoring for deployment {deployment_id}")
    
    def _check_thresholds_loop(self) -> None:
        """Background thread for checking metric thresholds."""
        while self.running:
            try:
                self._check_all_thresholds()
            except Exception as e:
                logger.error(f"Error checking thresholds: {str(e)}", exc_info=True)
            
            # Sleep until next check interval
            time.sleep(self.threshold_check_interval)
    
    def _check_all_thresholds(self) -> None:
        """Check thresholds for all deployments."""
        for deployment_id, collector in self.collectors.items():
            # Get recent metrics
            metrics = collector.get_metrics()
            
            # Check thresholds
            self.alert_manager.check_thresholds(metrics, deployment_id)
    
    def get_metrics(
        self,
        deployment_id: str,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get metrics for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            metric_names: List of metrics to retrieve (None for all)
            start_time: Start time for metrics query (ISO format)
            end_time: End time for metrics query (ISO format)
            
        Returns:
            Dictionary of metrics data
        """
        if deployment_id not in self.collectors:
            raise ValueError(f"Deployment {deployment_id} not registered for monitoring")
        
        return self.collectors[deployment_id].get_metrics(
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time,
        )
    
    def get_alerts(
        self,
        deployment_id: Optional[str] = None,
        active_only: bool = True,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get alerts for deployments.
        
        Args:
            deployment_id: Filter by deployment ID (None for all)
            active_only: Whether to return only active alerts
            severity: Filter by severity
            
        Returns:
            List of alert dictionaries
        """
        if active_only:
            alerts = self.alert_manager.get_active_alerts(
                deployment_id=deployment_id,
                severity=severity,
            )
        else:
            alerts = self.alert_manager.alerts
            
            if deployment_id:
                alerts = [a for a in alerts if a.deployment_id == deployment_id]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
        
        return [alert.to_dict() for alert in alerts]
    
    def resolve_alert(self, alert_id: int) -> None:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
        """
        self.alert_manager.resolve_alert(alert_id)
    
    def add_sample_for_drift_detection(
        self,
        deployment_id: str,
        sample: Dict[str, Any],
    ) -> None:
        """
        Add a sample for drift detection.
        
        Args:
            deployment_id: ID of the deployment
            sample: Sample data (input and output)
        """
        if deployment_id not in self.collectors:
            logger.warning(f"Deployment {deployment_id} not registered for monitoring")
            return
        
        self.collectors[deployment_id].add_sample_for_drift_detection(sample)
