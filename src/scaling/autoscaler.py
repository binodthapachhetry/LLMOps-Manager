"""
Auto-scaling mechanism for LLM deployments.
"""
from typing import Dict, Optional, Any, List
import logging
import threading
import time
import queue
from datetime import datetime, timedelta

from ..core.providers import CloudProvider

logger = logging.getLogger(__name__)


class ScalingPolicy:
    """Defines the scaling policy for a deployment."""
    
    def __init__(
        self,
        min_replicas: int = 1,
        max_replicas: int = 10,
        target_gpu_utilization: float = 70.0,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        scale_up_factor: float = 2.0,
        scale_down_factor: float = 0.5,
        cooldown_period_seconds: int = 300,
        max_queue_length: Optional[int] = None,
    ):
        """
        Initialize a scaling policy.
        
        Args:
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_gpu_utilization: Target GPU utilization percentage (0-100)
            scale_up_threshold: GPU utilization threshold for scaling up
            scale_down_threshold: GPU utilization threshold for scaling down
            scale_up_factor: Factor to multiply current replicas when scaling up
            scale_down_factor: Factor to multiply current replicas when scaling down
            cooldown_period_seconds: Cooldown period between scaling actions
            max_queue_length: Maximum queue length before scaling up
        """
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_gpu_utilization = target_gpu_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
        self.cooldown_period_seconds = cooldown_period_seconds
        self.max_queue_length = max_queue_length
        self.last_scale_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the policy to a dictionary."""
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_gpu_utilization": self.target_gpu_utilization,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "scale_up_factor": self.scale_up_factor,
            "scale_down_factor": self.scale_down_factor,
            "cooldown_period_seconds": self.cooldown_period_seconds,
            "max_queue_length": self.max_queue_length,
        }


class DeploymentState:
    """Tracks the state of a deployment for scaling decisions."""
    
    def __init__(self, deployment_id: str, initial_replicas: int = 1):
        """
        Initialize deployment state.
        
        Args:
            deployment_id: ID of the deployment
            initial_replicas: Initial number of replicas
        """
        self.deployment_id = deployment_id
        self.current_replicas = initial_replicas
        self.desired_replicas = initial_replicas
        self.last_scaling_time = None
        self.metrics_history = {
            "gpu_utilization": [],
            "queue_length": [],
            "throughput": [],
            "latency": [],
        }
        self.scaling_history = []
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update metrics history.
        
        Args:
            metrics: Current metrics
        """
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].append({
                    "timestamp": timestamp,
                    "value": value,
                })
                
                # Keep only recent history (last 24 hours)
                cutoff = timestamp - timedelta(hours=24)
                self.metrics_history[metric_name] = [
                    m for m in self.metrics_history[metric_name]
                    if m["timestamp"] >= cutoff
                ]
    
    def record_scaling_action(
        self,
        old_replicas: int,
        new_replicas: int,
        reason: str,
    ) -> None:
        """
        Record a scaling action.
        
        Args:
            old_replicas: Previous replica count
            new_replicas: New replica count
            reason: Reason for scaling
        """
        self.scaling_history.append({
            "timestamp": datetime.now(),
            "old_replicas": old_replicas,
            "new_replicas": new_replicas,
            "reason": reason,
        })
        
        self.last_scaling_time = datetime.now()
        self.current_replicas = new_replicas
    
    def get_recent_metrics(
        self,
        metric_name: str,
        window_seconds: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Get recent metrics within a time window.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            List of recent metric values
        """
        if metric_name not in self.metrics_history:
            return []
        
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        return [
            m for m in self.metrics_history[metric_name]
            if m["timestamp"] >= cutoff
        ]
    
    def get_average_metric(
        self,
        metric_name: str,
        window_seconds: int = 300,
    ) -> Optional[float]:
        """
        Get the average value of a metric within a time window.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            Average metric value, or None if no data
        """
        recent_metrics = self.get_recent_metrics(metric_name, window_seconds)
        
        if not recent_metrics:
            return None
        
        return sum(m["value"] for m in recent_metrics) / len(recent_metrics)
    
    def in_cooldown(self, cooldown_seconds: int) -> bool:
        """
        Check if the deployment is in cooldown period.
        
        Args:
            cooldown_seconds: Cooldown period in seconds
            
        Returns:
            True if in cooldown, False otherwise
        """
        if not self.last_scaling_time:
            return False
        
        return (datetime.now() - self.last_scaling_time).total_seconds() < cooldown_seconds


class AutoScaler:
    """
    Auto-scaling mechanism for LLM deployments.
    """
    
    def __init__(self, provider: CloudProvider, config: Dict[str, Any]):
        """
        Initialize the auto-scaler.
        
        Args:
            provider: Cloud provider instance
            config: Auto-scaler configuration
        """
        self.provider = provider
        self.config = config
        self.deployments = {}  # Map of deployment_id to DeploymentState
        self.policies = {}  # Map of deployment_id to ScalingPolicy
        self.check_interval = config.get("check_interval_seconds", 60)
        
        # Thread for checking scaling conditions
        self.running = False
        self.scaling_thread = None
        
        logger.info("Auto-scaler initialized")
    
    def start(self) -> None:
        """Start the auto-scaler."""
        if self.running:
            return
        
        self.running = True
        self.scaling_thread = threading.Thread(target=self._check_scaling_loop)
        self.scaling_thread.daemon = True
        self.scaling_thread.start()
        
        logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaler."""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
        
        logger.info("Auto-scaler stopped")
    
    def register_deployment(
        self,
        deployment_id: str,
        initial_replicas: int = 1,
        policy: Optional[ScalingPolicy] = None,
    ) -> None:
        """
        Register a deployment for auto-scaling.
        
        Args:
            deployment_id: ID of the deployment
            initial_replicas: Initial number of replicas
            policy: Scaling policy (None for default)
        """
        if deployment_id in self.deployments:
            logger.warning(f"Deployment {deployment_id} already registered for auto-scaling")
            return
        
        # Create deployment state
        self.deployments[deployment_id] = DeploymentState(
            deployment_id=deployment_id,
            initial_replicas=initial_replicas,
        )
        
        # Create scaling policy
        self.policies[deployment_id] = policy or ScalingPolicy(
            min_replicas=self.config.get("default_min_replicas", 1),
            max_replicas=self.config.get("default_max_replicas", 10),
            target_gpu_utilization=self.config.get("default_target_gpu_utilization", 70.0),
        )
        
        logger.info(f"Registered deployment {deployment_id} for auto-scaling")
    
    def unregister_deployment(self, deployment_id: str) -> None:
        """
        Unregister a deployment from auto-scaling.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id not in self.deployments:
            logger.warning(f"Deployment {deployment_id} not registered for auto-scaling")
            return
        
        # Remove deployment state and policy
        del self.deployments[deployment_id]
        del self.policies[deployment_id]
        
        logger.info(f"Unregistered deployment {deployment_id} from auto-scaling")
    
    def update_deployment(self, deployment_id: str) -> None:
        """
        Update a deployment's auto-scaling state.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id not in self.deployments:
            logger.warning(f"Deployment {deployment_id} not registered for auto-scaling")
            return
        
        # Get current replica count from provider
        try:
            status = self.provider.get_deployment_status(deployment_id)
            current_replicas = status.get("instance_count", 1)
            
            # Update deployment state
            self.deployments[deployment_id].current_replicas = current_replicas
            self.deployments[deployment_id].desired_replicas = current_replicas
            
            logger.info(f"Updated auto-scaling state for deployment {deployment_id}")
        except Exception as e:
            logger.error(f"Error updating deployment state: {str(e)}", exc_info=True)
    
    def update_policy(
        self,
        deployment_id: str,
        min_replicas: Optional[int] = None,
        max_replicas: Optional[int] = None,
        target_gpu_utilization: Optional[float] = None,
        max_queue_length: Optional[int] = None,
    ) -> None:
        """
        Update the scaling policy for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            target_gpu_utilization: Target GPU utilization percentage (0-100)
            max_queue_length: Maximum queue length before scaling up
        """
        if deployment_id not in self.policies:
            logger.warning(f"Deployment {deployment_id} not registered for auto-scaling")
            return
        
        policy = self.policies[deployment_id]
        
        # Update policy parameters
        if min_replicas is not None:
            policy.min_replicas = min_replicas
        
        if max_replicas is not None:
            policy.max_replicas = max_replicas
        
        if target_gpu_utilization is not None:
            policy.target_gpu_utilization = target_gpu_utilization
            
            # Update thresholds based on target
            policy.scale_up_threshold = min(100, target_gpu_utilization * 1.15)
            policy.scale_down_threshold = max(0, target_gpu_utilization * 0.5)
        
        if max_queue_length is not None:
            policy.max_queue_length = max_queue_length
        
        logger.info(f"Updated scaling policy for deployment {deployment_id}")
    
    def get_policy(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get the scaling policy for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Scaling policy as a dictionary
        """
        if deployment_id not in self.policies:
            raise ValueError(f"Deployment {deployment_id} not registered for auto-scaling")
        
        return self.policies[deployment_id].to_dict()
    
    def get_scaling_history(self, deployment_id: str) -> List[Dict[str, Any]]:
        """
        Get the scaling history for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            List of scaling actions
        """
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not registered for auto-scaling")
        
        return self.deployments[deployment_id].scaling_history
    
    def _check_scaling_loop(self) -> None:
        """Background thread for checking scaling conditions."""
        while self.running:
            try:
                self._check_all_deployments()
            except Exception as e:
                logger.error(f"Error checking scaling conditions: {str(e)}", exc_info=True)
            
            # Sleep until next check interval
            time.sleep(self.check_interval)
    
    def _check_all_deployments(self) -> None:
        """Check scaling conditions for all deployments."""
        for deployment_id in list(self.deployments.keys()):
            try:
                self._check_deployment_scaling(deployment_id)
            except Exception as e:
                logger.error(f"Error checking scaling for deployment {deployment_id}: {str(e)}", exc_info=True)
    
    def _check_deployment_scaling(self, deployment_id: str) -> None:
        """
        Check scaling conditions for a deployment.
        
        Args:
            deployment_id: ID of the deployment
        """
        deployment = self.deployments[deployment_id]
        policy = self.policies[deployment_id]
        
        # Skip if in cooldown period
        if deployment.in_cooldown(policy.cooldown_period_seconds):
            return
        
        # Get current metrics
        try:
            # In a real implementation, this would get metrics from the provider
            # For simplicity, we'll use simulated metrics
            metrics = self._get_deployment_metrics(deployment_id)
            
            # Update metrics history
            deployment.update_metrics(metrics)
            
            # Check scaling conditions
            self._check_scaling_conditions(deployment_id, metrics)
        except Exception as e:
            logger.error(f"Error getting metrics for deployment {deployment_id}: {str(e)}", exc_info=True)
    
    def _get_deployment_metrics(self, deployment_id: str) -> Dict[str, float]:
        """
        Get current metrics for a deployment.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Dictionary of metrics
        """
        # In a real implementation, this would get metrics from the provider
        # For simplicity, we'll use simulated metrics
        
        # Simulate GPU utilization between 0 and 100
        import random
        gpu_utilization = random.uniform(0, 100)
        
        # Simulate queue length
        queue_length = random.randint(0, 100)
        
        # Simulate throughput (requests per second)
        throughput = random.uniform(1, 100)
        
        # Simulate latency (milliseconds)
        latency = random.uniform(50, 500)
        
        return {
            "gpu_utilization": gpu_utilization,
            "queue_length": queue_length,
            "throughput": throughput,
            "latency": latency,
        }
    
    def _check_scaling_conditions(
        self,
        deployment_id: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Check if scaling is needed based on current metrics.
        
        Args:
            deployment_id: ID of the deployment
            metrics: Current metrics
        """
        deployment = self.deployments[deployment_id]
        policy = self.policies[deployment_id]
        current_replicas = deployment.current_replicas
        
        # Check GPU utilization for scaling
        gpu_utilization = metrics.get("gpu_utilization")
        if gpu_utilization is not None:
            if gpu_utilization > policy.scale_up_threshold:
                # Scale up based on GPU utilization
                new_replicas = min(
                    policy.max_replicas,
                    int(current_replicas * policy.scale_up_factor)
                )
                new_replicas = max(new_replicas, current_replicas + 1)
                
                if new_replicas > current_replicas:
                    self._scale_deployment(
                        deployment_id=deployment_id,
                        new_replicas=new_replicas,
                        reason=f"GPU utilization {gpu_utilization:.1f}% > threshold {policy.scale_up_threshold:.1f}%"
                    )
                    return
            elif gpu_utilization < policy.scale_down_threshold:
                # Scale down based on GPU utilization
                new_replicas = max(
                    policy.min_replicas,
                    int(current_replicas * policy.scale_down_factor)
                )
                new_replicas = min(new_replicas, current_replicas - 1)
                
                if new_replicas < current_replicas:
                    self._scale_deployment(
                        deployment_id=deployment_id,
                        new_replicas=new_replicas,
                        reason=f"GPU utilization {gpu_utilization:.1f}% < threshold {policy.scale_down_threshold:.1f}%"
                    )
                    return
        
        # Check queue length for scaling
        queue_length = metrics.get("queue_length")
        if queue_length is not None and policy.max_queue_length is not None:
            if queue_length > policy.max_queue_length:
                # Scale up based on queue length
                new_replicas = min(
                    policy.max_replicas,
                    int(current_replicas * policy.scale_up_factor)
                )
                new_replicas = max(new_replicas, current_replicas + 1)
                
                if new_replicas > current_replicas:
                    self._scale_deployment(
                        deployment_id=deployment_id,
                        new_replicas=new_replicas,
                        reason=f"Queue length {queue_length} > threshold {policy.max_queue_length}"
                    )
                    return
    
    def _scale_deployment(
        self,
        deployment_id: str,
        new_replicas: int,
        reason: str,
    ) -> None:
        """
        Scale a deployment to a new replica count.
        
        Args:
            deployment_id: ID of the deployment
            new_replicas: New number of replicas
            reason: Reason for scaling
        """
        deployment = self.deployments[deployment_id]
        current_replicas = deployment.current_replicas
        
        if new_replicas == current_replicas:
            return
        
        logger.info(f"Scaling deployment {deployment_id} from {current_replicas} to {new_replicas} replicas: {reason}")
        
        try:
            # Update the deployment using the provider
            self.provider.update_deployment(
                deployment_id=deployment_id,
                instance_count=new_replicas,
            )
            
            # Record the scaling action
            deployment.record_scaling_action(
                old_replicas=current_replicas,
                new_replicas=new_replicas,
                reason=reason,
            )
            
            logger.info(f"Scaled deployment {deployment_id} to {new_replicas} replicas")
        except Exception as e:
            logger.error(f"Error scaling deployment {deployment_id}: {str(e)}", exc_info=True)


class RequestQueueManager:
    """
    Manages request queues for LLM deployments.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the request queue manager.
        
        Args:
            config: Queue manager configuration
        """
        self.config = config
        self.queues = {}  # Map of deployment_id to request queue
        self.max_queue_size = config.get("max_queue_size", 1000)
        self.default_timeout = config.get("default_timeout_seconds", 30)
        
        logger.info("Request queue manager initialized")
    
    def create_queue(self, deployment_id: str) -> None:
        """
        Create a queue for a deployment.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id in self.queues:
            logger.warning(f"Queue already exists for deployment {deployment_id}")
            return
        
        self.queues[deployment_id] = queue.Queue(maxsize=self.max_queue_size)
        
        logger.info(f"Created queue for deployment {deployment_id}")
    
    def delete_queue(self, deployment_id: str) -> None:
        """
        Delete a queue for a deployment.
        
        Args:
            deployment_id: ID of the deployment
        """
        if deployment_id not in self.queues:
            logger.warning(f"No queue exists for deployment {deployment_id}")
            return
        
        del self.queues[deployment_id]
        
        logger.info(f"Deleted queue for deployment {deployment_id}")
    
    def enqueue_request(
        self,
        deployment_id: str,
        request: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Enqueue a request for processing.
        
        Args:
            deployment_id: ID of the deployment
            request: Request data
            timeout: Timeout in seconds (None for default)
            
        Returns:
            True if request was enqueued, False if queue is full
        """
        if deployment_id not in self.queues:
            raise ValueError(f"No queue exists for deployment {deployment_id}")
        
        timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            self.queues[deployment_id].put(request, block=True, timeout=timeout)
            return True
        except queue.Full:
            logger.warning(f"Queue full for deployment {deployment_id}")
            return False
    
    def dequeue_request(
        self,
        deployment_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dequeue a request for processing.
        
        Args:
            deployment_id: ID of the deployment
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Request data, or None if queue is empty
        """
        if deployment_id not in self.queues:
            raise ValueError(f"No queue exists for deployment {deployment_id}")
        
        timeout = timeout if timeout is not None else self.default_timeout
        
        try:
            return self.queues[deployment_id].get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def get_queue_length(self, deployment_id: str) -> int:
        """
        Get the current queue length.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Current queue length
        """
        if deployment_id not in self.queues:
            raise ValueError(f"No queue exists for deployment {deployment_id}")
        
        return self.queues[deployment_id].qsize()
    
    def is_queue_full(self, deployment_id: str) -> bool:
        """
        Check if a queue is full.
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            True if queue is full, False otherwise
        """
        if deployment_id not in self.queues:
            raise ValueError(f"No queue exists for deployment {deployment_id}")
        
        return self.queues[deployment_id].full()
