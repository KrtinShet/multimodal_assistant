import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Tracks performance metrics"""
    component: str
    latency_ms: float
    timestamp: float

class PerformanceMonitor:
    """Monitors and tracks system performance"""

    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self._timers: Dict[str, float] = {}

    def start_timer(self, component: str):
        """Start timing a component"""
        self._timers[component] = time.time()

    def end_timer(self, component: str) -> Optional[float]:
        """End timing and record latency"""
        if component not in self._timers:
            return None

        start_time = self._timers[component]
        latency = (time.time() - start_time) * 1000  # Convert to ms

        metric = PerformanceMetrics(
            component=component,
            latency_ms=latency,
            timestamp=time.time()
        )

        self.metrics[component].append(metric)
        del self._timers[component]

        return latency

    def get_average_latency(self, component: str) -> Optional[float]:
        """Get average latency for component"""
        if component not in self.metrics or not self.metrics[component]:
            return None

        latencies = [m.latency_ms for m in self.metrics[component]]
        return sum(latencies) / len(latencies)

    def print_summary(self):
        """Print performance summary"""
        print("\n=== Performance Summary ===")
        for component, metrics in self.metrics.items():
            if metrics:
                avg_latency = self.get_average_latency(component)
                print(f"{component}: {avg_latency:.2f}ms average")
