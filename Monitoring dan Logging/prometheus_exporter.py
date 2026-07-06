from prometheus_client import start_http_server
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Summary

import psutil
import time

REQUEST_COUNT = Counter(
    "request_total",
    "Total Request"
)



CPU_USAGE = Gauge(
    "cpu_usage_percent",
    "CPU Usage"
)

MEMORY_USAGE = Gauge(
    "memory_usage_percent",
    "Memory Usage"
)

LATENCY = Summary(
    "prediction_latency_seconds",
    "Prediction Latency"
)


@LATENCY.time()
def collect():

    REQUEST_COUNT.inc()

    CPU_USAGE.set(
        psutil.cpu_percent()
    )

    MEMORY_USAGE.set(
        psutil.virtual_memory().percent
    )


if __name__ == "__main__":

    start_http_server(8001)

    while True:

        collect()

        time.sleep(5)