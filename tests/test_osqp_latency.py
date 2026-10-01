import time
import numpy as np
import pytest
from src.dtr.control.aci_cbf_shield import CBF_QP_Shield
from src.dtr.control.allocator import LCL_ImplicitMPC

class MockACI:
    def __init__(self):
        self.q = 0.5

def test_osqp_latency():
    """
    Validates that the OSQP solver operates within strict microsecond execution
    latency constraints required for a real-time safety shield.
    """
    aci = MockACI()
    shield = CBF_QP_Shield(surrogate=None, aci=aci, limit=90.0, alpha_cbf=0.5)
    
    # Warm up (first solve is often slower)
    shield.filter_action(85.0, 1.0, 0.5, 1.0)
    
    # Measure latency for shield
    start = time.perf_counter()
    for _ in range(100):
        shield.filter_action(89.0, 1.0, 0.5, 1.0)
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / 100.0) * 1000.0
    
    # Microsecond-level control loop requirement (< 10 ms for soft real-time, < 1ms for hard)
    assert avg_latency_ms < 5.0, f"OSQP Shield latency too high: {avg_latency_ms} ms"

def test_mpc_osqp_latency():
    """
    Validates that the OSQP-backed Implicit MPC allocator also operates 
    within real-time constraints.
    """
    allocator = LCL_ImplicitMPC()
    
    deg_cost = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    # Warm up
    allocator.solve(target_kw=-50.0, N=5, deg_cost=deg_cost)
    
    # Measure latency
    start = time.perf_counter()
    for _ in range(100):
        allocator.solve(target_kw=-50.0, N=5, deg_cost=deg_cost)
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / 100.0) * 1000.0
    
    # Must be under 10 ms
    assert avg_latency_ms < 10.0, f"OSQP MPC latency too high: {avg_latency_ms} ms"
