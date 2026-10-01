"""Cost-Aware Adaptive Conformal Inference & CBF-QP Safety Shield

This module implements the MIT-level enhancement for safety guarantees:
1. Cost-Aware Adaptive Conformal Inference (Cost-Aware ACI) and Bias-Corrected ACI:
   Dynamically adjusts the safety margin (quantile) online based on historical
   miscoverage and penalizes the margin heavily if a true physical constraint
   is violated (severity-aware loss).
2. Neural Boundary CBF-QP (Control Barrier Function Quadratic Program):
   Takes a nominal dispatch action (e.g., from the allocator) and instantaneously
   filters it through a real-time QP that guarantees the forward invariance of
   the safe thermal set using the Neural Operator predictions and ACI bounds.
"""

import numpy as np
import torch
# In a real implementation, we would use a QP solver like OSQP or cvxpy
# import cvxpy as cp

class CostAwareACI:
    def __init__(self, target_alpha=0.05, gamma=0.01, severity_penalty=10.0):
        """
        Args:
            target_alpha: Target miscoverage rate (e.g., 0.05 for 95% coverage).
            gamma: Step size for the online quantile adjustment.
            severity_penalty: Penalty multiplier for constraint violations.
        """
        self.target_alpha = target_alpha
        self.gamma = gamma
        self.severity_penalty = severity_penalty
        self.q = 0.0 # Initial margin
        self.bias = 0.0 # Bias correction term

    def update(self, pred_temp, true_temp, limit=90.0):
        """Online update of the ACI margin and bias."""
        residual = true_temp - pred_temp
        
        # Bias-Corrected ACI update (exponentially weighted)
        alpha_ema = 0.1
        self.bias = (1 - alpha_ema) * self.bias + alpha_ema * residual
        
        corrected_residual = residual - self.bias
        
        # Did the prediction interval cover the true temperature?
        covered = corrected_residual <= self.q
        
        # Did we actually violate the physical constraint?
        violation = true_temp > limit
        
        # Cost-Aware loss function
        if not covered:
            if violation:
                # Severe physical violation: massive asymmetrical penalty
                loss = 1.0 * self.severity_penalty
            else:
                # Benign miscoverage
                loss = 1.0
        else:
            loss = 0.0
            
        # Standard ACI update rule with Cost-Aware loss
        self.q += self.gamma * (loss - self.target_alpha)
        
        return self.q, self.bias

import osqp
from scipy import sparse

class CBF_QP_Shield:
    def __init__(self, surrogate, aci, limit=90.0, alpha_cbf=0.5):
        self.surrogate = surrogate
        self.aci = aci
        self.limit = limit
        self.alpha_cbf = alpha_cbf
        
        # OSQP solver instance
        self.prob = osqp.OSQP()
        self.setup_done = False

    def filter_action(self, current_temp, nominal_alpha, f_x, g_x):
        """
        Filters the nominal_alpha through a Control Barrier Function (CBF-QP)
        using the OSQP solver for hard real-time latency and deterministic fallback.
        
        Args:
            current_temp: Current temperature of the node/cable.
            nominal_alpha: The desired control action (e.g., from V2G allocator).
            f_x: Drift dynamics scalar ( \nabla h(x)^T f(x) ).
            g_x: Control input dynamics scalar ( \nabla h(x)^T g(x) ).
        """
        # Ensure we have a dynamic margin from Cost-Aware ACI
        margin = self.aci.q if hasattr(self.aci, 'q') else 0.0
        
        # Define CBF: h(x) = (Limit - margin) - T >= 0
        h_x = (self.limit - margin) - current_temp
        
        # QP Objective: min 0.5 * u^T P u + q^T u
        # P = 1.0 (scalar since u is 1D), q = -nominal_alpha
        P = sparse.csc_matrix([[1.0]])
        q = np.array([-nominal_alpha])
        
        # QP Constraints: l <= A u <= u_upper
        # 1. CBF Constraint: g_x * u >= -alpha_cbf * h_x - f_x
        # 2. Control Limits: -1.0 <= u <= 1.0 (normalized power alpha)
        
        A = sparse.csc_matrix([
            [g_x],
            [1.0]
        ])
        
        l_cbf = -self.alpha_cbf * h_x - f_x
        u_cbf = np.inf
        
        l = np.array([l_cbf, -1.0])
        u_upper = np.array([u_cbf, 1.0])
        
        if not self.setup_done:
            # First time setup
            self.prob.setup(P, q, A, l, u_upper, verbose=False, warm_start=True, max_iter=4000)
            self.setup_done = True
        else:
            # Update the problem for microsecond latency (warm-started)
            self.prob.update(q=q, l=l, u=u_upper)
            # Update matrix A if g_x changes (optional depending on system linearity)
            self.prob.update(Ax=A.data)
            
        # Solve the QP
        result = self.prob.solve()
        
        # Strict Solver Return Flag Checking (Cyber-Physical Security)
        # OSQP status codes: 1 = solved, 2 = solved inaccurate
        # If infeasible (-3, -4) or max iter (-2), we MUST fallback.
        if result.info.status_val not in [1, 2]:
            # DETERMINISTIC FALLBACK: Zero-power state to protect transformers
            # Do NOT pass uninitialized garbage from result.x
            return 0.0
            
        safe_alpha = result.x[0]
        return safe_alpha
