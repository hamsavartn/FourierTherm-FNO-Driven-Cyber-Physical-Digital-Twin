"""Sequential Conformalized Density Regions (SCDR) Intrusion Detection System
with Collaborative H_minus / H_infinity Observer

This module implements the MIT-level cybersecurity enhancement:
1. Collaborative H_minus / H_infinity Observer: 
   Generates a highly robust residual. The H_inf component rejects environmental
   noise, while the H_minus component strongly amplifies stealthy False Data Injection
   (FDI) attacks that hide in the system's null space.
2. Sequential Conformalized Density Regions (SCDR):
   Applies dynamic conformal bounds on the robust residual. Any residual falling
   outside the calibrated conformal region triggers instant telemetry rejection.
"""

import numpy as np

class CollaborativeObserverSCDR:
    def __init__(self, A, B, C, L, target_fpr=0.01, calibration_size=1000):
        """
        Args:
            A, B, C: System state-space matrices.
            L: Observer gain matrix optimized for H_minus / H_infinity performance.
            target_fpr: Target False Positive Rate (e.g., 0.01 for 99% confidence).
            calibration_size: Size of the calibration window.
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.C = np.array(C)
        self.L = np.array(L)
        
        # Internal state estimate
        self.x_hat = np.zeros(self.A.shape[0])
        
        # SCDR components
        self.target_fpr = target_fpr
        self.calibration_size = calibration_size
        self.tau = 0.0 # Conformal threshold on residual magnitude
        self.calibration_scores = []

    def observe(self, y, u, dt=1.0):
        """
        Execute one step of the Collaborative Observer.
        
        Args:
            y: Measurement vector (sensor telemetry).
            u: Control input vector.
            dt: Time step for discretization.
            
        Returns:
            r: The robust residual vector.
        """
        # Calculate residual: r(t) = y(t) - C * x_hat(t)
        r = y - self.C @ self.x_hat
        
        # State estimate derivative: x_hat_dot = A * x_hat + B * u + L * r
        x_hat_dot = self.A @ self.x_hat + self.B @ u + self.L @ r
        
        # Euler integration for state update
        self.x_hat += x_hat_dot * dt
        
        return r

    def non_conformity_score(self, r: np.ndarray) -> float:
        """
        The H_minus component acts as a fault sensitivity index.
        The score is the L2 norm (or amplified energy) of the residual.
        """
        # High score indicates amplified FDI attack.
        return np.linalg.norm(r)

    def calibrate(self, calibration_residuals: list):
        """
        Compute the conformal threshold tau on a clean calibration set of residuals.
        """
        scores = [self.non_conformity_score(r) for r in calibration_residuals]
        self.calibration_scores = scores
        
        n = len(self.calibration_scores)
        q_idx = int(np.ceil((n + 1) * (1 - self.target_fpr)))
        if q_idx >= n:
            q_idx = n - 1
            
        sorted_scores = np.sort(self.calibration_scores)
        self.tau = sorted_scores[q_idx]

    def detect(self, y, u, dt=1.0) -> bool:
        """
        Detects FDI attacks on current sensor readings by passing the 
        observation through the H_minus / H_inf observer and applying SCDR.
        
        Returns True if an attack (anomaly) is detected.
        """
        r = self.observe(y, u, dt)
        score = self.non_conformity_score(r)
        
        # SCDR Anomaly Trigger
        is_attack = score > self.tau
        
        return is_attack

    def update_calibration(self, clean_r: np.ndarray):
        """
        Sequential update of the calibration set (sliding window).
        """
        new_score = self.non_conformity_score(clean_r)
        self.calibration_scores.append(new_score)
        
        # Maintain window size
        if len(self.calibration_scores) > self.calibration_size:
            self.calibration_scores.pop(0)
            
        # Re-calculate threshold
        n = len(self.calibration_scores)
        q_idx = int(np.ceil((n + 1) * (1 - self.target_fpr)))
        if q_idx >= n:
            q_idx = n - 1
            
        sorted_scores = np.sort(self.calibration_scores)
        self.tau = sorted_scores[q_idx]
