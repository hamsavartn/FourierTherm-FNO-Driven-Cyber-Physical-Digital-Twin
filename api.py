import os
import sys
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dtr.twin.rc_model import TwinParams, DigitalTwin

app = FastAPI(title="ML Project API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    days: int = 3
    pa: float = 800.0
    pb: float = 850.0
    pc: float = 750.0

@app.post("/simulate")
async def run_simulation(req: SimulationRequest):
    dt = 60  # 1 minute steps to fix RK4 numerical instability
    n_steps = int((req.days * 24 * 3600) / dt)
    
    # Load twin
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'twin.yaml')
    params = TwinParams.from_yaml(config_path)
    twin = DigitalTwin(params)
    
    # Initial steady state assuming 50% load
    P_init = [req.pa * 0.5, req.pb * 0.5, req.pc * 0.5]
    x0 = twin.steady_state(P_init, T_inf=20.0)
    
    P_series = np.array([[req.pa, req.pb, req.pc]] * n_steps)
    
    # Add some daily cyclic variation to T_inf
    t_sec = np.arange(n_steps) * dt
    T_inf_series = 20.0 + 2.0 * np.sin(2 * np.pi * t_sec / 86400.0)
    
    # Add cyclic variation to load (peak in evening, low at night)
    load_multiplier = 0.5 + 0.5 * np.sin(2 * np.pi * t_sec / 86400.0 - np.pi/2)
    P_series = P_series * load_multiplier[:, None]
    
    results = twin.rollout(x0, P_series, T_inf_series, dt)
    
    states = results['states']
    
    # Return time series data and summary
    response = {
        "time_series": [
            {
                "time": i * dt,
                "CoreA": round(float(states[i, 0]), 2),
                "CoreB": round(float(states[i, 1]), 2),
                "CoreC": round(float(states[i, 2]), 2),
                "Sheath": round(float(states[i, 4]), 2),
                "Soil": round(float(states[i, 5]), 2),
                "Moisture": round(float(1.0 - results['s'][i]), 4),
            }
            for i in range(n_steps)
        ],
        "max_temps": {
            "CoreA": round(float(states[:, 0].max()), 1),
            "CoreB": round(float(states[:, 1].max()), 1),
            "CoreC": round(float(states[:, 2].max()), 1),
        },
        "T_max": params.T_max
    }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
