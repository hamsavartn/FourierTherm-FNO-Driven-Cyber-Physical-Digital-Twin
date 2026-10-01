import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("Generating MIT-Level Visualizations...")
    
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summary_path = os.path.join(root, "outputs", "phase_summary_real.csv")
    out_dir = os.path.join(root, "outputs")
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Please run earlier steps first.")
        return
        
    df = pd.read_csv(summary_path)
    
    # Filter controllers
    no_control = df[df['controller'] == 'no_control']
    static_cap = df[df['controller'] == 'static_cap']
    shield = df[df['controller'] == 'shield']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Maximum Cable Temperature
    ax1.plot(no_control['day'], no_control['max_core'], marker='o', label='No Control (Greedy)', color='red')
    ax1.plot(static_cap['day'], static_cap['max_core'], marker='s', label='Static Limit', color='orange')
    ax1.plot(shield['day'], shield['max_core'], marker='^', label='Implicit MPC Shield', color='green')
    ax1.axhline(y=90.0, color='black', linestyle='--', label='90°C Safety Limit')
    ax1.set_title('Maximum Cable Temperature (°C)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Day')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Plot 2: Total Energy Delivered
    width = 0.25
    x = range(len(no_control['day']))
    ax2.bar([i - width for i in x], no_control['E_delivered_kwh'], width=width, label='No Control (Melt Risk)', color='red')
    ax2.bar(x, static_cap['E_delivered_kwh'], width=width, label='Static Limit (Safe but Slow)', color='orange')
    ax2.bar([i + width for i in x], shield['E_delivered_kwh'], width=width, label='Implicit MPC (Safe & Fast)', color='green')
    ax2.set_title('Total Energy Delivered (kWh)')
    ax2.set_ylabel('Energy (kWh)')
    ax2.set_xlabel('Day')
    ax2.set_xticks(x)
    ax2.set_xticklabels(no_control['day'])
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    out_file = os.path.join(out_dir, "mit_level_comparison.png")
    plt.savefig(out_file, dpi=300)
    print(f"Successfully generated visualization: {out_file}")

if __name__ == "__main__":
    main()
