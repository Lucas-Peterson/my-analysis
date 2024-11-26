import fastf1 as f1
import matplotlib.pyplot as plt
import asyncio
import numpy as np

session = f1.get_session(2024, 'Monaco', 'R')
session.load()

# Extracting data for the fastest lap of driver Verstappen
ver_laps = session.laps.pick_driver('VER').pick_fastest()

# Extracting telemetry for Verstappen's fastest lap
ver_telemetry = ver_laps.get_telemetry()

# Extracting gear, speed, distance, throttle, and brake data
gear_data = ver_telemetry['nGear']  # Gear
speed_data = ver_telemetry['Speed']  # Speed
distance_data = ver_telemetry['Distance']  # Distance
throttle_data = ver_telemetry['Throttle']  # Throttle

# Transforming brake data: any brake value > 0 is displayed as 100%, 0 remains 0
brake_data = np.where(ver_telemetry['Brake'] > 0, 100, 0)  # Scaling brake

# Asynchronous function to plot four graphs in one window
async def plot_verstappen_telemetry():
    await asyncio.sleep(0.1)  # Small delay for asynchronicity

    # Create a figure with four subplots (4 rows, 1 column)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # First graph: Speed
    ax1.set_ylabel('Speed (km/h)', color='blue')
    ax1.plot(distance_data, speed_data, label='Speed', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Verstappen Speed, Throttle, Brake and Gear Over Distance - Monaco GP 2024')
    ax1.grid(True)

    # Second graph: Throttle
    ax2.set_ylabel('Throttle (%)', color='green')
    ax2.plot(distance_data, throttle_data, label='Throttle', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.grid(True)

    # Third graph: Scaled Brake
    ax3.set_ylabel('Brake (Scaled)', color='red')
    ax3.plot(distance_data, brake_data, label='Brake', color='red', linestyle='-', linewidth=2)
    ax3.tick_params(axis='y', labelcolor='red')
    ax3.grid(True)

    # Fourth graph: Gear
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Gear', color='orange')
    ax4.plot(distance_data, gear_data, label='Gear', color='orange', linestyle='-.')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax4.grid(True)

    # Adjust layout of the plots
    plt.tight_layout()

    # Display the plots
    plt.show()

# Main asynchronous function
async def main():
    await plot_verstappen_telemetry()

# Run the program
asyncio.run(main())
