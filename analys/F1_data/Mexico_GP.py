import fastf1 as f1
import matplotlib.pyplot as plt
import asyncio
import numpy as np

session = f1.get_session(2024, 'Mexico', 'R')
session.load()

nor_lap = session.laps.pick_driver('NOR').pick_lap(10)
ver_lap = session.laps.pick_driver('VER').pick_lap(10)

nor_telemetry = nor_lap.get_telemetry()
ver_telemetry = ver_lap.get_telemetry()

gear_data_nor = nor_telemetry['nGear']
speed_data_nor = nor_telemetry['Speed']
distance_data_nor = nor_telemetry['Distance']
throttle_data_nor = nor_telemetry['Throttle']
brake_data_nor = np.where(nor_telemetry['Brake'] > 0, 100, 0)

gear_data_ver = ver_telemetry['nGear']
speed_data_ver = ver_telemetry['Speed']
distance_data_ver = ver_telemetry['Distance']
throttle_data_ver = ver_telemetry['Throttle']
brake_data_ver = np.where(ver_telemetry['Brake'] > 0, 100, 0)


async def plot_telemetry():
    await asyncio.sleep(0.1)
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axs[0].plot(distance_data_nor, speed_data_nor, label='Max Verstappen', color='blue')
    axs[0].plot(distance_data_ver, speed_data_ver, label='Lando Norris', color='red')
    axs[0].set_ylabel('Speed (km/h)')
    axs[0].set_title('Speed, Throttle, Brake and Gear Over Distance - Mexico GP 2024 (Lap 10)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(distance_data_nor, throttle_data_nor, label='Throttle (Lando Norris)', color='blue')
    axs[1].plot(distance_data_ver, throttle_data_ver, label='Throttle (Max Verstappen)', color='red')
    axs[1].set_ylabel('Throttle (%)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(distance_data_nor, brake_data_nor, label='Brake (Max Verstappen)', color='blue', linestyle='-')
    axs[2].plot(distance_data_ver, brake_data_ver, label='Brake (Lando Norris)', color='red', linestyle='-')
    axs[2].set_ylabel('Brake (Scaled)')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(distance_data_nor, gear_data_nor, label='Gear (Max Verstappen)', color='blue', linestyle='-.')
    axs[3].plot(distance_data_ver, gear_data_ver, label='Gear (Lando Norris)', color='red', linestyle='-.')
    axs[3].set_xlabel('Distance (m)')
    axs[3].set_ylabel('Gear')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


async def main():
    await plot_telemetry()
asyncio.run(main())
