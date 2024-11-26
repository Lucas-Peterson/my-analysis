import fastf1 as f1
import matplotlib.pyplot as plt
import asyncio


# Load the race session
session = f1.get_session(2024, 'Monaco', 'R')
session.load()

# Get data for Verstappen, Piastri, and Hamilton
ver_lap = session.laps.pick_driver('VER')
pia_lap = session.laps.pick_driver('PIA')
ham_lap = session.laps.pick_driver('HAM')

ver_laps = ver_lap.pick_fastest()
pia_laps = pia_lap.pick_fastest()
ham_laps = ham_lap.pick_fastest()

ver_telemetry = ver_laps.get_telemetry()
pia_telemetry = pia_laps.get_telemetry()
ham_telemetry = ham_laps.get_telemetry()

# Maximum and minimum speed for Verstappen
max_speed_ver = ver_telemetry['Speed'].max()
min_speed_ver = ver_telemetry['Speed'].min()

# Maximum and minimum speed for Piastri
max_speed_pia = pia_telemetry['Speed'].max()
min_speed_pia = pia_telemetry['Speed'].min()

# Maximum and minimum speed for Hamilton
max_speed_ham = ham_telemetry['Speed'].max()
min_speed_ham = ham_telemetry['Speed'].min()

# Coordinates for maximum and minimum speed
max_speed_ver_dist = ver_telemetry['Distance'][ver_telemetry['Speed'].idxmax()]
min_speed_ver_dist = ver_telemetry['Distance'][ver_telemetry['Speed'].idxmin()]

max_speed_pia_dist = pia_telemetry['Distance'][pia_telemetry['Speed'].idxmax()]
min_speed_pia_dist = pia_telemetry['Distance'][pia_telemetry['Speed'].idxmin()]

max_speed_ham_dist = ham_telemetry['Distance'][ham_telemetry['Speed'].idxmax()]
min_speed_ham_dist = ham_telemetry['Distance'][ham_telemetry['Speed'].idxmin()]


# Asynchronous function for plotting the graph
async def plot_speeds():
    await asyncio.sleep(0.1)  # Small delay for async behavior

    # Plotting the graph
    plt.figure(figsize=(12, 6))

    # Lines for Verstappen
    plt.plot(ver_telemetry['Distance'], ver_telemetry['Speed'], label='Verstappen', color='blue')
    plt.axhline(y=max_speed_ver, color='blue', linestyle='--', label='Max Verstappen Speed', alpha=0.7)
    plt.axhline(y=min_speed_ver, color='blue', linestyle='--', label='Min Verstappen Speed', alpha=0.7)

    # Displaying maximum and minimum speed for Verstappen
    plt.text(max_speed_ver_dist, max_speed_ver, f'{max_speed_ver:.2f} km/h', color='blue', fontsize=10,
             verticalalignment='bottom')
    plt.text(min_speed_ver_dist, min_speed_ver, f'{min_speed_ver:.2f} km/h', color='blue', fontsize=10,
             verticalalignment='top')

    # Lines for Piastri
    plt.plot(pia_telemetry['Distance'], pia_telemetry['Speed'], label='Piastri', color='orange')
    plt.axhline(y=max_speed_pia, color='orange', linestyle='--', label='Max Piastri Speed', alpha=0.7)
    plt.axhline(y=min_speed_pia, color='orange', linestyle='--', label='Min Piastri Speed', alpha=0.7)

    # Displaying maximum and minimum speed for Piastri
    plt.text(max_speed_pia_dist, max_speed_pia, f'{max_speed_pia:.2f} km/h', color='orange', fontsize=10,
             verticalalignment='bottom')
    plt.text(min_speed_pia_dist, min_speed_pia, f'{min_speed_pia:.2f} km/h', color='orange', fontsize=10,
             verticalalignment='top')

    # Lines for Hamilton
    plt.plot(ham_telemetry['Distance'], ham_telemetry['Speed'], label='Hamilton', color='green')
    plt.axhline(y=max_speed_ham, color='green', linestyle='--', label='Max Hamilton Speed', alpha=0.7)
    plt.axhline(y=min_speed_ham, color='green', linestyle='--', label='Min Hamilton Speed', alpha=0.7)

    # Displaying maximum and minimum speed for Hamilton
    plt.text(max_speed_ham_dist, max_speed_ham, f'{max_speed_ham:.2f} km/h', color='green', fontsize=10,
             verticalalignment='bottom')
    plt.text(min_speed_ham_dist, min_speed_ham, f'{min_speed_ham:.2f} km/h', color='green', fontsize=10,
             verticalalignment='top')

    # Graph styling
    plt.title('Speed Comparison - Monaco GP 2024 (Race)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Speed (km/h)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


# Asynchronous function for printing speed data
async def print_speeds():
    fastest_ver_lap_time = ver_laps['LapTime']
    fastest_pia_lap_time = pia_laps['LapTime']
    fastest_ham_lap_time = ham_laps['LapTime']

    # Print max and min speeds for all drivers with colors
    print(f"\033[34mVerstappen's Max Speed: {max_speed_ver:.2f} km/h\033[0m")  # Blue
    print(f"\033[34mVerstappen's Min Speed: {min_speed_ver:.2f} km/h\033[0m")  # Blue
    print(f"\033[34mVerstappen's Fastest Lap Time: {fastest_ver_lap_time}\033[0m")  # Blue

    print(f"\033[33mPiastri's Max Speed: {max_speed_pia:.2f} km/h\033[0m")  # Orange
    print(f"\033[33mPiastri's Min Speed: {min_speed_pia:.2f} km/h\033[0m")  # Orange
    print(f"\033[33mPiastri's Fastest Lap Time: {fastest_pia_lap_time}\033[0m")  # Orange

    print(f"\033[32mHamilton's Max Speed: {max_speed_ham:.2f} km/h\033[0m")  # Green
    print(f"\033[32mHamilton's Min Speed: {min_speed_ham:.2f} km/h\033[0m")  # Green
    print(f"\033[32mHamilton's Fastest Lap Time: {fastest_ham_lap_time}\033[0m")  # Green


# Main function to run asynchronous tasks
async def main():
    await asyncio.gather(print_speeds(), plot_speeds())


# Run the program
asyncio.run(main())
