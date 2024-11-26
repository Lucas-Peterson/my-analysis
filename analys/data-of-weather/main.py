import requests
import pandas as pd
import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt


API_KEY = 'YOUR_API_KEY'
CITY = 'London'
URL = f'http://api.openweathermap.org/data/2.5/onecall/timemachine'

# Create a directory for storing data if it doesn't exist
data_dir = Path('weather_data')
data_dir.mkdir(exist_ok=True)

def get_weather_data(date):
    params = {
        'lat': 51.5074,
        'lon': -0.1278,
        'dt': int(date.timestamp()),
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(URL, params=params)
    data = response.json()
    return data

# Get data for the last 5 days
weather_data = []
for i in range(5):
    date = datetime.datetime.now() - datetime.timedelta(days=i)
    data = get_weather_data(date)
    weather_data.append(data)

# Convert data to DataFrame
records = []
for day in weather_data:
    for hour in day['hourly']:
        records.append({
            'date': datetime.datetime.fromtimestamp(hour['dt']),
            'temp': hour['temp'],
            'humidity': hour['humidity'],
            'pressure': hour['pressure']
        })

df = pd.DataFrame(records)
csv_path = data_dir / 'weather_data.csv'
df.to_csv(csv_path, index=False)


# Load data
df = pd.read_csv(csv_path)

# Basic information about the data
print(df.head())
print(df.describe())


# Temperature plot
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['temp'], label='Temperature (C)')
plt.xlabel('Date')
plt.ylabel('Temperature (C)')
plt.title('Temperature over the last 5 days')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plot_path = data_dir / 'temperature_plot.png'
plt.savefig(plot_path)
plt.show()


# Humidity plot
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['humidity'], label='Humidity (%)', color='orange')
plt.xlabel('Date')
plt.ylabel('Humidity (%)')
plt.title('Humidity over the last 5 days')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plot_path = data_dir / 'humidity_plot.png'
plt.savefig(plot_path)
plt.show()


# Pressure plot
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['pressure'], label='Pressure (hPa)', color='green')
plt.xlabel('Date')
plt.ylabel('Pressure (hPa)')
plt.title('Pressure over the last 5 days')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plot_path = data_dir / 'pressure_plot.png'
plt.savefig(plot_path)
plt.show()
