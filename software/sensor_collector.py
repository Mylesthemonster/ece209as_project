import socket
import datetime
import os
import pandas as pd

watch_data_exists = os.path.exists('raw_sensor_data.csv')
if watch_data_exists:
    os.remove('raw_sensor_data.csv')

start_time = datetime.datetime.now()
#end time is 10 sec after the current time
end_time = start_time + datetime.timedelta(seconds=10)

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('192.168.68.124', 59554)
sock.connect(server_address)
try:
    print('Opening socket')
    while end_time > datetime.datetime.now():
        data = sock.recv(4096)
        with open('raw_sensor_data.csv', 'ab') as f:
            f.write(data)
finally:
    print('Closing socket')
    sock.close()
    
df = pd.read_csv('raw_sensor_data.csv') 
df.drop(columns=['loggingTime', 'accelerometerTimestamp_sinceReboot', 'gyroTimestamp_sinceReboot'])