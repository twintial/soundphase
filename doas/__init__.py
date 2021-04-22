import socket
import json
import numpy as np
import re

def vec2theta(vec, r=1):
    vec = np.array(vec)
    # r = np.linalg.norm(vec[0])
    theta = np.zeros((vec.shape[0], 2))
    theta[:, 0] = np.arctan2(vec[:, 1], vec[:, 0])
    theta[:, 1] = np.arcsin(vec[:, 2]/r)
    return theta

buffer_size = 10240
address = ('192.168.0.113', 9001)
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.bind(address)
tcp_socket.listen(1)
print(f'wait')
connection, add = tcp_socket.accept()
print(f'got connected from {add}')
while True:
    buffer = connection.recv(buffer_size)
    if len(buffer) == 0:
        break
    data = bytes.decode(buffer)
    strs = data.split('}\n{')
    for s in strs:
        if s[0] != '{':
            s = '{' + s
        if s[len(s) - 2] != '}':
            s = s + '}'
        location = None
        try:
            data_json = json.loads(s)
            location = data_json['src']
        except Exception as e:
            print(data)
            print('$' * 50)
            print(s)
            print('$' * 50)
            print(e)
            print("\033[1;31m error\033[0m")
            continue
        points = []
        for point in location:
            x = point['x']
            y = point['y']
            z = point['z']
            points.append([x, y, z])
        points = np.array(points)
        print(points)
        theta = vec2theta(points)
        for i, t in enumerate(np.rad2deg(theta)):
            print(f'top {i} theta: ', t)
        print('=' * 50)