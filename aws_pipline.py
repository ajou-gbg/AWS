import os
import socket
import cv2
import numpy as np
import struct
import time
from flask import Flask, send_from_directory, render_template_string
from collections import deque
from movenet import MovenetMPOpenvino
from tst import TSTModel
import torch

# Flask application initialization
app = Flask(__name__)

# Image saving directory
save_dir = './received_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sliding_window = deque(maxlen=15)  # 최대 15개의 프레임 저장

# Initialize MoveNet and TSTModel
input_dim = 6 * 17 * 3  # MoveNet keypoints 차원: 6명의 사람 * 17개의 keypoints * (x, y, confidence)
num_classes = 3
movenet = MovenetMPOpenvino()
tst = TSTModel(input_dim=input_dim, num_classes=num_classes).to('cpu')  # 모델을 CPU에서 실행
tst.load_model("tst.pth")


# HTML template for displaying images
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Received Images</title>
</head>
<body>
    <h1>Received Images</h1>
    {% if images %}
        {% for image in images %}
            <div>
                <h3>{{ image }}</h3>
                <img src="/images/{{ image }}" alt="{{ image }}" style="max-width: 300px; margin-bottom: 20px;">
            </div>
        {% endfor %}
    {% else %}
        <p>No images received yet.</p>
    {% endif %}
</body>
</html>
"""

# Flask route to list images
@app.route('/images', methods=['GET'])
def list_images():
    images = os.listdir(save_dir)
    images.sort()  # Sort images
    return render_template_string(HTML_TEMPLATE, images=images)

# Flask route to serve individual images
@app.route('/images/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(save_dir, filename)

def process_frame(frames):
    keypoints = movenet.run(frames)  # (6, 17, 3) 형태의 keypoints
    # x, y, confidence를 포함하여 (1, 6*17*3) 형태로 변환
    num_frames = keypoints.shape[0]
    keypoints_flattened = keypoints.reshape(num_frames, -1)
    return keypoints_flattened

# Function to process frames in the sliding window
def process_sliding_window(frames):
    
    # MoveNet을 사용하여 키포인트 처리
    keypoints_batch = process_frame(frames)  # (15, 6*17*3)

    # TSTModel 입력 준비
    tst_input = torch.tensor(keypoints_batch, dtype=torch.float32).unsqueeze(0)  # (1, 15, 6*17*3)

    # TSTModel로 예측
    results = tst.inference(tst_input)
    print(f"Prediction results: {results}")
    return results

# Socket server function
def socket_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 5002))  # Listening on port 5002
    server_socket.listen(1)
    print("Socket server is listening on port 5002...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address}")
    buffer = []
    while True:
        try:
            # Receive data size
            packed_size = client_socket.recv(struct.calcsize('>Q'))
            if not packed_size:
                print("No data received for frame size. Continuing...")
                continue

            # Unpack data size
            try:
                msg_size = struct.unpack('>Q', packed_size)[0]
                # print(f"Expecting frame of size: {msg_size} bytes")
            except struct.error as e:
                print(f"Failed to unpack frame size: {e}")
                continue

            # Receive data
            data = b""
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    print("Received incomplete packet. Continuing...")
                    break
                data += packet

            if len(data) != msg_size:
                print(f"Received incomplete frame. Expected {msg_size}, got {len(data)}. Continuing...")
                continue

            # Decode the received image
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if frame is not None:
                # Save the image
                buffer.append(frame)  # Add frame to the buffer

                # When buffer has 5 frames, process and update sliding window
                if len(buffer) == 5:
                    if len(sliding_window) == 15:
                        process_sliding_window(list(sliding_window))  # Process the sliding window
                        # Remove oldest frames to maintain max length of 15
                        for _ in range(len(buffer)):
                            sliding_window.popleft()

                    sliding_window.extend(buffer)  # Add 5 frames to the sliding window
                    buffer.clear()  # Clear the buffer

                current_time = time.time()
                filename = os.path.join(save_dir, f"received_frame_{current_time}.jpg")
                # cv2.imwrite(filename, frame)
                # print(f"Frame saved as {filename}")
            else:
                print("Failed to decode frame. Continuing...")

        except Exception as e:
            print(f"Error during frame handling: {e}")
            continue

    client_socket.close()
    server_socket.close()

# Run both socket server and Flask server
if __name__ == "__main__":
    import threading

    # Start the socket server thread
    socket_thread = threading.Thread(target=socket_server, daemon=True)
    socket_thread.start()

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)