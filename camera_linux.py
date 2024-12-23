import sys
import ctypes
import cv2
import time
import numpy as np
import os

# Add the path where the shared library is located
sys.path.append('/home/pi/Desktop/TmSDK/SDK/Linux/TmLinux/lib')

# Load the shared library
try:
    tmcore = ctypes.CDLL('/home/pi/Desktop/TmSDK/SDK/Linux/TmLinux/lib/libTmCore.so')
except OSError as e:
    print(f"Failed to load the TmCore library: {e}")
    sys.exit(1)

# Placeholder class to simulate the SDK classes
class TmCamera:
    class CameraManager:
        def get_local_cameras(self):
            # This method should interact with the shared library to get connected cameras
            # Here we simply simulate with a placeholder
            return [CameraInfo()]

    class TmCamera:
        def __init__(self, obj):
            self.obj = obj
            self.camera = None

        def open(self):
            # Open the thermal camera using SDK
            print("Attempting to open USB camera...")
            # 시도 1: 기본 인덱스 0 사용
            self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # V4L2로 시도
            if not self.camera.isOpened():
                print("Failed to open /dev/video0. Trying /dev/video1...")
                # 시도 2: 인덱스 1 사용
                self.camera = cv2.VideoCapture(1, cv2.CAP_V4L2)

            if not self.camera.isOpened():
                print("Could not open any USB camera. Please check the connection.")
                return False

            # Set the desired resolution (해상도 낮추기)
            width = 640  # 낮은 해상도 시도
            height = 480
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            print(f"Camera opened successfully with resolution {width}x{height}.")
            return True

        def query_frame(self, width, height):
            # Query frame from thermal camera with specified dimensions
            print("Attempting to query a frame from the camera...")
            if self.camera:
                ret, frame = self.camera.read()
                if ret:
                    print(f"Original frame size: {frame.shape}")
                    # 프레임의 크기가 작기 때문에 더 큰 크기로 확대
                    enlarged_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                    print("Frame resized to: 640x480.")
                    return enlarged_frame
                else:
                    print("Failed to capture frame. Please ensure the camera is properly connected.")
                    return None
            else:
                print("Camera not opened.")
                return None

        def close(self):
            # Release the camera
            if self.camera:
                self.camera.release()
                print("Camera closed.")

class CameraInfo:
    def get_local_camera_name(self):
        return "USB Thermal Camera"

    def get_local_camera_com_port(self):
        return "USB"

    def get_local_camera_index(self):
        return 0

    @property
    def obj(self):
        return None

def process_frame(frame):
    """
    Convert the thermal frame to temperature values and apply a color map.
    """
    print("Processing frame...")

    # 현재 프레임이 16비트 그레이스케일 형식이라면 이를 8비트로 변환합니다.
    if frame.dtype == np.uint16:
        # 16비트에서 8비트로 변환 (값을 줄여 시각적으로 볼 수 있도록)
        gray_frame = (frame / 256).astype(np.uint8)
        print("Converted 16-bit grayscale to 8-bit.")
    else:
        # 프레임이 이미 8비트라면 그대로 사용
        gray_frame = frame
        print("Frame is already in 8-bit format.")

    # 이미지 크기 확대
    enlarged_frame = cv2.resize(gray_frame, (640, 480), interpolation=cv2.INTER_LINEAR)

    # Normalize temperature data to 8-bit range for visualization
    normalized_frame = cv2.normalize(enlarged_frame, None, 0, 255, cv2.NORM_MINMAX)
    print("Frame normalized for visualization.")

    # Apply a color map to visualize the temperature differences
    # 온도 차이를 명확히 표현하기 위해 JET 컬러맵을 사용
    colored_frame = cv2.applyColorMap(normalized_frame, cv2.COLORMAP_JET)  # JET를 사용하여 온도 차이를 표현
    print("JET color map applied to frame.")

    return colored_frame

def save_image(image, filename):
    # Save the colored frame as an image file on the Desktop
    try:
        desktop_path = os.path.expanduser("~/Desktop")
        full_path = os.path.join(desktop_path, filename)
        cv2.imwrite(full_path, image)
        print(f"Saving image to {full_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

def main():
    # Initialize the Camera Manager
    camera_manager = TmCamera.CameraManager()

    # List all connected local cameras
    local_cameras = camera_manager.get_local_cameras()

    if not local_cameras:
        print("No local cameras found.")
        return

    # Select the first available camera
    selected_cam_info = local_cameras[0]

    # Open the selected camera
    camera = TmCamera.TmCamera(selected_cam_info.obj)

    if camera.open():
        try:
            # Query a frame from the camera
            frame = camera.query_frame(width=640, height=480)
            if frame is not None:
                # Process the frame to apply color mapping
                colored_frame = process_frame(frame)

                # Save the processed image
                save_image(colored_frame, "thermal_colored_image.jpg")
            else:
                print("No frame captured from the camera.")
        finally:
            # Ensure the camera is closed properly
            camera.close()
            print("Camera closed.")
    else:
        print("Failed to open the camera.")

if __name__ == "__main__":
    main()
