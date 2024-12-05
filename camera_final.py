import sys
import time
import socket
from PyQt5.QtCore import QThread, QBuffer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget
from TmCore import TmCamera, ColormapTypes, TempUnit

# AWS 서버 정보
AWS_SERVER_IP = "172.21.61.221"  # AWS 서버 IPi
AWS_SERVER_PORT = 5002           # AWS 서버 포트

class Camera:
    def __init__(self):
        self.tmCamera = TmCamera()
        self.preview_width = 640
        self.preview_height = 480
        self.worker = FrameWorker(self)

    def connect_camera(self):
        try:
            print("Scanning for cameras...")
            name, port, index, count = self.tmCamera.get_local_camera_list()
            if count == 0:
                raise Exception("No cameras found.")

            print(f"Cameras found: {name}, Ports: {port}")

            # 첫 번째 카메라에 연결
            ret = self.tmCamera.open_local_camera(name[0], port[0], index[0])
            if not ret:
                raise Exception("Failed to connect to the camera.")

            # 색상 맵과 온도 단위 설정
            self.tmCamera.set_color_map(ColormapTypes.GrayScale)
            self.tmCamera.set_temp_unit(TempUnit.CELSIUS)

            print(f"Camera connected: {name[0]} on port {port[0]}")
        except Exception as e:
            print(f"Error connecting camera: {e}")
            return False
        return True

    def disconnect_camera(self):
        try:
            self.tmCamera.close()
            print("Camera disconnected!")
        except Exception as e:
            print(f"Error disconnecting camera: {e}")


class FrameWorker(QThread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.isRun = False
        self.sock = None

    def stop(self):
        self.isRun = False

    def connect_to_server(self):
        """AWS 서버에 연결을 설정합니다."""
        for attemp in range(3):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((AWS_SERVER_IP, AWS_SERVER_PORT))
                print(f"Connected to AWS server at {AWS_SERVER_IP}:{AWS_SERVER_PORT}")
                return True
            except Exception as e:
                print(f"Connection attemp {attemp +1 } failed: {e}")
                time.sleep(5) # 5초 대기
        print("Failed to connect to server after 3 attemps.")
        self.isRun = False
        return False

    def run(self):
        self.isRun = True
        self.connect_to_server()

        while self.isRun:
            start_time = time.time()

            try:
                # 프레임 캡처
                frame = self.parent.tmCamera.query_frame(
                    self.parent.preview_width, self.parent.preview_height
                )
                if frame is None:
                    print("No frame captured.")
                    continue

                try:
                    bitmap = frame.to_bitmap(ColormapTypes.GrayScale)
                    image = QImage(bitmap, frame.width(), frame.height(), QImage.Format_RGB888)
                except Exception as e:
                    print(f"Error during frame conversion: {e}")
                    continue

                # 이미지를 JPEG로 변환
                buffer = QBuffer()
                buffer.open(QBuffer.ReadWrite)
                image.save(buffer, "JPEG")  # JPEG로 저장
                image_bytes = buffer.data()  # QByteArray로 변환
                buffer.close()

                # 데이터 길이를 먼저 전송 (8바이트)
                length = len(image_bytes)
                self.sock.sendall(length.to_bytes(8, byteorder='big'))  # 8바이트로 수정
                print(f"Sent frame size: {length} bytes")

                # 실제 이미지 데이터를 전송
                self.sock.sendall(image_bytes)
                print(f"Sent {length} bytes to AWS server.")
            except Exception as e:
                print(f"Error capturing or sending frame: {e}")
                self.isRun = False
                break

            #5 FPS 유지
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / 5.0) - elapsed_time)
            time.sleep(sleep_time)

        # 연결 종료
        if self.sock:
            self.sock.close()
            print("Disconnected from AWS server.")


class StreamApp(QWidget):
    def __init__(self):
        super().__init__()

        # 카메라 초기화
        self.camera = Camera()
        if self.camera.connect_camera():
            self.camera.worker.start()

    def closeEvent(self, event):
        # 종료 시 카메라와 스레드 정리
        self.camera.worker.stop()
        self.camera.worker.wait()
        self.camera.disconnect_camera()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    stream_app = StreamApp()
    stream_app.show()
    sys.exit(app.exec_())
