import cv2
import threading
import time

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

if __name__ == "__main__":
    cam = CameraStream(0, 640, 480)

    frame_count = 0
    fps = 0
    start_time = time.time()
    fps_display_interval = 1  # 每隔幾秒更新一次FPS顯示

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No frame received.")
            break

        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > fps_display_interval:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        # 在畫面左上角顯示 FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Fast Camera Input', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
