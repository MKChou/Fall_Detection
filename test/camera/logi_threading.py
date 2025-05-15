import cv2
import threading

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        # 設定解析度
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
            # 回傳複製的 frame，避免多執行緒衝突
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

if __name__ == "__main__":
    cam = CameraStream(0, 640, 480)  # 0 代表預設攝影機

    while True:
        ret, frame = cam.read()
        if ret:
            cv2.imshow('Fast Camera Input', frame)
        else:
            print("No frame received.")
            break

        # 按下 q 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()
