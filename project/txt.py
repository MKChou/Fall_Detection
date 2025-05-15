import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
import threading

# === 非同步攝影機擷取類別 ===
class CameraStream:
    def __init__(self, src=0, width=320, height=240):
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

# === TensorRT 載入與推理相關 ===
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()  # 正確讀取二進位數據
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_data)  # 傳入二進位數據

# === 主程式 ===
if __name__ == "__main__":
    # 1. 載入 TensorRT 引擎
    engine_path = "openpose_256.trt"
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"找不到 TensorRT 引擎檔案：{engine_path}")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # 2. 分配記憶體
    input_shape = engine.get_binding_shape(0)   # (1, 3, 256, 256)
    output_heatmap_shape = engine.get_binding_shape(1)  # (1, 19, 32, 32)
    output_paf_shape = engine.get_binding_shape(2)      # (1, 38, 32, 32)
    input_size = int(np.prod(input_shape))
    output_heatmap_size = int(np.prod(output_heatmap_shape))
    output_paf_size = int(np.prod(output_paf_shape))
    d_input = cuda.mem_alloc(int(input_size * np.float32().itemsize))
    d_output_heatmap = cuda.mem_alloc(int(output_heatmap_size * np.float32().itemsize))
    d_output_paf = cuda.mem_alloc(int(output_paf_size * np.float32().itemsize))
    bindings = [int(d_input), int(d_output_heatmap), int(d_output_paf)]

    # 3. 啟動非同步攝影機
    cam = CameraStream(src=0, width=320, height=240)
    input_size_hw = 256  # 模型輸入尺寸

    # FPS 計算
    fps = 0
    frame_count = 0
    start_time = time.time()
    alpha = 0.9
    last_time = time.time()

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("No frame received.")
                break

            # FPS 計算與移動平均
            now = time.time()
            dt = now - last_time
            last_time = now
            instant_fps = 1.0 / dt if dt > 0 else 0
            fps = alpha * fps + (1 - alpha) * instant_fps if fps > 0 else instant_fps

            h, w = frame.shape[:2]
            scale = input_size_hw / max(h, w)
            resized_h = int(h * scale)
            resized_w = int(w * scale)
            resized = cv2.resize(frame, (resized_w, resized_h))

            # 計算填充量
            pad_top = (input_size_hw - resized_h) // 2
            pad_bottom = input_size_hw - resized_h - pad_top
            pad_left = (input_size_hw - resized_w) // 2
            pad_right = input_size_hw - resized_w - pad_left

            padded = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0,0,0)
            )

            # 歸一化到 [-1,1]
            input_data = padded.astype(np.float32)
            input_data = (input_data - 127.5) / 127.5
            input_data = input_data.transpose(2,0,1)[np.newaxis, ...]  # (1,3,256,256)
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)

            # 輸入資料至 GPU
            cuda.memcpy_htod(d_input, input_data)

            # 執行推論
            context.execute_v2(bindings)

            # 從 GPU 拷貝結果回主機
            output_heatmap = np.empty(output_heatmap_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output_heatmap, d_output_heatmap)

            # 關鍵點後處理
            num_keypoints = output_heatmap.shape[1]
            heatmaps = output_heatmap[0]  # (19, 32, 32)
            points = []
            for i in range(num_keypoints):
                hm = cv2.resize(heatmaps[i], (input_size_hw, input_size_hw))
                _, conf, _, point = cv2.minMaxLoc(hm)
                x = (point[0] - pad_left) / scale
                y = (point[1] - pad_top) / scale
                x = max(0, min(x, w-1))
                y = max(0, min(y, h-1))
                points.append((int(x), int(y)) if conf > 0.3 else None)

            frame_count += 1

            # 每秒輸出一次
            if time.time() - start_time >= 1.0:
                os.system('cls' if os.name == 'nt' else 'clear')
                print(f"=== OpenPose TensorRT 終端輸出 ===")
                print(f"FPS: {fps:.1f}")
                print(f"關鍵點座標 (未偵測到為 None):")
                for idx, pt in enumerate(points):
                    print(f"  {idx:2d}: {pt}")
                print("\n按 Ctrl+C 結束")
                frame_count = 0
                start_time = time.time()

    except KeyboardInterrupt:
        print("\n程式已中斷，準備釋放資源...")

    finally:
        cam.stop()
        d_input.free()
        d_output_heatmap.free()
        d_output_paf.free()
        print("程式執行完畢")
