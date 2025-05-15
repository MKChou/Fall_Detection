import cv2
import onnxruntime as ort
import numpy as np

# === 1. 初始化 ONNX 模型 ===
model_path = "openpose_256.onnx"  # 請確認檔名與路徑
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# === 2. 關節連線定義 (COCO格式) ===
POSE_PAIRS = [
    [1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
    [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
    [1,0], [0,14], [14,16], [0,15], [15,17]
]

# === 3. 開啟 USB 攝影機 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

input_size = 256  # 請依模型調整

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    scale = input_size / max(h, w)
    resized_h = int(h * scale)
    resized_w = int(w * scale)
    resized = cv2.resize(frame, (resized_w, resized_h))

    # 計算不對稱填充量
    pad_top = (input_size - resized_h) // 2
    pad_bottom = input_size - resized_h - pad_top
    pad_left = (input_size - resized_w) // 2
    pad_right = input_size - resized_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0,0,0)
    )

    # 歸一化到 [-1,1]
    input_data = padded.astype(np.float32)
    input_data = (input_data - 127.5) / 127.5
    input_data = input_data.transpose(2,0,1)[np.newaxis, ...]  # (1,3,256,256)

    # === 4. 執行推理 ===
    outputs = session.run(None, {input_name: input_data})
    heatmaps = outputs[0][0]  # (num_keypoints, 64, 64) or (num_keypoints, 32, 32) 視模型而定

    # === 5. 關鍵點後處理 ===
    num_keypoints = heatmaps.shape[0]
    heatmap_size = heatmaps.shape[1]
    points = []
    for i in range(num_keypoints):
        hm = cv2.resize(heatmaps[i], (input_size, input_size))
        _, conf, _, point = cv2.minMaxLoc(hm)
        x = (point[0] - pad_left) / scale
        y = (point[1] - pad_top) / scale
        # 邊界保護
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        points.append((int(x), int(y)) if conf > 0.3 else None)

    # === 6. 繪製骨架 ===
    for pair in POSE_PAIRS:
        a, b = pair
        if points[a] and points[b]:
            cv2.line(frame, points[a], points[b], (0,255,0), 2)
            cv2.circle(frame, points[a], 3, (0,0,255), -1)

    cv2.imshow('USB Camera Pose', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
