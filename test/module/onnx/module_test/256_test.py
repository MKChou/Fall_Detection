import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("openpose_256.onnx")
input_name = session.get_inputs()[0].name
# 假設模型輸入為 (1, 3, 256, 256)
dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = session.run(None, {input_name: dummy_input})
print(output)
