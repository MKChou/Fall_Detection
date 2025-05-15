問題1：模型路徑錯誤
錯誤訊息
RuntimeError: Load model human-pose-estimation.onnx failed. File doesn't exist

原因

ONNX檔案未放置在正確路徑或檔名拼寫錯誤

解決方案

確認檔案存在性：ls -l /path/to/model.onnx

使用絕對路徑載入模型：

bash
--onnx=/home/jetson/.../openpose_256.onnx
問題2：輸入層名稱不匹配
錯誤訊息
Cannot find input tensor with name "input" in the network inputs!

原因

模型輸入層名稱非預設的input，而是自定義名稱（如image）

解決方案

查詢實際輸入名稱：

python
import onnx
model = onnx.load("model.onnx")
print([input.name for input in model.graph.input])
修正轉換指令的--shapes參數：

bash
--shapes=image:1x3x256x256
問題3：靜態模型指定形狀參數
錯誤訊息
Static model does not take explicit shapes

原因

模型本身已固定輸入尺寸，但轉換時強制指定--shapes

解決方案
移除--shapes參數，直接轉換：

bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
問題4：模型複雜度過高
潛在錯誤模式

轉換過程卡住或報錯Unsupported layer type

解決方案
使用ONNX Simplifier簡化模型：

bash
pip install onnx-simplifier
python -m onnxsim input.onnx output_sim.onnx
進階問題：版本兼容性
常見錯誤

不支援的算子（如反卷積）

INT64權重類型錯誤

解決方案

升級TensorRT版本：JetPack 4.x預設為TRT 8.2，可升級至8.4+

強制轉換數據類型：

bash
trtexec --onnx=model.onnx --onnxFlags=FP32_ENABLE=1
效能優化建議
啟用FP16加速：--fp16

調整工作空間：根據設備記憶體設定（如--workspace=2048）

關閉GUI省記憶體：

bash
sudo service lightdm stop  # 轉換完成後用 start 恢復
部署驗證步驟
檢查引擎是否生成：

bash
ls -lh *.trt  # 確認檔案大小合理（通常比ONNX小）
簡易Python載入測試：

python
import tensorrt as trt
engine = load_engine("model.trt")
print(engine.get_binding_shape(0))  # 應輸出模型輸入尺寸
總結流程：
確認模型存在 → 檢查輸入層名稱 → 簡化複雜模型 → 移除多餘參數 → 轉換後驗證
此流程可避免多數常見轉換錯誤，建議保存此清單供未來部署參考。