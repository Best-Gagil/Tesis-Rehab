import onnx, onnxruntime as ort
m = onnx.load("yolov11_meta_contract.onnx")
print("Outputs:", [o.name for o in m.graph.output])  # Debe mostrar ['coords','labels']
sess = ort.InferenceSession("yolov11_meta_contract.onnx", providers=["CPUExecutionProvider"])
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)
