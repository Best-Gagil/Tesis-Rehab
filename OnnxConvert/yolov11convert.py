from ultralytics import YOLO

# Configuración optimizada para Meta Quest 3
model = YOLO("yolo11n.pt")

model.export(
    format="onnx",
    imgsz=640,                   # Usar tamaño fijo recomendado para realidad mixta
    half=True,                    # FP16 para optimizar rendimiento en GPU móvil
    simplify=True,                # Simplificar grafo ONNX
    dynamic=False,                # Desactivar para mejor optimización en Quest 3
    opset=16,                    # Versión compatible con Sentis/Barracuda
    nms=True,                     # Incluir NMS en el modelo exportado
    batch=1,                      # Tamaño batch fijo para aplicaciones en tiempo real
    device='cpu'                  # Exportar en CPU para máxima compatibilidad
)

# Validación del modelo exportado
onnx_model = YOLO("yolo11n.onnx")
results = onnx_model(
    "https://ultralytics.com/images/bus.jpg",
    imgsz=640,
    conf=0.25,    # Umbral ajustado para realidad mixta
    iou=0.45      # Balance precisión/rendimiento
)