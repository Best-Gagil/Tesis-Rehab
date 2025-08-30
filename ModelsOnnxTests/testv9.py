import onnxruntime as ort
import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(box):
    x_c, y_c, w, h = box
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return [x1, y1, x2, y2]

def preprocess(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, input_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_expanded = np.expand_dims(img_transposed, axis=0)
    return img_expanded, img

def main():
    model_path = "bestv9.onnx"
    image_path = "ExoImage.jpg"
    class_names = ['Hand', 'Micro', 'Button']
    conf_threshold = 0.25  # Umbral de confianza más bajo para depuración
    nms_threshold = 0.3
    input_size = (640, 640)

    # Cargar modelo ONNX
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Preprocesar imagen
    input_tensor, original_img = preprocess(image_path, input_size)
    h, w, _ = original_img.shape

    # Inferencia
    outputs = session.run(None, {input_name: input_tensor})
    preds = outputs[0]  # (1, 7, 8400)
    preds = np.squeeze(preds)  # (7, 8400)
    preds = preds.T  # (8400, 7)

    boxes = preds[:, 0:4]  # center_x, center_y, w, h
    objectness = sigmoid(preds[:, 4])
    class_scores = sigmoid(preds[:, 5:])  # (8400, 3)

    class_ids = np.argmax(class_scores, axis=1)
    class_confidences = class_scores[np.arange(len(class_scores)), class_ids]

    confidences = objectness * class_confidences

    # Depuración: imprimir confidencias máximas y número de detecciones tras filtro
    print("Confidencias máximas (top 10):", np.sort(confidences)[-10:])
    mask = confidences > conf_threshold
    print(f"Número de detecciones tras filtro: {mask.sum()}")

    # Filtrar por confianza
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = []
    for box in boxes:
        x1, y1, x2, y2 = xywh2xyxy(box)
        # Escalar a tamaño original
        x1 = x1 * w / input_size[0]
        y1 = y1 * h / input_size[1]
        x2 = x2 * w / input_size[0]
        y2 = y2 * h / input_size[1]

        # Limitar cajas dentro de la imagen
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        boxes_xyxy.append([int(x1), int(y1), int(x2), int(y2)])

    boxes_xyxy = np.array(boxes_xyxy)

    # Aplicar NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_threshold,
        nms_threshold
    )

    if len(indices) == 0:
        print("No se detectaron objetos con suficiente confianza.")
        return

    for i in indices.flatten():
        x1, y1, x2, y2 = boxes_xyxy[i]
        conf = confidences[i]
        cls_id = class_ids[i]
        label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

        print(f"Detección: {label} conf={conf:.3f} bbox=({x1},{y1},{x2},{y2})")

        # Dibujar caja y etiqueta
        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_img, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detecciones YOLOv9 ONNX", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
