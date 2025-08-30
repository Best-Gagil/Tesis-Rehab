# ==== LIBRERÍAS ====
import cv2  # OpenCV para procesamiento de imágenes
import numpy as np  # NumPy para operaciones matriciales
import onnxruntime as ort  # ONNX Runtime para inferencia del modelo
import argparse  # Para manejo de argumentos de línea de comandos

# ==== CONSTANTES GLOBALES ====
CLASS_NAMES = ["Hand", "Micro", "Button"]  # Nombres de clases personalizadas
INPUT_SIZE = (640, 640)  # Tamaño de entrada requerido por el modelo (ancho, alto)
OUTPUT_SIZE = (1280, 720)  # Tamaño de visualización de la ventana
CONF_THRESH = 0.45  # Umbral de confianza para filtrar detecciones
IOU_THRESH = 0.45  # Umbral de Intersection Over Union para NMS
PADDING_COLOR = 114  # Color gris para el padding (RGB: 114,114,114)

def preprocess(image_path):
    """
    Preprocesa la imagen para adaptarla a la entrada del modelo.
    
    Args:
        image_path (str): Ruta de la imagen a procesar
        
    Returns:
        tuple: (imagen original, tensor de entrada, metadatos de transformación)
        
    Proceso detallado:
        1. Carga la imagen con OpenCV
        2. Calcula el escalado manteniendo relación de aspecto
        3. Añade padding centrado
        4. Normaliza y prepara el tensor
    """
    # Carga la imagen en formato BGR
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error cargando imagen: {image_path}")
    
    # Dimensiones originales
    h, w = img.shape[:2]
    
    # Calcula escala manteniendo relación de aspecto
    scale = min(INPUT_SIZE[1]/h, INPUT_SIZE[0]/w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Cálculo preciso de padding (centrado)
    pad_x = (INPUT_SIZE[0] - new_w) / 2
    pad_y = (INPUT_SIZE[1] - new_h) / 2
    
    # Redimensionamiento con interpolación bilineal
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Creación de imagen con padding
    img_padded = np.full((INPUT_SIZE[1], INPUT_SIZE[0], 3), PADDING_COLOR, dtype=np.uint8)
    
    # Aplicación del padding con redondeo preciso
    top = int(np.round(pad_y - 0.1))  # Compensación por posibles errores de redondeo
    bottom = int(np.round(pad_y + 0.1))
    left = int(np.round(pad_x - 0.1))
    right = int(np.round(pad_x + 0.1))
    
    # Inserta la imagen redimensionada en el centro
    img_padded[top:new_h+top, left:new_w+left] = img_resized
    
    # Metadatos para postprocesamiento
    meta = {
        'original_shape': (h, w),
        'resized_shape': (new_h, new_w),
        'scale_factor': scale,
        'pad': (left, right, top, bottom),
        'effective_pad': (pad_x, pad_y)
    }
    
    # Preparación del tensor (formato NCHW)
    input_tensor = img_padded.transpose(2, 0, 1).astype(np.float32)
    input_tensor /= 255.0  # Normalización [0-255] -> [0-1]
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Añade dimensión batch
    
    return img, input_tensor, meta

def postprocess(outputs, meta):
    """
    Postprocesa las salidas del modelo para obtener detecciones finales.
    
    Args:
        outputs (list): Salidas crudas del modelo ONNX
        meta (dict): Metadatos del preprocesamiento
        
    Returns:
        tuple: (cajas, scores, IDs de clase)
        
    Proceso detallado:
        1. Filtrado por confianza
        2. Conversión de coordenadas relativas a absolutas
        3. Aplicación de Non-Max Suppression (NMS)
    """
    predictions = np.squeeze(outputs[0])  # Elimina dimensiones unitarias
    
    # Recupera información de escalado
    orig_h, orig_w = meta['original_shape']
    new_h, new_w = meta['resized_shape']
    pad_x, pad_y = meta['effective_pad']
    
    # Manejo de caso sin detecciones
    if predictions.size == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    # Filtrado inicial por confianza
    scores = predictions[:, 4]
    mask = scores > CONF_THRESH
    filtered_preds = predictions[mask]
    
    if filtered_preds.shape[0] == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
    
    # Conversión de coordenadas normalizadas a espacio redimensionado
    boxes = filtered_preds[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / (INPUT_SIZE[0] - 2*pad_x) * new_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / (INPUT_SIZE[1] - 2*pad_y) * new_h
    
    # Clip para evitar valores fuera de rango
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, new_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, new_h)
    
    # Escalado a dimensiones originales
    boxes[:, [0, 2]] /= meta['scale_factor']
    boxes[:, [1, 3]] /= meta['scale_factor']
    
    # Clip final en espacio original
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    # Non-Max Suppression (Supresión de no máximos)
    boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
    areas = boxes_wh[:, 0] * boxes_wh[:, 1]
    
    # Usa NMS de OpenCV con parámetros ajustados
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        scores[mask].tolist(), 
        CONF_THRESH, 
        IOU_THRESH, 
        top_k=50,  # Máximo de detecciones a conservar
        eta=1.0    # Parámetro de adaptación de umbral
    )
    
    # Extrae clases y resultados filtrados
    class_ids = filtered_preds[:, 5].astype(int)
    return boxes[indices], scores[mask][indices], class_ids[indices]

def draw_detections(image, boxes, scores, class_ids):
    """
    Dibuja las detecciones en la imagen original.
    
    Args:
        image (np.array): Imagen original
        boxes (np.array): Cajas delimitadoras [x1, y1, x2, y2]
        scores (np.array): Puntuaciones de confianza
        class_ids (np.array): IDs de clase
        
    Returns:
        np.array: Imagen con anotaciones dibujadas
    """
    h, w = image.shape[:2]
    scale_factor = (h * w) ** 0.5 / 1000  # Factor de escala adaptativo
    
    for box, score, cls_id in zip(boxes, scores, class_ids):
        # Cálculo preciso de coordenadas enteras
        x1 = max(0, int(round(box[0] - 0.5)))  # Compensación por redondeo
        y1 = max(0, int(round(box[1] - 0.5)))
        x2 = min(w, int(round(box[2] + 0.5)))
        y2 = min(h, int(round(box[3] + 0.5)))
        
        color = (0, 255, 0)  # Color BGR para rectángulos
        thickness = max(1, int(3 * scale_factor))  # Grosor adaptativo
        font_scale = 0.8 * scale_factor  # Tamaño de fuente adaptativo
        
        # Dibuja rectángulo de detección
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Prepara texto de etiqueta
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calcula posición del fondo de texto
        y1_label = max(0, y1 - text_height - 5)
        cv2.rectangle(image, 
                     (x1, y1_label),
                     (x1 + text_width, y1),
                     color, -1)  # -1 = relleno completo
        
        # Posicionamiento inteligente del texto
        text_y = y1 - 5 if y1_label > 0 else y1 + text_height + 5
        cv2.putText(image, label, (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                   (0, 0, 0), thickness, cv2.LINE_AA)  # AA = Anti-aliasing
    
    return image

def main(image_path, model_path, output_path=None):
    """
    Función principal que ejecuta el pipeline completo de detección.
    
    Args:
        image_path (str): Ruta de la imagen de entrada
        model_path (str): Ruta del modelo ONNX
        output_path (str, optional): Ruta para guardar resultados
    """
    # Paso 1: Preprocesamiento
    orig_img, input_tensor, meta = preprocess(image_path)
    
    # Paso 2: Configuración de ONNX Runtime
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']  # Cambiar a 'CUDAExecutionProvider' para GPU
    )
    
    # Paso 3: Inferencia del modelo
    outputs = session.run(
        output_names=None,  # Usa todas las salidas del modelo
        input_feed={session.get_inputs()[0].name: input_tensor}
    )
    
    # Paso 4: Postprocesamiento
    boxes, scores, class_ids = postprocess(outputs, meta)
    
    # Paso 5: Visualización de resultados
    result_img = orig_img.copy()
    if boxes.size > 0:
        result_img = draw_detections(result_img, boxes, scores, class_ids)
    else:
        # Mensaje de advertencia si no hay detecciones
        cv2.putText(result_img, "Sin detecciones", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Redimensionamiento para visualización
    h, w = result_img.shape[:2]
    scale = min(OUTPUT_SIZE[0]/w, OUTPUT_SIZE[1]/h)
    display_img = cv2.resize(result_img, (int(w*scale), int(h*scale)), 
                           interpolation=cv2.INTER_LANCZOS4)  # Interpolación de alta calidad
    
    # Guardado de resultados
    if output_path:
        cv2.imwrite(output_path, result_img)
        print(f"Imagen guardada en: {output_path}")
    
    # Visualización interactiva
    cv2.namedWindow("Detecciones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detecciones", OUTPUT_SIZE[0], OUTPUT_SIZE[1])
    cv2.imshow("Detecciones", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='YOLOv11 Object Detection Inference Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=str, required=True,
                       help="Ruta a la imagen de entrada")
    parser.add_argument("--model", type=str, default="bestv11.onnx",
                       help="Ruta al modelo ONNX entrenado")
    parser.add_argument("--output", type=str, 
                       help="Ruta opcional para guardar la imagen resultante")
    
    args = parser.parse_args()
    
    # Ejecución principal
    main(args.image, args.model, args.output)
