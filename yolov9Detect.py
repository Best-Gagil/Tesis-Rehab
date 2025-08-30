import sys
import os
import argparse
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, 
                          scale_boxes, 
                          letterbox,
                          colors)

class YOLOv9_Detector:
    def __init__(self, weights_path='bestv9.pt'):
        # Configuración mejorada del dispositivo
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True  # Optimización CUDA
        
        # Parámetros ajustables
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.img_size = 640
        self.auto = True  # Tamaño automático basado en modelo
        
        # Carga del modelo con verificación
        try:
            self.model = DetectMultiBackend(
                weights_path, 
                device=self.device,
                data=Path(weights_path).with_suffix('.yaml'),  # Metadata del modelo
                fp16=True  # Usar mixed-precision
            )
            self.stride = self.model.stride
            self.class_names = self.model.names
            print(f"✅ Modelo cargado: {Path(weights_path).name}")
            print("Clases detectables:", self.class_names)
        except Exception as e:
            raise RuntimeError(f"❌ Error cargando modelo: {e}") from e

    def detect(self, img_path, output_dir='output'):
        # Procesamiento de entrada mejorado
        img_original = cv2.imread(img_path)
        if img_original is None:
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")
            
        # Preprocesamiento con letterbox (mantiene relación de aspecto)
        img_resized, ratio, pad = letterbox(
            img_original, 
            new_shape=(self.img_size, self.img_size), 
            auto=self.auto, 
            stride=self.stride
        )
        
        # Conversión a tensor optimizada
        tensor = torch.from_numpy(img_resized).to(self.device)
        tensor = tensor.permute(2, 0, 1).float()  # HWC to CHW
        tensor = tensor.unsqueeze(0) / 255  # 0-1 normalization
        
        # Inferencia con temporización
        with torch.no_grad():
            pred = self.model(tensor, augment=False, visualize=False)
            pred = non_max_suppression(
                pred, 
                self.conf_thres, 
                self.iou_thres, 
                max_det=1000
            )[0]
        
        # Post-procesamiento escalado
        if len(pred):
            pred[:, :4] = scale_boxes(
                tensor.shape[2:], 
                pred[:, :4], 
                img_original.shape
            ).round()
        
        # Guardar resultados
        Path(output_dir).mkdir(exist_ok=True)
        output_path = str(Path(output_dir) / Path(img_path).name)
        self._plot_results(img_original, pred, output_path)
        
        return output_path

    def _plot_results(self, img, detections, output_path):
        # Usar paleta de colores oficial de YOLOv5
        for *xyxy, conf, cls in reversed(detections):
            class_name = self.class_names[int(cls)]
            color = colors(int(cls))
            
            # Etiqueta mejorada
            label = f"{class_name} {conf:.2f}"
            
            # Dibujo optimizado
            cv2.rectangle(img, 
                         (int(xyxy[0]), int(xyxy[1])), 
                         (int(xyxy[2]), int(xyxy[3])), 
                         color, 2, lineType=cv2.LINE_AA)
            
            # Fondo de texto con transparencia
            (tw, th), _ = cv2.getTextSize(
                label, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 2
            )
            cv2.rectangle(img,
                         (int(xyxy[0]), int(xyxy[1]) - th - 5),
                         (int(xyxy[0]) + tw, int(xyxy[1]) - 5),
                         color, -1)
            
            cv2.putText(img, label,
                       (int(xyxy[0]), int(xyxy[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imwrite(output_path, img)
        print(f"🔍 Resultados guardados en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='Ruta de imagen/video')
    parser.add_argument('--conf', type=float, default=0.5, 
                       help='Umbral de confianza')
    parser.add_argument('--output', type=str, default='output',
                       help='Directorio de salida')
    args = parser.parse_args()
    
    try:
        detector = YOLOv9_Detector()
        detector.conf_thres = args.conf
        result_path = detector.detect(args.img_path, args.output)
        print(f"✅ Proceso completado: {result_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)
