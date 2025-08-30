import cv2
import torch

class YOLOv5_Detector:
    def __init__(self, weights_path='best.pt'):
        # Cargar modelo con verificación de clases
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=weights_path, 
                                  force_reload=True,
                                  skip_validation=True)
        
        # Mapeo de clases corregido
        self.classes = self.model.names  # Diccionario {id: nombre}
        self.class_to_idx = {v: k for k, v in self.classes.items()}  # Mapeo inverso
        self.palette = [(0,255,0), (255,0,0), (0,0,255)]  # Verde, Rojo, Azul
        
        print("Clases detectadas:", self.classes)  # Verificación requerida

    def detect(self, img_path, conf_thresh=0.5):
        # Cargar y procesar imagen
        img = cv2.imread(img_path)
        results = self.model(img)
        
        # Filtrar detecciones por confianza
        df = results.pandas().xyxy[0]
        df = df[df['confidence'] >= conf_thresh]
        
        # Dibujar bounding boxes
        for _, row in df.iterrows():
            x1, y1, x2, y2 = map(int, row[['xmin','ymin','xmax','ymax']])
            label = f"{row['name']} {row['confidence']:.2f}"
            
            # Obtener índice de clase corregido
            class_idx = self.class_to_idx[row['name']]
            color = self.palette[class_idx % len(self.palette)]  # Evitar índice fuera de rango
            
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Guardar y retornar resultados
        cv2.imwrite('output.jpg', img)
        print(f"Resultados guardados en: output.jpg")
        return img

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str)
    parser.add_argument('--conf-thresh', type=float, default=0.5)
    args = parser.parse_args()
    
    detector = YOLOv5_Detector()
    detector.detect(args.img_path, args.conf_thresh)