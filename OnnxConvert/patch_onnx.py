import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

IN_PATH  = "bestv11.onnx"              # <-- TU ONNX de entrada (1 salida, típicamente [1,N,6])
OUT_PATH = "yolov11_meta_contract.onnx"    # <-- ONNX parchado (2 salidas: coords y labels)

m = onnx.load(IN_PATH)
g = m.graph

# Info útil (no obligatorio)
opset = m.opset_import[0].version if len(m.opset_import) else 0
print("Detected ONNX opset:", opset)

# Verifica que solo hay una salida
assert len(g.output) == 1, "Se esperaba 1 salida (N×6 o 1×N×6). Exporta ONNX con nms=True."

dets_name = g.output[0].name

# === Inicializadores ===
# Forma objetivo [-1, 6] para aplanar la salida a [N,6]
shapeN6 = numpy_helper.from_array(np.array([-1, 6], dtype=np.int64), name='shapeN6')
# Tamaños para Split: 6 columnas -> [1,1,1,1,1,1]
split_sizes = numpy_helper.from_array(np.array([1,1,1,1,1,1], dtype=np.int64), name='split_sizes')
# Constante 0.5 para calcular centros
half = numpy_helper.from_array(np.array([0.5], dtype=np.float32), name='half')

# Añade inicializadores al grafo
g.initializer.extend([shapeN6, split_sizes, half])

# === 0) Reshape a [N,6] ===
reshape = helper.make_node(
    'Reshape',
    inputs=[dets_name, 'shapeN6'],
    outputs=['dets2d'],
    name='reshape_to_n6'
)

# === 1) Split N×6 a 6 tensores N×1 (eje=1) ===
# En opset >= 13, 'split' debe ir como segundo input (no como atributo).
split = helper.make_node(
    'Split',
    inputs=['dets2d', 'split_sizes'],
    outputs=['x1','y1','x2','y2','score','cls_f'],
    name='split_dets',
    axis=1
)

# === 2) cx, cy, w, h ===
add_x  = helper.make_node('Add', inputs=['x1','x2'], outputs=['xsum'], name='add_x')
mul_cx = helper.make_node('Mul', inputs=['xsum','half'], outputs=['cx'], name='mul_cx')

add_y  = helper.make_node('Add', inputs=['y1','y2'], outputs=['ysum'], name='add_y')
mul_cy = helper.make_node('Mul', inputs=['ysum','half'], outputs=['cy'], name='mul_cy')

sub_w = helper.make_node('Sub', inputs=['x2','x1'], outputs=['w'], name='sub_w')
sub_h = helper.make_node('Sub', inputs=['y2','y1'], outputs=['h'], name='sub_h')

# === 3) coords = [cx, cy, w, h] (N×4) ===
concat = helper.make_node(
    'Concat',
    inputs=['cx','cy','w','h'],
    outputs=['coords'],
    name='concat_coords',
    axis=1
)

# === 4) labels = int32(cls_f) (N×1) ===
cast_labels = helper.make_node(
    'Cast',
    inputs=['cls_f'],
    outputs=['labels'],
    name='cast_labels',
    to=TensorProto.INT32
)

# Inserta nodos al grafo
g.node.extend([reshape, split, add_x, mul_cx, add_y, mul_cy, sub_w, sub_h, concat, cast_labels])

# Reemplaza la única salida por las dos nuevas
g.output.clear()
coords_vi = helper.make_tensor_value_info('coords', TensorProto.FLOAT,  None)   # N×4 (shape puede inferirse)
labels_vi = helper.make_tensor_value_info('labels', TensorProto.INT32, None)    # N×1
g.output.extend([coords_vi, labels_vi])

# (Opcional) Inferencia de shapes
try:
    m = onnx.shape_inference.infer_shapes(m)
except Exception as e:
    print("Shape inference warning:", e)

# Validación y guardado
onnx.checker.check_model(m)
onnx.save(m, OUT_PATH)
print("OK → ONNX parchado guardado en:", OUT_PATH)
