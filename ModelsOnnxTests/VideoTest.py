import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
from collections import deque
import sys

# ================== CLASES ==================
DEFAULT_CLASS_NAMES = ["Hand", "Micro", "Button"]

# ================== PREPROCESO (LETTERBOX) ==================
def letterbox(frame, size=640, pad_color=114):
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    canvas[top:top+nh, left:left+nw] = resized
    tensor = canvas.transpose(2,0,1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, 0)
    meta = {
        "orig_shape": (h, w),
        "scale": scale,
        "pad": (left, top),
        "resized_shape": (nh, nw),
        "input_size": size
    }
    return tensor, meta

def undo_letterbox(boxes, meta):
    if boxes.size == 0:
        return boxes
    left, top = meta["pad"]
    scale = meta["scale"]
    h0, w0 = meta["orig_shape"]
    # boxes en coordenadas letterbox → restar pad → dividir scale
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    return boxes

# ================== DECODIFICACIÓN BASE (OUTPUT (N,6)) ==================
def decode_raw_format6(raw, meta, class_thresholds, num_classes, size_filter_cfg):
    # raw: (N,6): x1,y1,x2,y2,score,class_id EN ESPACIO LETTERBOX
    if raw.ndim != 2 or raw.shape[1] != 6:
        return np.empty((0,4)), np.empty(0), np.empty(0)

    boxes = raw[:, :4].astype(np.float32).copy()
    scores = raw[:, 4].astype(np.float32)
    class_ids = raw[:, 5].astype(int)

    valid = (class_ids >= 0) & (class_ids < num_classes) & (scores > 0)
    if not np.any(valid):
        return np.empty((0,4)), np.empty(0), np.empty(0)
    boxes = boxes[valid]; scores = scores[valid]; class_ids = class_ids[valid]

    # Aplicar thresholds por clase
    keep = scores >= class_thresholds[class_ids]
    if not np.any(keep):
        return np.empty((0,4)), np.empty(0), np.empty(0)
    boxes = boxes[keep]; scores = scores[keep]; class_ids = class_ids[keep]

    # Undo letterbox
    boxes = undo_letterbox(boxes, meta)

    # Filtro dinámico de tamaño mínimo para clase Button
    btn_id = size_filter_cfg["button_class_id"]
    min_side_factor = size_filter_cfg["dynamic_factor"]
    min_px_floor = size_filter_cfg["min_px_floor"]
    if boxes.size:
        h0, w0 = meta["orig_shape"]
        min_side_dyn = max(min_px_floor, min_side_factor * min(h0, w0))
        sel = []
        for i,(b,c) in enumerate(zip(boxes,class_ids)):
            if c == btn_id:
                bw = b[2]-b[0]; bh = b[3]-b[1]
                if bw < min_side_dyn and bh < min_side_dyn:
                    continue
            sel.append(i)
        if not sel:
            return np.empty((0,4)), np.empty(0), np.empty(0)
        boxes = boxes[sel]; scores = scores[sel]; class_ids = class_ids[sel]

    return boxes, scores, class_ids

# ================== NMS POR CLASE + COEXISTENCIA BUTTON ==================
def per_class_nms(boxes, scores, class_ids, thresholds, iou_thr, coexist_button_id):
    if boxes.size == 0:
        return boxes, scores, class_ids
    final_b = []
    final_s = []
    final_c = []
    for cls in np.unique(class_ids):
            mask = class_ids == cls
            cb = boxes[mask]; cs = scores[mask]
            if cb.shape[0] == 0:
                continue
            # Formato para NMSBoxes: [x,y,w,h]
            b_xywh = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in cb]
            idxs = cv2.dnn.NMSBoxes(b_xywh, cs.tolist(), thresholds[cls], iou_thr)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    final_b.append(cb[i])
                    final_s.append(cs[i])
                    final_c.append(cls)

    if not final_b:
        return np.empty((0,4)), np.empty(0), np.empty(0)

    final_b = np.array(final_b); final_s = np.array(final_s); final_c = np.array(final_c)

    # Resolver solapes entre distintas clases excepto cuando una es Button
    keep_idx = []
    for i in range(len(final_b)):
        bi = final_b[i]
        ci = final_c[i]
        discard = False
        for j in keep_idx:
            cj = final_c[j]
            if ci == coexist_button_id or cj == coexist_button_id:
                continue
            # IoU
            iou = iou_single(bi, final_b[j])
            if iou > 0.45 and ci != cj:
                discard = True
                break
        if not discard:
            keep_idx.append(i)

    return final_b[keep_idx], final_s[keep_idx], final_c[keep_idx]

def iou_single(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

# ================== ENHANCEMENT GLOBAL OPCIONAL ==================
def enhance_frame(frame, use_clahe=False):
    if not use_clahe:
        return frame
    # Convertir a YCrCb y aplicar CLAHE en canal Y
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y2 = clahe.apply(y)
    ycc2 = cv2.merge([y2, cr, cb])
    enh = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2BGR)
    return enh

# ================== SHARPEN ROI ==================
def sharpen(img):
    g = cv2.GaussianBlur(img,(0,0), sigmaX=1.0)
    return cv2.addWeighted(img, 1.6, g, -0.6, 0)

# ================== CASCADA SOBRE MICRO ==================
def cascade_button(session, frame, micro_boxes, cfg, class_thresholds, num_classes, size_filter_cfg):
    if not cfg["enable"] or micro_boxes.size == 0:
        return np.empty((0,4)), np.empty(0), np.empty(0)
    out_b=[]; out_s=[]; out_c=[]
    h0,w0 = frame.shape[:2]
    for mb in micro_boxes:
        x1,y1,x2,y2 = mb
          # Expand
        bw = x2-x1; bh = y2-y1
        if bw < 4 or bh < 4:
            continue
        ex = bw * cfg["expand"]; ey = bh * cfg["expand"]
        rx1 = int(max(0, x1-ex)); ry1 = int(max(0, y1-ey))
        rx2 = int(min(w0, x2+ex)); ry2 = int(min(h0, y2+ey))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue
        if cfg["sharpen"]:
            roi = sharpen(roi)

        # Upscale interno
        if cfg["upscale"] > 0:
            side_max = max(roi.shape[:2])
            scale_factor = cfg["upscale"] / side_max
            if scale_factor > 1.0:
                new_w = int(roi.shape[1] * scale_factor)
                new_h = int(roi.shape[0] * scale_factor)
                roi = cv2.resize(roi, (new_w,new_h), interpolation=cv2.INTER_CUBIC)

        tensor, meta = letterbox(roi, size=cfg["input_size"], pad_color=cfg["pad_color"])
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if raw.ndim == 3:
            raw = np.squeeze(raw,0)

        # Ajustar threshold Button temporalmente
        temp_thr = class_thresholds.copy()
        temp_thr[cfg["button_class_id"]] = min(temp_thr[cfg["button_class_id"]], cfg["button_cascade_thr"])

        b2, s2, c2 = decode_raw_format6(raw, meta, temp_thr, num_classes, size_filter_cfg)
        if b2.size == 0:
            continue
        # Filtrar sólo Button
        mask_btn = c2 == cfg["button_class_id"]
        if not np.any(mask_btn):
            continue
        b2 = b2[mask_btn]; s2 = s2[mask_btn]; cbtn = c2[mask_btn]
        # Ajustar coords
        b2[:,[0,2]] += rx1
        b2[:,[1,3]] += ry1
        out_b.append(b2); out_s.append(s2); out_c.append(cbtn)

    if not out_b:
        return np.empty((0,4)), np.empty(0), np.empty(0)
    return np.vstack(out_b), np.concatenate(out_s), np.concatenate(out_c)

# ================== ACUMULACIÓN TEMPORAL ==================
class TemporalButtonAccumulator:
    def __init__(self, window=10, min_consec=3, dist_px=18, confirm_score=0.24, button_class_id=2, maxbuf=400):
        self.window = window
        self.min_consec = min_consec
        self.dist_px = dist_px
        self.confirm_score = confirm_score
        self.button_class_id = button_class_id
        self.buf = deque(maxlen=maxbuf)

    def update(self, frame_idx, boxes, scores, class_ids):
        for b,s,c in zip(boxes, scores, class_ids):
            if c == self.button_class_id:
                self.buf.append((frame_idx, b.copy(), float(s)))

    def propose(self, frame_idx):
        recent = [r for r in self.buf if frame_idx - r[0] <= self.window]
        if len(recent) < self.min_consec:
            return None
        # cluster simple alrededor de la última
        recent.sort(key=lambda x: x[0])
        ref = recent[-1][1]
        rcx = (ref[0]+ref[2])/2; rcy = (ref[1]+ref[3])/2
        cluster = []
        for fr,b,s in [(r[0], r[1], r[2]) for r in recent]:
            cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            if abs(cx-rcx) < self.dist_px and abs(cy-rcy) < self.dist_px:
                cluster.append((fr,b,s))
        if len(cluster) < self.min_consec:
            return None
        scores_cluster = [x[2] for x in cluster]
        mean_score = np.mean(scores_cluster)
        if mean_score < self.confirm_score:
            return None
        boxes_cluster = np.array([x[1] for x in cluster])
        box_mean = boxes_cluster.mean(axis=0)
        return box_mean, mean_score

# ================== DIBUJO ==================
def draw_annotations(frame, boxes, scores, class_ids, class_names, button_class_id=2):
    colors = {
        0:(0,255,0),
        1:(255,140,0),
        2:(0,0,255)
    }
    # Dibujar primero no-button
    order = np.argsort(class_ids)
    for i in order:
        cid = class_ids[i]
        if cid == button_class_id: continue
        x1,y1,x2,y2 = boxes[i].astype(int)
        cv2.rectangle(frame,(x1,y1),(x2,y2),colors.get(cid,(255,255,255)),2)
        cv2.putText(frame, f"{class_names[cid]} {scores[i]:.2f}",
                    (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors.get(cid,(255,255,255)),2)
    # Button resaltado
    for i in order:
        cid = class_ids[i]
        if cid != button_class_id: continue
        x1,y1,x2,y2 = boxes[i].astype(int)
        cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(255,255,255),3)
        cv2.rectangle(frame,(x1,y1),(x2,y2),colors.get(cid,(255,255,255)),2)
        cv2.putText(frame, f"Button {scores[i]:.2f}",
                    (x1+2, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255),2)
    return frame

# ================== MULTI-ESCALA (SEGUNDO PASO) ==================
def multi_scale_pass(session, frame, scale_factor, base_input_size, class_thresholds, num_classes, size_filter_cfg):
    # Reescalar frame globalmente
    h,w = frame.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    up = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    tensor, meta = letterbox(up, size=base_input_size)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if raw.ndim == 3: raw = np.squeeze(raw,0)
    b,s,c = decode_raw_format6(raw, meta, class_thresholds, num_classes, size_filter_cfg)
    if b.size == 0:
        return b,s,c
    # Ajustar coords al espacio original (dividir por scale_factor)
    b[:,[0,2]] /= scale_factor
    b[:,[1,3]] /= scale_factor
    return b,s,c

# ================== LOOP PRINCIPAL VIDEO ==================
def run_video(cfg):
    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        print("No se pudo abrir el video.", file=sys.stderr)
        return

    session = ort.InferenceSession(cfg.model_path, providers=['CPUExecutionProvider'])

    class_names = cfg.class_names
    num_classes = len(class_names)
    button_class_id = cfg.button_class_id

    # Thresholds por clase
    thresholds = np.array(cfg.base_thresholds, dtype=np.float32)
    min_thresholds = np.array(cfg.min_thresholds, dtype=np.float32)

    temporal = TemporalButtonAccumulator(
        window=cfg.temp_window,
        min_consec=cfg.temp_min_consec,
        dist_px=cfg.temp_max_dist,
        confirm_score=cfg.temp_confirm_score,
        button_class_id=button_class_id,
        maxbuf=400
    ) if cfg.enable_temporal else None

    size_filter_cfg = {
        "button_class_id": button_class_id,
        "dynamic_factor": cfg.button_min_side_factor,
        "min_px_floor": cfg.button_min_px
    }

    cascade_cfg = {
        "enable": cfg.enable_cascade,
        "expand": cfg.cascade_expand,
        "upscale": cfg.cascade_upscale,
        "sharpen": cfg.cascade_sharpen,
        "button_cascade_thr": cfg.cascade_button_thr,
        "button_class_id": button_class_id,
        "input_size": cfg.input_size,
        "pad_color": cfg.pad_color
    }

    detection_counts = {i:0 for i in range(num_classes)}
    last_button_frame = -1

    frame_idx = 0
    writer = None
    printed_shape = False

    raw_btn_totals = []
    raw_btn_overthr = []

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()

        # Enhacement global opcional
        proc_frame = enhance_frame(frame, use_clahe=cfg.use_clahe)

        tensor, meta = letterbox(proc_frame, size=cfg.input_size, pad_color=cfg.pad_color)
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if raw.ndim == 3:
            raw = np.squeeze(raw,0)

        if not printed_shape:
            print("[DEBUG] Output raw shape:", raw.shape)
            if raw.shape[0] > 0:
                print("[DEBUG] Primera fila:", raw[0])
            printed_shape = True

        # Debug distribución por clase cada X frames
        if cfg.debug_interval > 0 and frame_idx % cfg.debug_interval == 0:
            unique, counts = np.unique(raw[:,5].astype(int), return_counts=True)
            dist = dict(zip(unique, counts))
            btn_mask = raw[:,5].astype(int) == button_class_id
            tot_btn = int(np.sum(btn_mask))
            over_thr = int(np.sum(btn_mask & (raw[:,4] > thresholds[button_class_id])))
            print(f"[DBG f{frame_idx:04d}] class_dist={dist} btn_total={tot_btn} btn>thr={over_thr} thrBtn={thresholds[button_class_id]:.2f}")
            raw_btn_totals.append(tot_btn)
            raw_btn_overthr.append(over_thr)
            if cfg.show_topk > 0:
                top_idx = np.argsort(-raw[:,4])[:cfg.show_topk]
                print("[TOPK]")
                for i in top_idx:
                    print(f"  {i}: {raw[i]}")

        # Decode base
        boxes, scores, class_ids = decode_raw_format6(
            raw, meta, thresholds, num_classes, size_filter_cfg
        )

        # Multi-escala global si habilitado y (a) no hay Button todavía
        if cfg.enable_multiscale and (not np.any(class_ids == button_class_id)):
            b2,s2,c2 = multi_scale_pass(
                session, proc_frame, cfg.multiscale_factor,
                cfg.input_size, thresholds, num_classes, size_filter_cfg
            )
            if b2.size:
                boxes = np.vstack([boxes,b2]) if boxes.size else b2
                scores = np.concatenate([scores,s2]) if scores.size else s2
                class_ids = np.concatenate([class_ids,c2]) if class_ids.size else c2

        # Cascada sobre Micro (solo si aún sin Button o queremos más recall)
          # Obtenemos micro boxes antes de NMS final para densidad
        if cascade_cfg["enable"]:
            micro_boxes = boxes[class_ids == 1] if boxes.size else np.empty((0,4))
            if micro_boxes.size:
                cb, cs, cc = cascade_button(
                    session, proc_frame, micro_boxes,
                    cascade_cfg, thresholds, num_classes, size_filter_cfg
                )
                if cb.size:
                    boxes = np.vstack([boxes, cb]) if boxes.size else cb
                    scores = np.concatenate([scores, cs]) if scores.size else cs
                    class_ids = np.concatenate([class_ids, cc]) if class_ids.size else cc

        # NMS al final
        if cfg.enable_nms and boxes.size:
            boxes, scores, class_ids = per_class_nms(
                boxes, scores, class_ids, thresholds,
                cfg.iou_nms, button_class_id
            )

        # Temporal accumulation
        if temporal and boxes.size:
            temporal.update(frame_idx, boxes, scores, class_ids)
            if not np.any(class_ids == button_class_id):
                prop = temporal.propose(frame_idx)
                if prop is not None:
                    bmean, smean = prop
                    boxes = np.vstack([boxes, bmean]) if boxes.size else np.array([bmean])
                    scores = np.concatenate([scores, [smean]]) if scores.size else np.array([smean])
                    class_ids = np.concatenate([class_ids, [button_class_id]]) if class_ids.size else np.array([button_class_id])

        # Conteo
        for c in class_ids:
            detection_counts[int(c)] += 1
            if c == button_class_id:
                last_button_frame = frame_idx

        # Relajación threshold Button
        if frame_idx - last_button_frame > cfg.relax_every:
            thresholds[button_class_id] = max(
                min_thresholds[button_class_id],
                thresholds[button_class_id] - cfg.relax_step
            )

        # Dibujo
        vis = frame.copy()
        vis = draw_annotations(vis, boxes, scores, class_ids, class_names, button_class_id=button_class_id)
        stats = f"{class_names[0][0]}:{detection_counts.get(0,0)} {class_names[1][0]}:{detection_counts.get(1,0)} {class_names[2][0]}:{detection_counts.get(2,0)} ThrB:{thresholds[button_class_id]:.2f}"
        cv2.putText(vis, stats, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,255,0),2)
        fps = 1.0 / (time.time()-t0+1e-6)
        cv2.putText(vis, f"{fps:.1f} FPS",(10,50), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)

        if cfg.output_path:
            if writer is None:
                h,w = vis.shape[:2]
                writer = cv2.VideoWriter(cfg.output_path,
                                         cv2.VideoWriter_fourcc(*'mp4v'),
                                         cfg.output_fps, (w,h))
            writer.write(vis)

        cv2.imshow("Deteccion", vis)
        if cv2.waitKey(1) == 27:
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = time.time() - t_start
    print("=== RESUMEN ===")
    print("Frames:", frame_idx, "Tiempo(s):", f"{total_time:.1f}", "FPS medio:", f"{frame_idx/(total_time+1e-6):.2f}")
    for i,nm in enumerate(class_names):
        print(f"{nm}: {detection_counts.get(i,0)}")
    print(f"Threshold final Button: {thresholds[button_class_id]:.3f}")
    if raw_btn_totals:
        print("Promedio propuestas Button crudas:", f"{np.mean(raw_btn_totals):.2f}")
        print("Promedio propuestas > thr Button:", f"{np.mean(raw_btn_overthr):.2f}")


# ================== CONFIG PARSER ==================
class Cfg:
    pass

def build_cfg(args):
    cfg = Cfg()
    cfg.video_path = args.video
    cfg.model_path = args.model
    cfg.output_path = args.output
    cfg.output_fps = args.out_fps

    cfg.class_names = DEFAULT_CLASS_NAMES
    cfg.button_class_id = 2

    cfg.input_size = args.input_size
    cfg.pad_color = 114

    # Thresholds
    cfg.base_thresholds = [args.hand_base_thr, args.micro_base_thr, args.btn_base_thr]
    cfg.min_thresholds  = [args.hand_min_thr,  args.micro_min_thr,  args.btn_min_thr]

    cfg.relax_every = args.relax_every
    cfg.relax_step = args.relax_step

    # NMS
    cfg.enable_nms = not args.no_nms
    cfg.iou_nms = args.iou_nms

    # Multi escala
    cfg.enable_multiscale = args.multiscale
    cfg.multiscale_factor = args.multiscale_factor

    # Cascade
    cfg.enable_cascade = args.cascade
    cfg.cascade_expand = args.cascade_expand
    cfg.cascade_upscale = args.cascade_upscale
    cfg.cascade_sharpen = args.cascade_sharpen
    cfg.cascade_button_thr = args.cascade_btn_thr

    # Temporal
    cfg.enable_temporal = not args.no_temporal
    cfg.temp_window = args.temp_window
    cfg.temp_confirm_score = args.temp_confirm_score
    cfg.temp_min_consec = args.temp_min_consec
    cfg.temp_max_dist = args.temp_max_dist

    # Button size dynamic
    cfg.button_min_side_factor = args.btn_min_side_factor
    cfg.button_min_px = args.btn_min_px

    # Enhancement
    cfg.use_clahe = args.clahe

    # Debug
    cfg.debug_interval = args.debug_every
    cfg.show_topk = args.show_topk

    return cfg

# ================== MAIN ==================
def parse_args():
    ap = argparse.ArgumentParser(description="Video detection con refuerzo para Button")
    ap.add_argument("--video", required=True, help="Ruta video")
    ap.add_argument("--model", required=True, help="Ruta modelo ONNX")
    ap.add_argument("--output", default=None, help="Ruta opcional video salida")
    ap.add_argument("--out_fps", type=int, default=30)

    ap.add_argument("--input_size", type=int, default=640)

    # Thresholds base y mínimos
    ap.add_argument("--hand_base_thr", type=float, default=0.45)
    ap.add_argument("--micro_base_thr", type=float, default=0.45)
    ap.add_argument("--btn_base_thr", type=float, default=0.18)
    ap.add_argument("--hand_min_thr", type=float, default=0.28)
    ap.add_argument("--micro_min_thr", type=float, default=0.30)
    ap.add_argument("--btn_min_thr", type=float, default=0.06)
    ap.add_argument("--relax_every", type=int, default=90, help="Frames sin Button para relajar")
    ap.add_argument("--relax_step", type=float, default=0.025)

    # NMS
    ap.add_argument("--no_nms", action="store_true")
    ap.add_argument("--iou_nms", type=float, default=0.55)

    # Multi escala
    ap.add_argument("--multiscale", action="store_true", help="Activar segundo pase global escalado")
    ap.add_argument("--multiscale_factor", type=float, default=1.5)

    # Cascada
    ap.add_argument("--cascade", action="store_true", help="Activar cascada sobre Micro")
    ap.add_argument("--cascade_expand", type=float, default=0.12)
    ap.add_argument("--cascade_upscale", type=int, default=800)
    ap.add_argument("--cascade_sharpen", action="store_true")
    ap.add_argument("--cascade_btn_thr", type=float, default=0.05)

    # Temporal
    ap.add_argument("--no_temporal", action="store_true")
    ap.add_argument("--temp_window", type=int, default=10)
    ap.add_argument("--temp_confirm_score", type=float, default=0.24)
    ap.add_argument("--temp_min_consec", type=int, default=3)
    ap.add_argument("--temp_max_dist", type=int, default=18)

    # Tamaño mínimo dinámico Button
    ap.add_argument("--btn_min_side_factor", type=float, default=0.005, help="Factor * min(h,w)")
    ap.add_argument("--btn_min_px", type=float, default=4, help="Piso duro px")

    # Enhancement
    ap.add_argument("--clahe", action="store_true", help="Aplicar CLAHE global")

    # Debug
    ap.add_argument("--debug_every", type=int, default=40, help="Intervalo frames para imprimir dist clases (0=off)")
    ap.add_argument("--show_topk", type=int, default=0, help="Mostrar top-K filas primera vez (0=off)")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = build_cfg(args)
    run_video(cfg)