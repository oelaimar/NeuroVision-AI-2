from pathlib import Path
import cv2
from src.config import cfg

CLASS_NAMES = list(cfg.CLASS_NAMES)

def read_yolo_label(label_path):
    """Read a YOLO .txt file. Returns a list of (class_id, xc, yc, w, h)."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls, xc, yc, w, h = parts[:5]
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """Normalized center format -> pixel corners (x1, y1, x2, y2).
        YOLO to pixels
    """
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return x1, y1, x2, y2

def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Pixel corners -> normalized center format (xc, yc, w, h).
        pixels to YOLO
    """
    return (
        (x1 + x2) / 2 / img_w,
        (y1 + y2) / 2 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )

def draw_boxes(image, boxes, class_names = CLASS_NAMES, color = (0, 255, 0)): # (0, 255, 0) is pure green
    """Draw YOLO boxes on a BGR image (OpenCV). Returns a copy."""
    img = image.copy()
    img_h, img_w = img.shape[:2]
    for cls, xc, yc, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[cls], (x1, max(y1 -  6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def validate_label(label_path, num_classes=len(CLASS_NAMES)):
    """Return a list of error messages (empty list = valid file)."""
    errors = []
    for i, line in enumerate(Path(label_path).read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line {i}: non-numeric value (class must be an integer)")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"line {i}: class {cls} not in 0..{num_classes - 1}")
        if any(not 0.0 <= v <= 1.0 for v in coords):
            errors.append(f"line {i}: coordinates must be between 0 and 1")
    return errors

if __name__ == "__main__":
    # Demo: python src/annotations.py path/to/image.jpg path/to/label.txt
    import sys
    img_path, lbl_path = sys.argv[1:3]
    image = cv2.imread(img_path)
    boxes = read_yolo_label(lbl_path)
    print("boxes:", boxes)
    print("errors:", validate_label(lbl_path) or "none")
    cv2.imwrite("preview.jpg", draw_boxes(image, boxes))
    print("saved preview.jpg")
