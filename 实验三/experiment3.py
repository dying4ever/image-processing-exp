from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def imread(path, flags=cv2.IMREAD_COLOR):
    # 兼容中文路径读取，避免 cv2.imread 直接读取失败。
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def imwrite(path, image):
    # 兼容中文路径保存，结果图统一写入 results 目录。
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    buf.tofile(str(path))


def green_mask(image):
    # 绿色工件与黑色背景差异明显，HSV 阈值用于提取工件区域。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 70, 60]), np.array([90, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return mask


def hole_contours(image):
    # 先填充工件外轮廓，再只在工件内部寻找黑色孔洞，排除外部黑背景。
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = green_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    if contours:
        cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
    dark = cv2.inRange(gray, 0, 70)
    holes = cv2.bitwise_and(dark, solid)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) > 30], holes


def detect_round_holes(image_path, output_name):
    # 圆孔检测：提取孔洞轮廓，按圆度过滤，再计算最小外接圆。
    image = imread(BASE_DIR / image_path)
    contours, holes = hole_contours(image)

    circles = []
    marked = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        # 圆度越接近 1 越接近圆形，用于过滤非圆孔噪声。
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if circularity > 0.45 and 3 < radius < 80:
            circles.append((float(x), float(y), float(radius), float(area)))

    circles = sorted(circles, key=lambda item: (item[1], item[0]))
    for idx, (x, y, radius, area) in enumerate(circles):
        cv2.circle(marked, (int(x), int(y)), int(radius), (0, 0, 255), 2)
        cv2.circle(marked, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv2.putText(marked, f"C{idx}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    imwrite(RESULT_DIR / f"{output_name}_holes_mask.png", holes)
    imwrite(RESULT_DIR / f"{output_name}_round_holes.png", marked)
    return circles


def detect_slot_holes():
    # 长形孔检测：对孔洞轮廓求最小外接矩形，长宽比大于 2 视为长形孔。
    image = imread(BASE_DIR / "image2.jpg")
    contours, holes = hole_contours(image)

    slots = []
    marked = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 250:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        length = max(w, h)
        width = min(w, h)
        if width == 0 or length / width < 2.0:
            continue
        slots.append((cx, cy, length, width, area, rect, contour))

    slots = sorted(slots, key=lambda item: item[1])
    for idx, (cx, cy, length, width, area, rect, contour) in enumerate(slots):
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(marked, [box], 0, (0, 0, 255), 2)
        cv2.putText(marked, f"S{idx}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

    imwrite(RESULT_DIR / "image2_slots_mask.png", holes)
    imwrite(RESULT_DIR / "image2_slot_detection.png", marked)
    return slots


def pair_distances(circles, limit=None):
    # 根据圆心坐标计算两两孔心距，结果单位为像素。
    pairs = []
    selected = circles[:limit] if limit else circles
    for (i, a), (j, b) in combinations(enumerate(selected), 2):
        distance = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        pairs.append((i, j, distance))
    return pairs


def main():
    circles1 = detect_round_holes("image1.jpg", "image1")
    slots = detect_slot_holes()
    circles3 = detect_round_holes("image3.jpg", "image3")

    top_two_image3 = sorted(circles3, key=lambda item: item[1])[:2]
    top_distance = pair_distances(top_two_image3)

    lines = [
        "Experiment 3 result summary",
        f"image1 round holes: {len(circles1)}",
    ]
    for idx, (x, y, radius, area) in enumerate(circles1):
        lines.append(f"image1 C{idx}: center=({x:.1f},{y:.1f}), radius={radius:.1f}px, area={area:.1f}px2")
    for i, j, distance in pair_distances(circles1):
        lines.append(f"image1 C{i}-C{j}: {distance:.2f}px")

    lines.append(f"image2 slot count: {len(slots)}")
    for idx, (cx, cy, length, width, area, rect, _) in enumerate(slots):
        lines.append(f"image2 S{idx}: center=({cx:.1f},{cy:.1f}), length={length:.1f}px, width={width:.1f}px, area={area:.1f}px2")

    lines.append(f"image3 round holes: {len(circles3)}")
    for idx, (x, y, radius, area) in enumerate(circles3):
        lines.append(f"image3 C{idx}: center=({x:.1f},{y:.1f}), radius={radius:.1f}px, area={area:.1f}px2")
    for i, j, distance in top_distance:
        lines.append(f"image3 top two centers C{i}-C{j}: {distance:.2f}px")

    (RESULT_DIR / "result_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("experiment3 ok")


if __name__ == "__main__":
    main()
