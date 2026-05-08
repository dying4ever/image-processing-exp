from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def imread(path, flags=cv2.IMREAD_COLOR):
    # 中文文件名用 imdecode 读取，避免 Windows 下 cv2.imread 路径编码问题。
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def imwrite(path, image):
    # 中文目录下保存图片时使用 imencode + tofile。
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    buf.tofile(str(path))


def license_plate_segmentation():
    # 指导书步骤：灰度转换并反色，然后阈值分割获得字符区域。
    image = imread(BASE_DIR / "车牌.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverse = 255 - gray

    # 先用蓝色背景定位车牌 ROI，减少白色外背景对字符分割的干扰。
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 50, 20]), np.array([140, 255, 255]))
    blue_contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(blue_contours, key=cv2.contourArea))

    roi_gray = gray[y:y + h, x:x + w]
    roi_inverse = inverse[y:y + h, x:x + w]
    # Otsu 阈值自动确定分割阈值，适合车牌字符和背景对比明显的图像。
    _, binary = cv2.threshold(roi_inverse, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary[:, : int(w * 0.06)] = 0
    binary[:, int(w * 0.94):] = 0
    binary[: int(h * 0.08), :] = 0
    binary[int(h * 0.92):, :] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_boxes = []
    height, width = binary.shape
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        # 按字符宽高和面积筛掉圆点、边框、噪声等非字符区域。
        if 6 < cw < width * 0.22 and height * 0.25 < ch < height * 0.85 and area > 80:
            char_boxes.append((x + cx, y + cy, cw, ch))
    char_boxes = sorted(char_boxes, key=lambda box: box[0])

    marked = image.copy()
    for idx, (bx, by, bw, bh) in enumerate(char_boxes, start=1):
        cv2.rectangle(marked, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(marked, str(idx), (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    imwrite(RESULT_DIR / "01_plate_gray.png", gray)
    imwrite(RESULT_DIR / "02_plate_inverse.png", inverse)
    imwrite(RESULT_DIR / "03_plate_threshold.png", binary)
    imwrite(RESULT_DIR / "04_plate_char_segmentation.png", marked)
    return char_boxes


def component_width_measurement():
    # 指导书步骤：截取目标区域 -> 阈值分割 -> 提取轮廓 -> 绘制宽度线。
    image = imread(BASE_DIR / "器件测量.bmp")
    # 图中左侧器件作为测量对象，先截取 ROI。
    image = image[:, : image.shape[1] // 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 5000]
    # 最大轮廓对应被测器件主体，用外接矩形估算其像素宽度。
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    marked = image.copy()
    cv2.drawContours(marked, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.line(marked, (x, y), (x, y + h), (255, 0, 0), 2)
    cv2.line(marked, (x + w, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(marked, f"width={w}px", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    imwrite(RESULT_DIR / "05_component_threshold.png", binary)
    imwrite(RESULT_DIR / "06_component_width.png", marked)
    return {"x": x, "y": y, "width_px": w, "height_px": h, "area_px": cv2.contourArea(contour)}


def object_distance_measurement():
    # 指导书步骤：灰度去噪 -> Canny -> 膨胀腐蚀 -> 外接矩形 -> 距离换算。
    image = imread(BASE_DIR / "物件距离检测.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edge = cv2.Canny(blur, 50, 120)
    edge = cv2.dilate(edge, None, iterations=2)
    edge = cv2.erode(edge, None, iterations=1)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 15 or h < 15:
            continue
        objects.append({"box": (x, y, w, h), "center": (x + w / 2, y + h / 2), "area": area})
    objects = sorted(objects, key=lambda item: item["box"][0])

    # 以最左侧硬币为参考对象，设定其实际宽度为 24.26 mm。
    reference_width_mm = 24.26
    reference = objects[0]
    pixels_per_mm = reference["box"][2] / reference_width_mm

    marked = image.copy()
    measurements = []
    rx, ry = reference["center"]
    for idx, obj in enumerate(objects):
        x, y, w, h = obj["box"]
        cx, cy = obj["center"]
        cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(marked, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(marked, f"O{idx}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if idx > 0:
            # 根据两个目标中心点坐标计算像素距离，再用像素比例换算为毫米。
            distance_px = float(np.hypot(cx - rx, cy - ry))
            distance_mm = distance_px / pixels_per_mm
            measurements.append((idx, distance_px, distance_mm))
            cv2.line(marked, (int(rx), int(ry)), (int(cx), int(cy)), (255, 0, 0), 2)
            cv2.putText(marked, f"{distance_mm:.1f}mm", (int((rx + cx) / 2), int((ry + cy) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    imwrite(RESULT_DIR / "07_distance_gray.png", gray)
    imwrite(RESULT_DIR / "08_distance_canny.png", edge)
    imwrite(RESULT_DIR / "09_object_distance.png", marked)
    return reference_width_mm, pixels_per_mm, objects, measurements


def main():
    char_boxes = license_plate_segmentation()
    width_info = component_width_measurement()
    reference_width_mm, pixels_per_mm, objects, measurements = object_distance_measurement()

    lines = [
        "Experiment 2 result summary",
        f"Plate segmented character boxes: {len(char_boxes)}",
        f"Character boxes x,y,w,h: {char_boxes}",
        f"Component width: {width_info['width_px']} px",
        f"Component bbox x,y,w,h: ({width_info['x']}, {width_info['y']}, {width_info['width_px']}, {width_info['height_px']})",
        f"Reference width: {reference_width_mm} mm",
        f"Pixels per mm: {pixels_per_mm:.4f}",
        f"Detected objects: {len(objects)}",
    ]
    for idx, distance_px, distance_mm in measurements:
        lines.append(f"O0 to O{idx}: {distance_px:.2f} px, {distance_mm:.2f} mm")
    (RESULT_DIR / "result_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("experiment2 ok")


if __name__ == "__main__":
    main()
