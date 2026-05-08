import os
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)


def imread(path, flags=cv2.IMREAD_COLOR):
    # 使用 imdecode 读取中文路径图片，避免 cv2.imread 在 Windows 中文路径下失败。
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def imwrite(path, image):
    # 使用 imencode 保存中文路径图片，保证结果能写入对应实验目录。
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    buf.tofile(str(path))


def put_label(image, text, org):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


def edge_extract():
    # 指导书步骤：读取原图 -> 阈值分割 -> 反色 -> 提取边缘轮廓。
    image = imread(BASE_DIR / "齿轮.bmp")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value = 180
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    inverse = 255 - binary
    # RETR_EXTERNAL 提取外部轮廓，CHAIN_APPROX_SIMPLE 压缩冗余轮廓点。
    contours, _ = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    put_label(contour_image, f"contours: {len(contours)}", (20, 40))

    imwrite(RESULT_DIR / "01_gray.png", gray)
    imwrite(RESULT_DIR / "02_threshold.png", binary)
    imwrite(RESULT_DIR / "03_inverse.png", inverse)
    imwrite(RESULT_DIR / "04_edge_contours.png", contour_image)

    return {
        "source": "齿轮.bmp",
        "shape": image.shape,
        "threshold": threshold_value,
        "contours": len(contours),
    }


def image_filter():
    # 指导书步骤：读取原图后分别进行均值、中值、高斯滤波。
    image = imread(BASE_DIR / "20210416144049781.png")
    mean_filter = cv2.blur(image, (5, 5))
    median_filter = cv2.medianBlur(image, 5)
    gaussian_filter = cv2.GaussianBlur(image, (5, 5), 0)
    # 高斯滤波先抑制噪声，再用 Canny 得到较稳定的边缘。
    gaussian_edge = cv2.Canny(gaussian_filter, 80, 160)

    imwrite(RESULT_DIR / "05_filter_original.png", image)
    imwrite(RESULT_DIR / "06_mean_filter.png", mean_filter)
    imwrite(RESULT_DIR / "07_median_filter.png", median_filter)
    imwrite(RESULT_DIR / "08_gaussian_filter.png", gaussian_filter)
    imwrite(RESULT_DIR / "09_gaussian_canny_edge.png", gaussian_edge)

    return {
        "source": "20210416144049781.png",
        "shape": image.shape,
        "mean_kernel": "5x5",
        "median_kernel": 5,
        "gaussian_kernel": "5x5",
        "canny_threshold": "80,160",
    }


def edge_detection():
    # 指导书步骤：读取原图 -> 反色 -> Sobel 和 Robert 边缘检测。
    image = imread(BASE_DIR / "齿轮.bmp")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverse = 255 - gray

    sobel_x = cv2.Sobel(inverse, cv2.CV_16S, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(inverse, cv2.CV_16S, 0, 1, ksize=3)
    sobel = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)

    # Robert 算子使用 2x2 对角差分模板，适合检测细小灰度突变。
    robert_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    robert_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    rx = cv2.filter2D(inverse, cv2.CV_16S, robert_x)
    ry = cv2.filter2D(inverse, cv2.CV_16S, robert_y)
    robert = cv2.addWeighted(cv2.convertScaleAbs(rx), 0.5, cv2.convertScaleAbs(ry), 0.5, 0)

    imwrite(RESULT_DIR / "10_detection_inverse.png", inverse)
    imwrite(RESULT_DIR / "11_sobel_edge.png", sobel)
    imwrite(RESULT_DIR / "12_robert_edge.png", robert)

    return {
        "source": "齿轮.bmp",
        "sobel_kernel": 3,
        "robert_kernel": "2x2",
    }


def main():
    edge_info = edge_extract()
    filter_info = image_filter()
    detection_info = edge_detection()

    lines = [
        "Experiment 1 result summary",
        f"Edge source: {edge_info['source']}, shape: {edge_info['shape']}",
        f"Threshold: {edge_info['threshold']}, contours: {edge_info['contours']}",
        f"Filter source: {filter_info['source']}, shape: {filter_info['shape']}",
        f"Mean/median/gaussian parameters: {filter_info['mean_kernel']}, {filter_info['median_kernel']}, {filter_info['gaussian_kernel']}",
        f"Gaussian Canny threshold: {filter_info['canny_threshold']}",
        f"Detection source: {detection_info['source']}, Sobel kernel: {detection_info['sobel_kernel']}, Robert kernel: {detection_info['robert_kernel']}",
    ]
    (RESULT_DIR / "result_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("experiment1 ok")


if __name__ == "__main__":
    main()
