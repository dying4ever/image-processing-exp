# 数字图像处理实验

本仓库包含三个数字图像处理实验：

- 实验一：边缘提取与滤波
- 实验二：字符分割与距离测量
- 实验三：视觉测量及参数计算

每个实验目录中包含：

- `experiment*.py`：Python/OpenCV 实验代码
- `results/`：脚本运行生成的结果图和结果摘要
- `实验过程记录.md`：实验运行过程记录，已直接嵌入结果图片
- `实验报告.md`：依据实验指导书整理的实验报告，已直接嵌入结果图片

## 运行环境

使用 Conda 环境 `opencv` 运行：

```powershell
conda run -n opencv python 实验一/experiment1.py
conda run -n opencv python 实验二/experiment2.py
conda run -n opencv python 实验三/experiment3.py
```

主要依赖：

- Python
- OpenCV
- NumPy

