# 灵境第二阶段桌面物品视觉对比

本项目使用同一组桌面图片，对比两种视觉方案：

- 传统方案：OpenCV、YOLO 目标检测与 SAM 图像分割
- 多模态方案：视觉语言模型生成物品清单、空间描述和场景总结

项目重点不是单纯判断哪个模型更准确，而是比较两种方案在识别范围、
可解释性、复杂场景表现、响应速度和使用成本方面的差异。

## 主要结果

项目使用五张桌面图片完成对比，大模型为智谱 `glm-4.6v-flash`。

| 指标 | 传统方案 YOLO+SAM | 视觉大模型 |
|---|---:|---:|
| 平均耗时 | 9.43 秒 | 38.80 秒 |
| 平均 Token | - | 3691.8 |
| 输出边界框 | 支持 | 不支持 |
| 分割掩膜 | 支持 | 不支持 |
| 自然语言描述 | 不支持 | 支持 |
| 固定类别限制 | COCO 80 类 | 开放类别描述 |

五张图片中，大模型声明的物品总数与物品数量之和一致率为 `5/5`。

主要观察：

- YOLO 提供稳定的类别、置信度、边界框和 SAM 掩膜。
- YOLO 受固定 COCO 类别限制，无法识别插座、支架、装饰物等类别。
- 视觉大模型能够描述鼠标垫、桌面支架、插座、装饰物等更丰富的内容。
- 视觉大模型没有稳定的边界框和像素级分割结果。
- 传统方案平均约 9.43 秒，大模型平均约 38.80 秒，传统方案约快 4.14 倍。
- 免费视觉大模型存在限流和推理 Token 消耗，需要重试和输出长度控制。

## 项目结构

```text
.
├── assets/samples/
├── configs/
├── data/
│   ├── images/
│   └── ground_truth/
├── docs/
│   ├── project_report.md
│   └── video_script.md
├── outputs/
│   ├── yolo/
│   ├── sam/
│   ├── vlm/
│   ├── comparison/
│   ├── timing/
│   └── summary/
├── scripts/
│   └── run_batch.ps1
├── src/
│   ├── yolo_demo.py
│   ├── sam_segment.py
│   ├── vlm_describe.py
│   ├── compare_results.py
│   └── summarize_results.py
├── .env.example
├── environment.yml
├── requirements.txt
└── README.md