# 模型权重说明

更新时间：2026-08-25

## 当前文件

| 文件 | Ultralytics task | checkpoint 训练数据 | 当前用途 |
|---|---|---|---|
| `yolo26n.pt` | `detect` | COCO | 人体及环境物体检测 |
| `yolo26n-pose.pt` | `pose` | COCO-Pose | 人体姿态关键点提取 |

原始交付文件位于：

- `wwh/yolo26n..pt`
- `wwh/yolo26n-pose..pt`

接入时只修正了重复句点文件名，没有修改权重内容。

## SHA-256

```text
9B09CC8BF347F0FC8A5F7657480587F25DB09B34BF33B0652110FB03A8AD4FEF  yolo26n.pt
EB3BB8268828AEAF515CEC23A4BFAFD793944A86FE9AF94BA7823609C14522A9  yolo26n-pose.pt
```

## 能力边界

这两份文件是真实、可加载的 YOLO26 通用预训练权重，但不是本项目使用跌倒数据微调得到的分类模型：

- `yolo26n.pt` 的 checkpoint 元数据指向 COCO；
- `yolo26n-pose.pt` 的 checkpoint 元数据指向 COCO-Pose；
- 交付包说明科学 baseline 通过多折模型评估，目前没有单一冻结的可部署 predictor artifact；
- 当前系统的风险输出仍来自姿态/运动启发、特征、个体基线与规则融合，不能解释为校准跌倒概率。
- 音频识别当前仅保留接口占位，不属于本次权重替换或视觉/环境多模态验收范围。

旧的随机 `test_best.pt`、合成关键点样本和占位标签已删除。
