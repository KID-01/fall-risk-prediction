# Wet-Floor Provider 接入决策（SynSpill 审计）

## 决策

**NO-GO for automatic detector integration in v0.3.**

v0.3 保留 `wet_floor_unavailable` 契约：

```json
{
  "source": "wet_floor_unavailable",
  "available": false,
  "risk_index": null,
  "state": "UNKNOWN",
  "regions": [],
  "reason_codes": ["detector_unavailable"]
}
```

## 审计对象

- 项目：`eternal-f1ame/SynSpill`；
- 日期：2026-08-25；
- 方式：GitHub README、recursive tree、data 目录、license、release API 只读审计；
- 未克隆、未下载大数据、未训练模型。

## 证据

1. 仓库公开重点是合成水渍数据、annotation masks、ComfyUI generation/inpainting workflow 和展示前端；
2. recursive tree 未确认可直接部署的 `.pt/.pth/.onnx/.ckpt` checkpoint；
3. 未确认独立实时 wet-floor detection/segmentation inference 入口；
4. GitHub `releases/latest` 返回 404，无可直接下载 release；
5. 数据和 associated materials 许可证明确为 non-commercial research/education only，并要求引用、保留条款；
6. 当前项目没有真实湿地面测试样本，无法做 false-positive sanity；
7. 因而无法在 1–2 天内诚实声明具备可验证的自动水渍识别能力。

## 允许借鉴

- 水渍区域应使用 segmentation/mask 契约，而非普通 COCO object box；
- 合成数据需人工审核；
- 数据集与模型许可证需独立记录；
- 未来可把 wet-floor mask 接入 `interaction_v0` 的 predicted path corridor。

## 禁止

- 人工绘制水渍 mask 后声称自动检测；
- 把 SynSpill annotation mask 当推理输出；
- 未训练/未验证就显示 Wet Floor HIGH；
- 把水渍模块说成提高 frozen EWR@0.5；
- 把受限材料打进比赛包而不附许可证和引用。

## 重新打开条件

仅在以下条件全部满足后重新评估：

1. 有许可证可用的 checkpoint；
2. 有清晰推理入口和输出 schema；
3. 当前环境可运行；
4. 有真实 wet/dry floor sanity 集；
5. 有固定 split、标签和误报评估；
6. 单独 Human Gate 批准模型/数据接入。
