# SAR-clear

ClearSAR 比赛冲榜实用流程（单类别目标检测，指标 mAP50-95）。

## 1) 训练

仓库里已有训练脚本：

- `scripts/train.py`：常规训练。
- `scripts/train_full.py`：关闭 early stopping，跑满轮数。

## 2) 生成原始预测

先用较低置信度导出原始预测（保留更多候选框，后处理再筛）：

```bash
python scripts/predict.py
```

可导出多份不同模型/不同 checkpoint 的 `submission_xxx.json`，后续做融合。

## 3) 后处理（NMS / Box Voting）

`postprocess_predictions.py` 支持：

- 多个预测文件融合（直接合并候选框）
- 每个输入文件设置权重（`--input-weights`）
- `nms` 或 `vote`（加权框融合）
- 统一 `conf / iou / max_det` 控制
- 可选按 COCO `images` 尺寸裁剪边界框，避免越界框（`--clip-with-ann`）

示例：

```bash
python scripts/postprocess_predictions.py \
  --inputs scripts/submission_modelA.json scripts/submission_modelB.json \
  --input-weights 1.0,0.8 \
  --method vote \
  --conf 0.15 \
  --iou 0.55 \
  --max-det 300 \
  --clip-with-ann /path/to/instances_val.json \
  --output scripts/submission_fused.json
```

## 4) 在本地 val 上网格搜索最优参数

`search_postprocess.py` 用本地验证集自动搜索参数，目标直接对齐 mAP50-95：

```bash
python scripts/search_postprocess.py \
  --inputs scripts/submission_modelA_val.json scripts/submission_modelB_val.json \
  --input-weights 1.0,0.8 \
  --ann /path/to/instances_val.json \
  --methods vote,nms \
  --confs 0.05,0.1,0.15,0.2,0.25 \
  --ious 0.45,0.5,0.55,0.6 \
  --max-dets 100,200,300 \
  --save-best scripts/submission_best_val_tuned.json \
  --save-meta scripts/submission_best_val_tuned_meta.json
```

然后把最佳参数迁移到 test 预测文件，生成最终提交。

---

## 冲榜建议（每天 1 次提交）

- 固定一个“主力模型 + 主力后处理参数”，只让一次提交承担**一个明确增益点**。
- 提交前先在本地 val 对比：
  - 单模 vs 多模融合
  - `vote` vs `nms`
  - `max_det` 提升是否带来长尾召回
- 记录每次实验（模型、epoch、imgsz、conf、iou、max_det、val mAP），避免重复试错。
