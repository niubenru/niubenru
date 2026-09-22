# 猫狗图像分类实验：从零搭建到部署上线

按《猫狗分类实验指导书》完成的完整实验项目：手动搭建模型 → 理解训练/测试/评价指标流程 → 迁移学习 → 数据增强 → 多模型对比 → 保存最佳权重 → FastAPI 前后端部署。

## 项目结构

| 文件                    | 作用                                                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `download_data.py`    | 下载猫狗图片，组织成 `datasets/`（cat.N.jpg / dog.N.jpg，标签藏在文件名）                                                                       |
| `common.py`           | 公共模块：`FileNameDataset`（文件名解析标签）、`get_loaders`（固定 seed=42 划分 80/20）、`evaluate`（准确率/精确率/召回率/F1）、`train_and_eval`（训练循环+最佳模型保存） |
| `step1_scratch.py`    | 第一步：手动搭建精简版 AlexNet，Xavier 初始化从零训练                                                                                          |
| `step2_pretrained.py` | 第二步：ImageNet 预训练 ResNet18，冻结特征层只训分类头（`finetune` 参数可切换为全模型微调）                                                                |
| `step3_4_compare.py`  | 第三+四步：数据增强 + resnet18/mobilenet_v3_small/shufflenet_v2 横向对比                                                                 |
| `train_all.py`        | 实验编排：跑全部对比实验，指标写入 `metrics.json`，最佳模型存 `best_model.pth`                                                                     |
| `predict.py`          | 第五步：单张图片推理（`python predict.py best_model.pth 图片路径`）                                                                         |
| `app.py`              | 第六步：FastAPI 后端（`/predict`、`/metrics` 接口 + 托管前端页面）                                                                           |
| `static/index.html`   | 前端：拍照/上传 → 识别 → 展示类别概率 + 实验指标表                                                                                              |
| `metrics.json`        | 各实验最终指标（前端自动加载展示）                                                                                                           |

## 快速开始

```bash
pip install -r requirements.txt

# 1. 训练（CPU 约 10~20 分钟；quick 模式只跑核心两个实验）
python train_all.py

# 2. 单图测试
python predict.py best_model.pth datasets/cat.1.jpg

# 3. 启动前后端服务
python app.py
# 浏览器访问 http://127.0.0.1:8000
# 手机（同一 Wi-Fi）访问 http://<本机IP>:8000
```

## 实验设计要点（控制变量）

- **同一份数据划分**：`get_loaders` 固定 `seed=42`，80% 训练 / 20% 验证，各实验严格可比；
- **同一套指标**：macro 平均的 Precision / Recall / F1 + Accuracy，零除保护（`zero_division=0`）；
-     
- **早停思想**：每个 epoch 验证一次，只保存验证集准确率最高的 `best_model.pth`。

## API 说明

| 接口         | 方法   | 说明                                                          |
| ---------- | ---- | ----------------------------------------------------------- |
| `/`        | GET  | 前端页面                                                        |
| `/predict` | POST | multipart 上传图片字段 `file`，返回 `{"label","confidence","probs"}` |
| `/metrics` | GET  | 返回 `metrics.json` 中的各实验指标                                   |
