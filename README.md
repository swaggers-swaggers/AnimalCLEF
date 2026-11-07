# AnimalCLEF 2026 — V7

动物个体重识别实验代码与结果。V7 的历史开发留出六项平均 ARI 为 **0.637717**，V6 为 **0.565657**；该结果不是 Kaggle 官方分数，也不是全新独立测试。蝾螈提升明显，猞猁有所下降；详细结果见 [实验报告](animal_reid_v7/runs/advanced/V7实验报告.md)。

本仓库包含 V7 代码、V2/V3/V6 的本地代码依赖、最终配置、验证结果与 [提交文件](animal_reid_v7/runs/advanced/submission.csv)。V7 结合 DINOv3-L 微调与原生 SuperPoint/LightGlue 斑纹匹配；RoMa v2 经评估后未选入最终方案。实验方法及参考来源见 [V7 说明](animal_reid_v7/README.md) 和 [研究记录](animal_reid_v7/RESEARCH.md)。

## 复现限制

这份仓库是实验代码和结果归档，**克隆后不能直接一键运行**。代码依赖本地比赛图片、历史版本的预计算特征、分割掩码和评分缓存、训练得到的 V7 适配器及校准器，以及 DINOv3、LightGlue 和 RoMa 的第三方代码和权重。由于文件体积及各自的数据和权重使用条款，这些材料不包含在仓库中；需自行按原始提供方的条款取得。原始路径与环境位于 `animal_reid_v7/settings.py` 和 `run_ai.sh`，使用前须按实际环境调整。

模型选择和评估遵循身份隔离；历史 audit 已用于架构诊断，不应当作全新独立测试。公开仓库不包含比赛原始图片、身份标注、密钥或完整中间缓存。
