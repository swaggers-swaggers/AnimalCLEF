# AnimalCLEF 2026 V7：图像级适配与对应匹配

V6 只在现有描述子和图聚类上取得小幅提升。本轮直接训练 DINOv3-L 的后 8 个块，联合 CLS 与空间 GeM 表示；另行评估原生 SuperPoint/LightGlue 斑纹对应及 RoMa v2 稠密对应。各分支是否进入提交，由固定调参流程决定，不以架构新旧决定。

## 实验约束

- 原 V3/V6 训练权重、分数缓存、身份划分和提交保持不变；新增实验位于本目录。打包入口修复记录见下文。
- 使用 V3 固定的身份隔离拆分。calibration 内再按身份分为 80% 图像训练、20% 分数校准；tune 负责选 checkpoint、融合和聚类。没有使用测试标签、测试伪标签或测试目标簇数。
- audit 是历史上已查看的开发留出集，不能宣称为全新独立测试。本轮配置在读取新分支的 audit 评估结果前锁定。
- 六项 ARI 的等权平均是本地开发指标，不是 Kaggle 分数或普通分类准确率。正式成绩只能由官方评测获得。
- 参考公开方法的思路，自行组合与训练；不使用参赛团队微调权重或成品提交。通用预训练模型、LightGlue 与 RoMa 的原始实现按各自许可证使用，详见 [研究记录](RESEARCH.md)。

## 训练与选择

DINOv3-L：384 像素、原生比例主体裁剪，最后 8 块的 qkv/proj/fc1/fc2 使用 rank-16 低秩更新，并训练末段 LayerNorm 与空间池化参数；24 × 80 个训练步，batch 16，每批 4 个身份各 4 图。CosFace 与同物种跨批次对比记忆联合训练，物种 checkpoint 由 tune 检索 mAP 选择。原始和训练后表示都作为候选，避免把训练本身误当收益。

蝾螈斑纹：主体坐标内的颜色区域、PCA 方向对齐、保持比例的 1024 像素画布；每图最多 512 个真实局部点，不添加虚拟点；比较两种相对方向。LightGlue 使用 float32 计算。早期 float16 结果因批量一致性检查失败，保留在 `spots_fp16_diagnostic.sqlite` 仅作诊断，最终流程不读取它。存储的图像特征仍为半精度，所有推理使用相同特征值转换为 float32；变化仅在匹配运算。

RoMa v2：固定官方 v2.0.1 权重，fast 双向推理。先用 calibration pilot 确认分辨能力，再对蝾螈候选对提取 12 项稠密对应特征；候选由基线、斑纹、新图像表示各自的近邻联合产生。未计算的对应保持基线分数，不视为负例。calibration 对的初始筛选曾用早期半精度斑纹结果；该训练侧名单随后锁定，稠密特征独立重新计算，未用于评估标签筛选。

融合候选在 tune 比较 average/complete/Leiden，另对 6 个 75% 身份子集同时重建候选和基线聚类。需要全量平均至少提升 0.005、每种模式下降不超过 0.02、至少赢 4/6 个子集，才有资格替换基线。大量调参和同一开发集的重复使用仍可能带来选择偏差。

## 运行

环境为 `/home/liu/miniconda3/envs/ai`，使用本工作区既有 V3/V4/V6 数据、模型和参考实现；`run_ai.sh` 设置依赖路径。显卡训练需在可以访问 CUDA 的执行环境中运行。

```bash
bash animal_reid_v7/run_ai.sh dino_train.py --stage train
bash animal_reid_v7/run_ai.sh dino_train.py --stage infer
bash animal_reid_v7/run_ai.sh patterns.py --stage extract
bash animal_reid_v7/run_ai.sh patterns.py --stage tune
bash animal_reid_v7/run_ai.sh roma_stage.py --stage calibration
bash animal_reid_v7/run_ai.sh roma_stage.py --stage tune
bash animal_reid_v7/run_ai.sh fusion.py --stage tune
```

`fusion.py` 会锁定配置、模型及校准/调参对应证据。之后只补齐选中分支需要的 audit/test 推理：若选中斑纹则运行 `patterns.py --stage complete`；若选中 RoMa，还需该斑纹结果用于候选筛选，然后运行 `roma_stage.py --stage audit` 和 `--stage test`。最后运行 `deliver.py` 与 `make_report.py`。锁定后禁止再次调参；需要新实验时应使用新的运行目录。

`runs/advanced/STOP` 存在时，在计算批次边界安全停止；移除后可从已提交的对应对缓存恢复。图像特征文件与 SQLite 匹配文件分开保存。所有匹配必须完成后才做统计，不能用部分缓存宣称最终结果。

最终报告、提交文件及核验信息由实际完成的 `deliver.py` 结果生成。交付压缩包不包含原始图片、官方大模型权重和完整匹配缓存；它依赖本工作区上述输入，不是脱离数据即可一键运行的独立包。

## 打包入口修复记录

首次自动交付时，V3 依赖修改了 Python 搜索路径，导致同名 `deliver` 模块被误导入。旧版检查报告、图表、清单及 `AnimalCLEF2026_强化架构交付包.zip` 被重新生成，V7 评估当时未执行。现已改为以绝对文件路径启动 V7 交付程序，重新运行成功。旧版权重、分数缓存和提交没有被此入口改写，V7 冻结输入与基线重放检查均通过；没有据此声称旧版派生归档的字节完全不变。旧版官方分数记录仍保存在原 `kaggle_score_record.json` 中。
