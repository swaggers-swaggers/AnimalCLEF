"""Generate a course-readable report strictly from actual saved experiment results."""
import json
from pathlib import Path
from common import RUN, ROOT, json_write

def main():
    audit=json.loads((RUN/'audit_report.json').read_text())
    tune=json.loads((RUN/'tune_report.json').read_text())
    config=json.loads((RUN/'selected_config.json').read_text())
    complete=json.loads((RUN/'completion_check.json').read_text())
    training=json.loads((RUN/'training_summary.json').read_text())
    text=['# AnimalCLEF 2026 重构实验结果','', '## 实际完成情况','',f'已完成 15,483 张图片的双视图全局特征、局部特征提取、前景过滤、{training["fit_pairs"]:,} 个图片对的相似度训练、物种级调参和独立审核。提交文件共 {complete["rows"]:,} 行，完整性检查通过。','', '全局骨干为冻结的动物重识别预训练网络；训练的是监督相似度融合模型及校准器。没有声称重新预训练基础网络。','', '## 独立审核','', '| 物种 | 未知个体 ARI | 已知/未知混合 ARI | 未知场景真实个体/预测簇 |','|---|---:|---:|---:|']
    for name in list(config)[:3]:
        a=audit[name]['metrics'];d=a['discovery'];mix=a['mixed']
        text.append(f'| {name} | {d["ari"]:.4f} | {mix["ari"]:.4f} | {d["true_identities"]}/{d["predicted_clusters"]} |')
    baseline=sum(audit[name]['metrics'][mode]['global_only_baseline_ari'] for name in list(config)[:3] for mode in ['discovery','mixed'])/6
    text+=['',f'六项本地审核 ARI 的平均为 **{audit["macro_ari"]:.4f}**；调参集同口径为 {tune["macro_ari"]:.4f}。同一审核划分、同样只用调参身份选择阈值的全局分支对照为 **{baseline:.4f}**。这是本地审核，不是 Kaggle 官方得分，也不能与旧实验中不同划分的 ARI 直接比较。','', '角蜥无训练标签，未报告其本地 ARI；测试集官方 ARI 仍未知。猞猁与蝾螈审核中仍有明显的个体拆分，调参集与审核集之间也有差距；本轮结果不能据此承诺高排名。','', '## 最终选择','', '| 物种 | 融合 | 聚类 | 相似度阈值 |','|---|---|---|---:|']
    for name,c in config.items():text.append(f'| {name} | {c["family"]} / {c.get("candidate_policy","expanded")} | {c["algorithm"]} | {c["threshold"]:.3f} |')
    text+=['','## 审核误差诊断','','| 物种 | 未知个体场景：同身份图片对的局部候选覆盖率 |','|---|---:|']
    for name in list(config)[:3]:
        recall=audit[name]['metrics']['discovery']['local_candidate_positive_pair_recall']
        text.append(f'| {name} | {recall:.1%} |')
    text+=['','蝾螈的候选覆盖不足，限制了局部匹配能够纠正的图片对；猞猁候选覆盖接近完整，但仍出现明显拆分，说明候选覆盖本身不能保证识别成功。以上仅作为冻结后的误差诊断，没有据此重新选择参数。']
    text+=['',f'![独立审核对照]({RUN / "audit_comparison.png"})','','## 测试集输出统计','','| 物种 | 图片 | 簇 | 单例簇 | 最大簇 |','|---|---:|---:|---:|---:|']
    for name,d in complete['species'].items():text.append(f'| {name} | {d["images"]} | {d["clusters"]} | {d["singletons"]} | {d["largest_cluster"]} |')
    text+=['','## 交付与复现','',f'- 提交文件：`{RUN / "submission.csv"}`',f'- 提交 SHA-256：`{complete["submission_sha256"]}`','- 配置、分数矩阵、图片对缓存、原始日志均保留在本机工作目录。','- 轻量交付包包含代码、融合模型、配置、划分和报告；不包含原始图片、基础权重、大型特征/匹配缓存及完整运行日志。','- 重跑及停止/续跑方法见项目 README.md。','- 架构来源、公开模型及赛后方案在 README.md 中列出，应在课程报告中注明。','','## 限制','', '审核身份与模型拟合和调参身份隔离，但仍来自训练数据分布，不能覆盖未知测试分布的全部变化。验证每个身份最多 12 张；角蜥使用跨物种阈值转移。公开预训练模型的数据历史由模型发布方决定。尚未上传 Kaggle，也没有获得官方评分。','']
    (RUN/'实验结果.md').write_text('\n'.join(text))
    print(RUN/'实验结果.md')

if __name__=='__main__':main()
