"""Describe fixed results, pair errors, ablations, and identity-subset sensitivity."""
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from experiment import ROOT, CV, RUN, V3, SPECIES, write, digest, verify_frozen


def pair_counts(frame):
    a, b = np.triu_indices(len(frame), 1)
    truth = frame.identity.to_numpy()[a] == frame.identity.to_numpy()[b]
    pred = frame.cluster.to_numpy()[a] == frame.cluster.to_numpy()[b]
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
    return {'correct_pairs': tp, 'false_merges': fp, 'missed_pairs': fn,
            'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1)}


def main():
    verify_frozen()
    report = json.loads((RUN / 'audit_report.json').read_text())
    tune = json.loads((RUN / 'tune_search.json').read_text())
    verification = json.loads((RUN / 'verification.json').read_text())
    pred = pd.read_csv(RUN / 'audit_predictions.csv')
    pair_report = {}; sensitivity = {}
    for name in SPECIES:
        pair_report[name] = {}
        for mode in ['discovery', 'mixed']:
            pair_report[name][mode] = {version: pair_counts(pred[(pred.dataset == name) & (pred['mode'] == mode) & (pred.version == version)])
                                       for version in ['baseline', 'selected']}
        delta = []
        for seed in range(200):
            keys = pred.loc[pred.dataset == name, 'identity'].unique()
            chosen = np.random.default_rng(81106 + seed).choice(keys, max(2, int(.8 * len(keys))), replace=False)
            values = {}
            for version in ['baseline', 'selected']:
                values[version] = []
                for mode in ['discovery', 'mixed']:
                    p = pred[(pred.dataset == name) & (pred['mode'] == mode) & (pred.version == version) & pred.identity.isin(chosen)]
                    values[version].append(adjusted_rand_score(p.identity, p.cluster))
            delta.append(float(np.mean(values['selected']) - np.mean(values['baseline'])))
        sensitivity[name] = {'method': '200 identity subsets without replacement, 80% identities; fixed predictions, no re-selection; not a confidence interval',
                             'mean_delta': float(np.mean(delta)), 'positive_fraction': float(np.mean(np.array(delta) > 0)),
                             'q05_delta': float(np.quantile(delta, .05)), 'q95_delta': float(np.quantile(delta, .95))}
    write(pair_report, RUN / 'pair_errors.json'); write(sensitivity, RUN / 'sensitivity.json')
    names = dict(zip(SPECIES, ['猞猁', '蝾螈', '海龟']))
    before, after = report['baseline_macro_ari'], report['selected_macro_ari']
    lines = ['# AnimalCLEF 2026 V6 实验报告', '',
             f'固定方案的六项开发留出平均 ARI：**{before:.6f} → {after:.6f}**，绝对增加 **{after-before:.6f}**，相对增加 **{(after/before-1)*100:.2f}%**。', '',
             '本轮已实际完成 RTX 5060 Ti GPU 训练、对照、身份子集选择和测试推理。最终收益来自蝾螈的原有四种特征融合及聚类改进；新增训练分支没有通过稳定性选择，没有纳入最终提交。', '',
             '| 物种 | 情形 | V3 ARI | V6 ARI | 差值 |', '|---|---|---:|---:|---:|']
    for name in SPECIES:
        for mode in ['discovery', 'mixed']:
            row = report[name][mode]; a = row['baseline']['ari']; b = row['selected']['ari']
            lines.append(f'| {names[name]} | {mode} | {a:.4f} | {b:.4f} | {b-a:+.4f} |')
    lines += ['', '## 训练与选择', '', '| 物种 | 训练前检索 mAP | 最佳检索 mAP | 选中轮次 | 最终 tune ARI：V3 → V6 |', '|---|---:|---:|---:|---|']
    for name in SPECIES:
        history = json.loads((RUN / f'training_{name}.json').read_text())
        candidates = [r for r in history['history'] if 'tune_retrieval_map' in r]
        best = max(candidates, key=lambda r: r['tune_retrieval_map'])
        lines.append(f'| {names[name]} | {candidates[0]["tune_retrieval_map"]:.4f} | {best["tune_retrieval_map"]:.4f} | {best["epoch"]} | {tune[name]["baseline"]["mean"]:.4f} → {tune[name]["selected"]["mean"]:.4f} |')
    lines += ['', '检索 mAP 与最终聚类 ARI 是不同指标。上表的检索初值是四特征随机投影适配器，不是 V3 全部局部/全局融合系统。共训练 3 × 16 × 50 = 2,400 步；所有候选、未选分支和四个 tune 身份子集结果保存在 `tune_search.json`。', '',
              '最终蝾螈配置：90% V3 分数 + 10% 未经本轮训练的四特征融合校准分数；互为近邻 k=10、Leiden resolution=0.02、threshold=0.7166667。其他物种保留 V3。未根据本轮 audit 再选择。', '',
              '## 误合并与漏配代价', '', '| 蝾螈情形 | 版本 | 正确匹配对 | 误合并对 | 漏配对 | 精确率 | 召回率 | 簇数 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in ['discovery', 'mixed']:
        for version, label in [('baseline', 'V3'), ('selected', 'V6')]:
            p = pair_report['SalamanderID2025'][mode][version]; n = report['SalamanderID2025'][mode][version]['predicted_clusters']
            lines.append(f'| {mode} | {label} | {p["correct_pairs"]} | {p["false_merges"]} | {p["missed_pairs"]} | {p["precision"]:.3f} | {p["recall"]:.3f} | {n} |')
    s = sensitivity['SalamanderID2025']
    lines += ['', 'V6 更保守，减少部分错误合并，同时增加漏配/碎片化。不能把簇数更多描述为身份识别全面改善。', '',
              f'对固定留出预测做 200 次 80% 身份子集敏感性检查，蝾螈两情形平均 ARI 改善比例为 {s["positive_fraction"]:.1%}；差值的 5%–95% 分位区间为 [{s["q05_delta"]:+.4f}, {s["q95_delta"]:+.4f}]。这不是置信区间，也没有用于重新选模型。', '',
              '## 交付与验证', '', f'- 新提交：`submission.csv`，{verification["rows"]:,} 行。', f'- SHA-256：`{verification["submission_sha256"]}`。',
              '- V3 调参、开发留出及测试聚类重放通过；冻结配置与所有上游输入哈希未变。',
              '- 模板顺序、图片唯一性、非空预测、物种隔离和 cluster 前缀检查通过。',
              '- 独立 CPU 推理从原始特征缓存重建所选方案，逐字节复现提交。',
              '- 神经网络与聚类行为测试 7 项通过。', '',
              '测试蝾螈从 527 簇变为 606 簇，不能从没有真值的测试簇数判断成绩。旧 V3 CSV 与既有压缩包保持原样。', '',
              '## 结果边界', '',
              '历史 audit 已被以前实验用于诊断，本轮沿用同一身份划分；这些是开发留出结果，不是全新独立测试。六项平均只覆盖三个有标签物种，角蜥没有本地身份真值。新版本尚未提交 Kaggle，不能声称官方 Public/Private 分数提高。', '',
              '完整方法、身份划分、冻结机制和运行步骤见项目 README。']
    (RUN / 'V6实验报告.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, name, title in zip(axes, SPECIES, ['Lynx', 'Salamander', 'Sea turtle']):
        x = np.arange(2)
        a = [report[name][mode]['baseline']['ari'] for mode in ['discovery', 'mixed']]
        b = [report[name][mode]['selected']['ari'] for mode in ['discovery', 'mixed']]
        ax.bar(x - .18, a, .35, label='V3', color='#8294a8')
        bars = ax.bar(x + .18, b, .35, label='V6', color='#138d87')
        ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)
        ax.set_xticks(x, ['Discovery', 'Mixed']); ax.set_ylim(0, 1.05); ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel('Adjusted Rand index'); axes[-1].legend(loc='upper left')
    fig.suptitle(f'Development holdout: macro ARI {before:.4f} → {after:.4f}\nNot an official Kaggle score', fontsize=12)
    fig.savefig(RUN / 'comparison.png', dpi=180); plt.close(fig)
    files = list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh')) + [ROOT / 'README.md']
    files += [p for p in RUN.iterdir() if p.suffix in ['.json', '.csv', '.md', '.png', '.pkl', '.log']]
    files += list(RUN.glob('adapter_*.pt'))
    archive = CV / 'AnimalCLEF2026_V6改进交付包.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files): z.write(p, str(p.relative_to(CV)))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    write({'archive': str(archive), 'sha256': digest(archive), 'files': len(files),
           'size_bytes': archive.stat().st_size}, RUN / 'bundle_manifest.json')
    print(json.dumps({'before': before, 'after': after, 'archive': str(archive), 'sensitivity': s}, ensure_ascii=False))


if __name__ == '__main__':
    main()
