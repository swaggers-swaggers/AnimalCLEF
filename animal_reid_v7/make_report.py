"""Build a report and a compact handoff from completed, verified results."""
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings import *


def main():
    report=json.loads((RUN/'audit_report.json').read_text())
    verification=json.loads((RUN/'verification.json').read_text()); assert verification['passed']
    assert verification['submission_sha256']==sha(RUN/'submission.csv')
    tune=json.loads((RUN/'tune_search.json').read_text())
    config=json.loads((RUN/'selected_config.json').read_text())
    history=json.loads((RUN/'dino_history.json').read_text())
    baseline=json.loads((V6/'audit_report.json').read_text())
    labels={'LynxID2025':'猞猁','SalamanderID2025':'蝾螈','SeaTurtleID2022':'海龟'}
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in SPECIES:
        points=[(x['epoch'],x['retrieval'][name]) for x in history['history'] if 'retrieval' in x]
        axes[0].plot(*np.array(points).T,label=name,marker='.',lw=1.5)
    axes[0].set(xlabel='Epoch',ylabel='Tune retrieval mAP',title='New image backbone: fixed retrieval sample')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.2)
    ticks=[]; old=[]; new=[]
    for name in SPECIES:
        for mode in ['discovery','mixed']:
            ticks.append(name.split('ID')[0]+'\n'+mode)
            old.append(report[name][mode]['baseline']['ari']); new.append(report[name][mode]['selected']['ari'])
    x=np.arange(len(ticks)); axes[1].bar(x-.18,old,width=.36,label='V6'); axes[1].bar(x+.18,new,width=.36,label='V7')
    axes[1].set(xticks=x,xticklabels=ticks,ylabel='ARI',ylim=(0,1),title='Historically seen development holdout')
    axes[1].tick_params(axis='x',labelsize=7); axes[1].legend(); axes[1].grid(axis='y',alpha=.2)
    fig.savefig(RUN/'results.png',dpi=180); plt.close(fig)
    delta=report['delta']; relative=delta/report['baseline_macro_ari']*100
    lines=['# V7 图像级与对应匹配实验报告','',
           f"开发留出六项平均 ARI：V3 **{baseline['baseline_macro_ari']:.6f}** → V6 **{report['baseline_macro_ari']:.6f}** → V7 **{report['selected_macro_ari']:.6f}**。V7 比 V6 变化 **{delta:+.6f}**（ARI ×100 后 {delta*100:+.3f} 个点，相对 {relative:+.2f}%）。",'',
           '这是历史上已查看的开发留出集结果，不是全新独立测试，也不是 Kaggle 分数。新方案尚无官方成绩。没有用 audit 结果重新选择本轮配置。','',
           '| 物种 | 模式 | V6 ARI | V7 ARI | 变化 |','|---|---|---:|---:|---:|']
    for name in SPECIES:
        for mode in ['discovery','mixed']:
            r=report[name][mode]; a=r['baseline']['ari']; b=r['selected']['ari']
            lines.append(f'| {labels[name]} | {mode} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines += ['','![训练曲线与开发留出对照](results.png)','','## 最终采用的分支','','| 物种 | 固定配置 | tune 平均 ARI：V6 → V7 |','|---|---|---:|']
    for name in SPECIES:
        r=tune[name]; c=config[name]
        description='保持 V6' if c.get('kind')=='baseline' else f"{c['source']}，新分数权重 {c['weight']:.2f}，{c['algorithm']}，阈值 {c['threshold']:.3f}"
        lines.append(f"| {labels[name]} | {description} | {r['baseline']['mean']:.6f} → {r['selected']['mean']:.6f} |")
    lines += ['','### 蝾螈主要分支对照（同一 tune 搜索）','','| 分支 | 最佳 tune 平均 ARI |','|---|---:|']
    branches={}
    for trial in tune['SalamanderID2025']['trials']:
        source=trial['config']['source']; branches[source]=max(branches.get(source,-1),trial['mean'])
    for source,label in [('trained','训练后图像表示（校准）'),('initial_raw','未训练图像表示（余弦）'),
                         ('spot_1','局部斑纹对应'),('roma','RoMa 稠密对应'),('spot_roma','斑纹与 RoMa 组合')]:
        lines.append(f'| {label} | {branches[source]:.6f} |')
    lines += ['','这是调参候选的最佳值，不是额外独立测试。RoMa 已完成计划的校准与调参匹配，未因架构更新而强行加入最终方案。']
    lines += ['','各分支名称：`trained` 是按身份训练的 DINOv3-L 联合表示并校准；`initial` 是同输入和池化的未训练对照；`_raw` 使用原始余弦相似度；`spot_*` 是局部高置信度对应或几何内点；`roma*` 是候选上的双向稠密匹配；`base` 是 V6 分数。训练过或使用更新模型并不保证进入最终方案。','',
              '## 新图像模型训练证据','','| 物种 | 同一采样检索集：初始 mAP | 最佳 mAP |','|---|---:|---:|']
    initial=history['history'][0]['retrieval']
    for name in SPECIES: lines.append(f"| {labels[name]} | {initial[name]:.6f} | {history['best'][name]:.6f} |")
    lines += ['','以上 mAP 只比较同一新骨干的训练前后，不能与整套 V3/V6 的 ARI 直接比较。24 个 epoch、1,920 次优化，2,131,969 个可训练参数，实际使用 RTX 5060 Ti。训练集和分数校准集均与 tune/audit 按身份隔离。','',
              '## 误合并与漏匹配','','| 物种/模式 | V6→V7 成对精确率 | V6→V7 成对召回率 | 错误合并对数 V6→V7 |','|---|---:|---:|---:|']
    for name in SPECIES:
        for mode in ['discovery','mixed']:
            a=report[name][mode]['baseline']; b=report[name][mode]['selected']
            lines.append(f"| {labels[name]}/{mode} | {a['pair_precision']:.4f} → {b['pair_precision']:.4f} | {a['pair_recall']:.4f} → {b['pair_recall']:.4f} | {a['false_merge_pairs']} → {b['false_merge_pairs']} |")
    lines += ['','### 不同已知视角的同身份图片对','','| 物种/模式 | 同身份跨视角对数 | V6→V7 召回率 |','|---|---:|---:|']
    for name in SPECIES:
        for mode in ['discovery','mixed']:
            a=report[name][mode]['baseline']; b=report[name][mode]['selected']
            count=a['cross_orientation_positive_pairs']
            value=f"{a['cross_orientation_pair_recall']:.4f} → {b['cross_orientation_pair_recall']:.4f}" if count else '无可用图片对'
            lines.append(f'| {labels[name]}/{mode} | {count} | {value} |')
    lines += ['','## 验证与局限','',
              '- 已独立重放 V6 开发留出和测试分组，重放 V7 tune 结果；锁定输入的哈希与匹配证据保持一致。',
              '- 局部匹配使用真实特征点，完整精度的批量/逐对检查通过。早期半精度匹配因一致性问题被隔离，未用于最终匹配分数。RoMa 的训练侧候选名单保留了初始筛选，以避免中途改变样本。',
              '- 提交文件包含 2,409 行，顺序、唯一性、完整性、物种隔离均通过检查；未替换分支和蜥蜴的行保持 V6 原样。',
              '- 六次身份子集检验都是 tune 上的稳定性检查，不是六个独立外部数据集。反复使用历史开发集可能高估泛化提升。',
              '- RoMa 是稀疏候选验证；局部参考图库只比较基线前 32 张。候选外的配对仍可能遗漏。公开预训练权重的来源和数据覆盖遵循原模型说明，本实验没有证明其训练数据与比赛完全无重合。','',
              '## 交付','','- `submission.csv`：本轮锁定方案的正式格式文件，尚未提交 Kaggle。',
              '- `audit_report.json`、`audit_predictions.csv`：开发指标和逐图预测。',
              '- `selected_config.json`、`frozen.json`、`verification.json`：选择、输入和核验记录。',
              '- `tune_search.json`、训练日志、checkpoint、校准器和代码：实验依据。',
              '- 完整数据、原始大模型权重、全部特征与匹配缓存保留在工作区，精简交付包不重复装入这些大文件。','',
              f"提交 SHA-256：`{verification['submission_sha256']}`。",'',
              '公开方法来源与本轮独立改动见交付包根目录 `RESEARCH.md`；完整流程见 `README.md`。']
    (RUN/'V7实验报告.md').write_text('\n'.join(lines)+'\n')
    package=CV/'AnimalCLEF2026_V7架构改进交付包.zip'
    files=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[ROOT/'run_ai.sh']
    names=['submission.csv','V7实验报告.md','results.png','audit_report.json','audit_predictions.csv','verification.json',
           'frozen.json','selected_config.json','score_calibrators.pkl','score_calibration.json','tune_plan.json','tune_search.json',
           'roles.csv','dino_plan.json','dino_history.json','dino_smoke.json','native_checks.json','core_checks.log','input_provenance.json','matching_checks.json',
           'roma_pilot_report.json','roma_calibration_complete.json','roma_tune_complete.json','spot_pair_plan.json','preprocessing_review.jpg']
    files += [RUN/n for n in names] + list(RUN.glob('dino_best_*.pt')) + list(RUN.glob('frozen_*.npz'))
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            assert path.exists(),path
            archive.write(path,Path('animal_reid_v7')/path.relative_to(ROOT))
    with zipfile.ZipFile(package) as archive: assert archive.testzip() is None
    write({'path':str(package),'sha256':sha(package),'bytes':package.stat().st_size,'zip_test_passed':True},RUN/'package.json')
    log(f'Report and package ready: {package}')


if __name__=='__main__': main()
