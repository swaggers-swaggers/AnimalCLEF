"""Verify actual completed artifacts, compare saved results, and package the run."""
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,RUN,BASE_RUN,DATA,json_write,fingerprint
from strong import freeze,BANKS,BRANCHES,SPECIES
from extract import load_features

def main():
    assert freeze()==json.loads((RUN/'audit_plan.json').read_text()),'Frozen inputs changed'
    audit=json.loads((RUN/'audit_report.json').read_text());tune=json.loads((RUN/'tune_report.json').read_text());configs=json.loads((RUN/'selected_config.json').read_text())
    old=json.loads((BASE_RUN/'audit_report.json').read_text());complete=json.loads((RUN/'completion_check.json').read_text());history=json.loads((RUN/'finetune_history.json').read_text())
    training=json.loads((RUN/'training_summary.json').read_text());ablation=json.loads((RUN/'ablation_tuning.json').read_text())
    out=pd.read_csv(RUN/'submission.csv');template=pd.read_csv(DATA/'sample_submission.csv');metadata=pd.read_csv(DATA/'metadata.csv')
    assert out.columns.tolist()==['image_id','cluster'] and len(out)==2409 and out.image_id.is_unique and out.cluster.notna().all()
    assert out.image_id.tolist()==template.image_id.tolist();joined=metadata[metadata.split=='test'].merge(out,on='image_id',validate='one_to_one')
    assert joined.groupby('cluster').dataset.nunique().max()==1
    assert all(c.startswith('cluster_'+d+'_') for d,c in zip(joined.dataset,joined.cluster))
    assert fingerprint(RUN/'submission.csv')==complete['submission_sha256']
    features={}
    for branch in ['dinov3','miew_pp','miew_arc','kaze','dino_patch','superpoint_pattern','superpoint_regions']:
        f=load_features(branch);expected=15483 if branch in ['dinov3','miew_pp','miew_arc'] else 2351
        assert len(f['ids'])==expected and len(set(f['ids'].tolist()))==expected
        nonfinite=sum(int((~torch.isfinite(v)).sum()) for k,v in f.items() if v.is_floating_point())
        assert nonfinite==0
        features[branch]={'images':expected,'nonfinite_values':nonfinite}
    json_write(features,RUN/'feature_checks.json')
    pairs=np.load(RUN/'training_v3.npz')['pairs'];split=pd.read_csv(RUN/'splits.csv')
    assert set(pairs.flatten()).issubset(set(split.loc[split.partition=='calibration','image_id']))
    assert split.groupby(['dataset','identity']).partition.nunique().max()==1
    assert json.loads((RUN/'gpu_checks.json').read_text())['passed']
    assert json.loads((RUN/'core_checks.json').read_text())['passed']
    assert json.loads((RUN/'geometry_checks.json').read_text())['passed']
    check={'passed':True,'rows':2409,'official_cluster_name_format':True,'template_order_exact':True,'no_cross_species_clusters':True,'frozen_inputs_unchanged':True,'fitting_calibration_identities_only':True,'feature_checks_passed':True,'audit_previously_seen_for_architecture_diagnosis':True,'official_ari':None}
    json_write(check,RUN/'verification.json')
    best=torch.load(RUN/'miew_arc_best.pt',weights_only=True,map_location='cpu')
    names=['Lynx','Salamander','Sea turtle'];fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,mode,title in zip(axes,['discovery','mixed'],['Unseen identities','Known + unseen identities']):
        x=np.arange(3);a=[old[s]['metrics'][mode]['ari'] for s in SPECIES[:3]];b=[audit[s]['metrics'][mode]['ari'] for s in SPECIES[:3]]
        bars=ax.bar(x-.18,a,.36,label='V2',color='#a8b3c5');ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        bars=ax.bar(x+.18,b,.36,label='V3',color='#167d8d');ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.set_xticks(x,names);ax.set_title(title);ax.set_ylim(-.05,1.08);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('ARI');axes[1].legend(frameon=False);fig.suptitle('AnimalCLEF 2026: development holdout comparison')
    fig.text(.5,.01,'V2 holdout informed architecture diagnosis. This is not a pristine audit or a Kaggle score.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(RUN/'comparison.png',dpi=180);plt.close(fig)
    text=['# AnimalCLEF 2026 强化架构实验报告','','## 实际完成','','已完成新增特征提取、度量学习微调、图片对监督融合、物种级图结构选择和全部测试匹配。',f'本轮融合训练使用 {training["pairs"]:,} 个校准图片对。ArcFace 尝试 6 个 epoch，最终选择第 **{best["epoch"]}** 轮（0 表示保留微调前权重）。','','## 相同身份划分下的留出验证','','| 物种 | v2 未知个体 | v3 未知个体 | v2 混合 | v3 混合 |','|---|---:|---:|---:|---:|']
    for s in SPECIES[:3]:text.append(f'| {s} | {old[s]["metrics"]["discovery"]["ari"]:.4f} | {audit[s]["metrics"]["discovery"]["ari"]:.4f} | {old[s]["metrics"]["mixed"]["ari"]:.4f} | {audit[s]["metrics"]["mixed"]["ari"]:.4f} |')
    text+=['',f'六项本地 ARI 均值：v2 **{old["macro_ari"]:.4f}**，v3 **{audit["macro_ari"]:.4f}**；v3 调参集均值 {tune["macro_ari"]:.4f}。角蜥无训练身份标签，不计入此均值。','', '**验证边界：**audit 身份不参与本轮拟合与参数选择，但 v2 的 audit 结果已用于本轮架构误差诊断，因此这些数值属于开发阶段留出验证，不能声称是从未查看的最终独立审核。官方测试 ARI 未知。','',f'![新旧对比]({RUN/"comparison.png"})','','## 实际选择','','| 物种 | 分数分支 | 重排序权重 | 聚类 | 阈值 |','|---|---|---:|---|---:|']
    for s,c in configs.items():text.append(f'| {s} | {c["family"]} | {c["rerank"]:.2f} | {c["algorithm"]} | {c["threshold"]:.4f} |')
    text+=['','## 调参阶段的分支对照','','以下是按相同 tune 身份搜索的 ARI + 0.05 AMI 目标值，不属于独立测试分数。所有分支共享已经计算的图片对证据；去掉 DINO 的对照保留共用候选集合。','','| 分数分支 | 猞猁 | 蝾螈 | 海龟 |','|---|---:|---:|---:|']
    for family in ablation[SPECIES[0]]:text.append('| '+family+' | '+' | '.join(f'{ablation[s][family]["objective"]:.4f}' for s in SPECIES[:3])+' |')
    text+=['','## 测试输出','','| 物种 | 图片 | 预测簇 | 单例簇 | 最大簇 |','|---|---:|---:|---:|---:|']
    for s,v in complete['species'].items():text.append(f'| {s} | {v["images"]} | {v["clusters"]} | {v["singletons"]} | {v["largest_cluster"]} |')
    text+=['',f'提交文件：`{RUN/"submission.csv"}`',f'SHA-256：`{complete["submission_sha256"]}`','','提交格式、2,409 行顺序、唯一性、物种隔离、特征完整性和配置冻结检查通过。尚未上传 Kaggle。','','## 来源与复现','','架构、参考前列方案、公开权重版本、停止和续跑方法见项目 README.md。没有使用其他团队的成品提交或测试身份标注；测试总体统计不用于指定目标簇数。微调与校准器仅拟合 calibration 身份。','','归档包含源码、结果、融合模型、选中的微调检查点与许可证资料。基础权重、图片、完整特征/匹配缓存留在本机工作目录。','']
    (RUN/'强化实验报告.md').write_text('\n'.join(text))
    files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.py','.md','.sh','.lock']]
    files += [p for p in RUN.iterdir() if p.is_file() and p.name!='bundle_manifest.json' and p.suffix in ['.json','.csv','.pkl','.png','.md']]
    files += [RUN/'miew_arc_best.pt']
    files += [p for p in (ROOT/'models/dinov3').iterdir() if p.suffix in ['.md','.json']]
    oldroot=ROOT.parent/'animal_reid_v2';files += [p for p in oldroot.iterdir() if p.is_file() and p.suffix in ['.py','.md','.sh','.lock']]
    files += [BASE_RUN/n for n in ['selected_config.json','audit_report.json','tune_report.json','splits.csv','training_plan.npz','fusion_models.pkl','completion_check.json']]
    for name in ['lightglue','sam2']:
        files += [p for p in (ROOT.parent/'references'/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.yaml','.yml','.toml'] or p.name.startswith('LICENSE'))]
    files=sorted(set(files));manifest={str(p.relative_to(ROOT.parent)):fingerprint(p) for p in files};json_write(manifest,RUN/'bundle_manifest.json')
    archive=ROOT.parent/'AnimalCLEF2026_强化架构交付包.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files+[RUN/'bundle_manifest.json']:z.write(p,str(p.relative_to(ROOT.parent)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(json.dumps({'complete':True,'verification':check,'archive':str(archive),'local_ari_v2':old['macro_ari'],'local_ari_v3':audit['macro_ari']},ensure_ascii=False))

if __name__=='__main__':main()
