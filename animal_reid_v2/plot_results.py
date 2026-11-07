"""Static scientific comparison, from the saved independent audit only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUN

def main():
    audit=json.loads((RUN/'audit_report.json').read_text())
    species=['LynxID2025','SalamanderID2025','SeaTurtleID2022']
    names=['Lynx','Salamander','Sea turtle']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,mode,title in zip(axes,['discovery','mixed'],['Unseen identities','Known + unseen identities']):
        baseline=[audit[s]['metrics'][mode]['global_only_baseline_ari'] for s in species]
        full=[audit[s]['metrics'][mode]['ari'] for s in species]
        x=np.arange(3);w=.35
        a=ax.bar(x-w/2,baseline,w,color='#a8b3c5',label='Global-only baseline')
        b=ax.bar(x+w/2,full,w,color='#167d8d',label='Selected pipeline')
        ax.bar_label(a,fmt='%.3f',padding=3,fontsize=9);ax.bar_label(b,fmt='%.3f',padding=3,fontsize=9)
        ax.set_xticks(x,names);ax.set_title(title,fontsize=12);ax.set_ylim(-.05,1.08);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Adjusted Rand Index (ARI)');axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    fig.suptitle('AnimalCLEF 2026 — independent identity-disjoint local audit',fontsize=13)
    fig.text(.5,.01,'Three labeled species only. This is not a Kaggle leaderboard score.',ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(RUN/'audit_comparison.png',dpi=180);plt.close(fig)
    print(RUN/'audit_comparison.png')

if __name__=='__main__':main()
