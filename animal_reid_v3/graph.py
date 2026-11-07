"""Reciprocal neighborhoods, graph communities, and guarded identity anchors."""
import random
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pipeline import chinese_whispers,attach as original_attach

def rerank(score,k=12,blend=.3):
    n=len(score)
    if n<3:return score.copy()
    k=min(k,n-1);rank=np.argsort(-score,axis=1)[:,:k+1]
    knn=np.zeros((n,n),bool);knn[np.arange(n)[:,None],rank]=True
    reciprocal=knn&knn.T
    affinity=np.exp(-(1-score)*4)*reciprocal
    affinity/=np.maximum(affinity.sum(1,keepdims=True),1e-8)
    expansion=np.stack([affinity[r[:min(3,len(r))]].mean(0) for r in rank])
    # Weighted Jaccard, computed in blocks to avoid a cubic memory allocation.
    result=np.zeros_like(score)
    for i in range(n):
        common=np.minimum(expansion[i],expansion).sum(1)
        result[i]=common/np.maximum(2-common,1e-8)
    result=blend*result+(1-blend)*score
    result=(result+result.T)/2;np.fill_diagonal(result,1.)
    return result.astype(np.float32)

def tree(score,method):
    if len(score)<2:return None
    d=np.maximum(0,1-(score+score.T)/2);np.fill_diagonal(d,0)
    return linkage(squareform(d,checks=False),method=method)

def cluster(score,cfg,precomputed_tree=None):
    n=len(score)
    if n<2:return np.arange(n)
    algorithm=cfg['algorithm'];threshold=cfg['threshold']
    if algorithm in ('average','complete'):
        z=precomputed_tree if precomputed_tree is not None else tree(score,algorithm)
        return fcluster(z,t=max(1-threshold,1e-8),criterion='distance')-1
    if algorithm=='cw':return chinese_whispers(score,threshold,topk=20)
    import igraph as ig
    ig.set_random_number_generator(random.Random(20261104))
    values=score.copy();np.fill_diagonal(values,-1)
    k=min(20,n-1);neighbors=np.argpartition(-values,k-1,axis=1)[:,:k]
    keep=np.zeros((n,n),bool);keep[np.arange(n)[:,None],neighbors]=True
    keep=(keep&keep.T)&(values>=threshold)
    a,b=np.where(np.triu(keep,1));weights=(values[a,b]-threshold+.02)/(1-threshold+.02)
    graph=ig.Graph(n=n,edges=list(zip(a.tolist(),b.tolist())),directed=False)
    if not len(a):return np.arange(n)
    return np.asarray(graph.community_leiden(objective_function='CPM',weights=weights.tolist(),resolution=.08,n_iterations=-1).membership)

def attach(labels,sr,identities,threshold,margin,sq,cluster_threshold,mode='safe'):
    if mode=='original':return original_attach(labels,sr,identities,threshold,margin)
    result=np.array([f'new_{x}' for x in labels],object)
    if not sr.shape[1] or threshold>1:return result
    names=np.unique(identities)
    # One spurious reference image cannot alone dominate a larger reference identity.
    scores=np.stack([np.sort(sr[:,identities==key],axis=1)[:,-min(2,int((identities==key).sum())):].mean(1) for key in names],1)
    order=np.argsort(-scores,axis=1);first=scores[np.arange(len(labels)),order[:,0]]
    second=scores[np.arange(len(labels)),order[:,1]] if len(names)>1 else np.zeros(len(labels))
    confident=(first>=threshold)&(first-second>=margin)
    groups={}
    for label in np.unique(labels):
        members=np.flatnonzero(labels==label);voters=members[confident[members]]
        if not len(voters):continue
        votes=np.bincount(order[voters,0],weights=first[voters],minlength=len(names));winner=int(votes.argmax())
        if votes[winner]<.9*votes.sum():continue
        support=voters[order[voters,0]==winner]
        if len(members)>2 and len(support)<2:continue
        groups.setdefault(winner,[]).append(members)
    for winner,components in groups.items():
        accepted=[]
        for members in sorted(components,key=lambda x:-len(x)):
            # Shared references nominate a merge; query evidence verifies the bridge.
            for j,other in enumerate(accepted):
                cross=sq[np.ix_(members,other)]
                if cross.max()>=cluster_threshold and np.quantile(cross,.5)>=max(.15,cluster_threshold-.25):
                    accepted[j]=np.r_[other,members];break
            else:accepted.append(members)
        for j,members in enumerate(accepted):result[members]=f'known_{names[winner]}_{j}'
    return result
