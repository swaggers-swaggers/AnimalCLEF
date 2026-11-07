"""Checks for the high-impact clustering and leakage regressions."""
import unittest
import numpy as np
from pipeline import cluster, attach, select_pairs, metrics, score_matrix, root_candidates

class ClusteringTests(unittest.TestCase):
    def test_two_identities_and_isolated_individual(self):
        s=np.full((6,6),.04,dtype=np.float32)
        s[:3,:3]=.95;s[3:5,3:5]=.9;np.fill_diagonal(s,1)
        for algorithm in ('average','cw'):
            p=cluster(s,{'algorithm':algorithm,'threshold':.6})
            self.assertEqual(metrics([0,0,0,1,1,2],p)['ari'],1.)

    def test_conflicting_anchors_do_not_merge_identities(self):
        labels=np.array([0,0]);s=np.array([[.99,.01],[.01,.99]])
        p=attach(labels,s,np.array(['a','b']),.8,.2)
        self.assertEqual(list(p),['new_0','new_0'])

    def test_same_known_identity_can_connect_separated_query_clusters(self):
        p=attach(np.array([0,1,2]),np.array([[.99,.01],[.95,.02],[.1,.1]]),np.array(['a','b']),.8,.2)
        self.assertEqual(p[0],p[1]);self.assertNotEqual(p[0],p[2])

    def test_candidate_selection_is_symmetric_union_of_backbones(self):
        g=np.zeros((4,4,6),np.float32)
        g[0,1,1]=.9;g[0,2,4]=.95
        a,b,p=select_pairs(g,np.arange(4),np.arange(4),1,True)
        self.assertIn((0,1),list(map(tuple,p)));self.assertIn((0,2),list(map(tuple,p)))
        self.assertTrue(np.all(a<b))

    def test_cluster_names_do_not_affect_ari(self):
        self.assertEqual(metrics(['x','x','y'],['arbitrary','arbitrary','other'])['ari'],1.)

    def test_uncomputed_local_pairs_retain_global_evidence(self):
        class Model:
            def __init__(self,p):self.p=p
            def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
        class Matcher:
            def compute(self,pairs,branch):return np.zeros((len(pairs),9))
        rng=np.random.default_rng(3);g=rng.uniform(0,1,(8,8,6)).astype(np.float32)
        for c in range(6):g[:,:,c]=(g[:,:,c]+g[:,:,c].T)*.5
        a,b,_=select_pairs(g,np.arange(8),np.arange(8),1,True)
        score=score_matrix(g,np.arange(8),np.arange(8),{'global':Model(.8),'local':Model(.1),'branches':['aliked','disk']},Matcher(),1,True,'learned')
        computed=np.eye(8,dtype=bool);computed[a,b]=True;computed[b,a]=True
        self.assertTrue(np.any(~computed))
        np.testing.assert_allclose(score[~computed],.8)
        np.testing.assert_allclose(score[a,b],.1)

    def test_local_retrieval_recovers_pair_missing_from_global_shortlist(self):
        g=np.zeros((6,6,6),np.float32)
        for i in range(6):
            g[i,(i+1)%6,1]=.9;g[i,(i+1)%6,4]=.9
        _,_,global_pairs=select_pairs(g,np.arange(6),np.arange(6),1,True)
        self.assertNotIn((0,3),list(map(tuple,global_pairs)))
        evidence=np.zeros((6,6,9),np.float32);evidence[0,3,0]=evidence[3,0,0]=np.log1p(20)
        _,_,expanded=root_candidates(g,np.arange(6),evidence,1)
        self.assertIn((0,3),list(map(tuple,expanded)))

if __name__=='__main__':unittest.main()
