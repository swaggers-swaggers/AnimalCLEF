import unittest
import numpy as np
from graph import cluster,attach,rerank

class GraphChecks(unittest.TestCase):
    def test_complete_link_rejects_a_false_transitive_bridge(self):
        s=np.array([[1,.95,.05],[.95,1,.91],[.05,.91,1]],np.float32)
        labels=cluster(s,{'algorithm':'complete','threshold':.8})
        self.assertNotEqual(labels[0],labels[2])

    def test_two_real_pairs_and_one_singleton(self):
        s=np.full((5,5),.05,np.float32);np.fill_diagonal(s,1);s[0,1]=s[1,0]=.98;s[2,3]=s[3,2]=.97
        for algorithm in ['average','complete','cw','leiden']:
            labels=cluster(s,{'algorithm':algorithm,'threshold':.8})
            self.assertEqual(labels[0],labels[1]);self.assertEqual(labels[2],labels[3]);self.assertEqual(len(set(labels)),3)

    def test_reference_alone_cannot_merge_incompatible_clusters(self):
        labels=np.arange(2);sr=np.array([[.99,.99,.05],[.99,.99,.05]]);sq=np.array([[1,.01],[.01,1]])
        result=attach(labels,sr,np.array(['a','a','b']),.9,.2,sq,.8)
        self.assertNotEqual(result[0],result[1])

    def test_reference_can_join_compatible_clusters(self):
        labels=np.arange(2);sr=np.array([[.99,.99,.05],[.99,.99,.05]]);sq=np.array([[1,.95],[.95,1]])
        result=attach(labels,sr,np.array(['a','a','b']),.9,.2,sq,.8)
        self.assertEqual(result[0],result[1])

    def test_conflicting_anchors_do_not_spread(self):
        sr=np.array([[.99,.01],[.01,.99],[.01,.99]])
        result=attach(np.zeros(3,int),sr,np.array(['a','b']),.9,.2,np.eye(3),.8)
        self.assertTrue(all(x=='new_0' for x in result))

    def test_rerank_preserves_score_contract(self):
        s=np.random.default_rng(4).uniform(0,1,(20,20));s=(s+s.T)/2;np.fill_diagonal(s,1)
        out=rerank(s)
        np.testing.assert_allclose(out,out.T,atol=1e-6);np.testing.assert_allclose(out.diagonal(),1)
        self.assertTrue(np.isfinite(out).all() and out.min()>=0 and out.max()<=1)

    def test_training_and_inference_feature_columns_align(self):
        from strong import Bank,BANKS,GLOBAL_DIM
        bank=object.__new__(Bank);rng=np.random.default_rng(7)
        bank.lookup={10:0,20:1,30:2};bank.orientation=np.array([1,2,0]);bank.days=np.array([1.,0.,31.]);bank.date_known=np.array([True,False,True])
        bank.features={name:rng.normal(size=(3,2,4)).astype(np.float32) for name in BANKS}
        pairs=np.array([[10,20],[10,30]])
        x=bank.pair(pairs);matrix=bank.matrices([10],[20,30]).reshape(-1,GLOBAL_DIM)
        np.testing.assert_allclose(x,matrix,atol=1e-6)
        self.assertEqual(x.shape,(2,GLOBAL_DIM))

if __name__=='__main__':unittest.main()
