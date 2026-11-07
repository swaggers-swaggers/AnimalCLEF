import unittest
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from dino_model import LowRank,queue_contrastive
from settings import roles,metadata
from roma_stage import nominate


class Checks(unittest.TestCase):
    def test_low_rank_initialization_and_real_backbone_gradient(self):
        layer=nn.Linear(12,16); layer.requires_grad_(False); x=torch.randn(3,12)
        old=layer(x).detach(); parametrize.register_parametrization(layer,'weight',LowRank(layer.weight.shape,rank=4))
        torch.testing.assert_close(layer(x),old)
        layer(x).square().mean().backward()
        self.assertIsNone(layer.parametrizations.weight.original.grad)
        self.assertGreater(float(layer.parametrizations.weight[0].b.grad.abs().sum()),0)

    def test_contrastive_queue_masks_self_but_retains_real_positives(self):
        z=torch.nn.functional.normalize(torch.randn(8,16),dim=1).requires_grad_(); y=torch.arange(4).repeat_interleave(2)
        loss=queue_contrastive(z,y,torch.empty(0,16),torch.empty(0,dtype=torch.long));loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(torch.isfinite(z.grad).all())

    def test_candidate_selection_excludes_self_and_is_canonical(self):
        s=np.eye(3); s[0,1]=s[1,0]=.9; s[1,2]=s[2,1]=.8
        p=nominate(s,np.array([9,2,5]),np.array([9,2,5]),1,True)
        self.assertTrue((p[:,0]<p[:,1]).all()); self.assertEqual(len(p),len(np.unique(p,axis=0)))
        self.assertTrue(any(np.array_equal(x,[2,9]) for x in p))

    def test_new_calibration_has_no_tune_audit_or_test_identity(self):
        r=roles();m,s=metadata(); self.assertEqual(r.groupby('identity').role.nunique().max(),1)
        self.assertFalse(set(r.identity)&set(s.loc[s.partition!='calibration','identity']))
        self.assertFalse(set(r.image_id)&set(m.loc[m.split=='test','image_id']))


if __name__=='__main__': unittest.main()
