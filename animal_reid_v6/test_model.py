import unittest
import numpy as np
import torch
from torch.nn import functional as F
from model import MetricAdapter, supervised_contrastive, normalized_mean
from experiment import graph_cluster, subset_cases, metadata, initialize, RUN


class ModelChecks(unittest.TestCase):
    def test_initialization_preserves_projected_teacher(self):
        model = MetricAdapter(32, 16).eval()
        x = F.normalize(torch.randn(8, 32), dim=1)
        torch.testing.assert_close(model(x), F.normalize(x @ model.projection, dim=1))

    def test_real_positive_pairs_train_and_singletons_are_safe(self):
        model = MetricAdapter(32, 16)
        x = F.normalize(torch.randn(8, 32), dim=1)
        loss = supervised_contrastive(model(x), torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.residual[-1].weight.grad.abs().sum()), 0)
        self.assertEqual(float(supervised_contrastive(model(x), torch.arange(8)).detach()), 0.)

    def test_mean_view_normalization(self):
        z = normalized_mean(np.array([[[1., 0.], [0., 1.]]]))
        np.testing.assert_allclose(z, [[2 ** -.5, 2 ** -.5]])

    def test_graph_retains_isolated_images_and_real_pairs(self):
        s = np.eye(5, dtype=np.float32)
        s[0, 1] = s[1, 0] = .99; s[2, 3] = s[3, 2] = .98
        for kind in ['mutual', 'union']:
            cfg = {'topk': 10, 'threshold': .8, 'neighborhood': kind, 'resolution': .08}
            p = graph_cluster(s, cfg)
            self.assertEqual(p[0], p[1]); self.assertEqual(p[2], p[3]); self.assertEqual(len(set(p)), 3)
            np.testing.assert_array_equal(p, graph_cluster(s, cfg))

    def test_no_edges_never_fabricates_merges(self):
        cfg = {'topk': 10, 'threshold': .8, 'neighborhood': 'union', 'resolution': .02}
        self.assertEqual(len(set(graph_cluster(np.eye(5), cfg))), 5)

    def test_subset_removes_references_of_excluded_identities(self):
        c = {'mode': 'mixed', 'qi': np.arange(8), 'ri': np.arange(20, 24),
             'truth': np.repeat(['a', 'b', 'c', 'd'], 2), 'refs': np.array(['a', 'b', 'c', 'd']),
             'sq': np.eye(8), 'sr': np.ones((8, 4)), 'extra': {'learned': (np.eye(8), np.ones((8, 4)))}}
        s = subset_cases([c], 1)[0]
        self.assertEqual(len(s['qi']), 6)
        self.assertEqual(set(s['truth']), set(s['refs']))
        self.assertEqual(s['sr'].shape, (6, 3))

    def test_identity_partitions(self):
        initialize()
        import pandas as pd
        roles = pd.read_csv(RUN / 'calibration_roles.csv')
        _, split = metadata()
        self.assertFalse(set(roles.identity) & set(split.loc[split.partition != 'calibration', 'identity']))
        self.assertEqual(roles.groupby('identity').role.nunique().max(), 1)


if __name__ == '__main__':
    unittest.main()
