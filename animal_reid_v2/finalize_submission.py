"""Apply the official cluster-name convention without changing any grouping."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import RUN, DATA, json_write, fingerprint


def main():
    path = RUN / 'submission.csv'
    out = pd.read_csv(path)
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    metadata = pd.read_csv(DATA / 'metadata.csv')
    assert out.image_id.tolist() == sample.image_id.tolist()
    assert len(out) == 2409 and out.image_id.is_unique and out.cluster.notna().all()
    before_sha = fingerprint(path)
    before_labels = out.cluster.astype(str).copy()
    out['cluster'] = before_labels.map(lambda x: x if x.startswith('cluster_') else 'cluster_' + x)
    # Equality of canonical partitions proves that renaming cannot change ARI.
    assert np.array_equal(pd.factorize(before_labels)[0], pd.factorize(out.cluster)[0])
    joined = metadata[metadata.split == 'test'].merge(out, on='image_id', validate='one_to_one')
    assert all(label.startswith('cluster_' + dataset + '_') and len(label) > len('cluster_' + dataset + '_')
               for dataset, label in zip(joined.dataset, joined.cluster))
    if not out.cluster.equals(before_labels):
        backup = RUN / 'submission_before_format_fix.csv'
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        tmp = path.with_suffix('.tmp.csv')
        out.to_csv(tmp, index=False)
        tmp.replace(path)
    after_sha = fingerprint(path)
    complete = json.loads((RUN / 'completion_check.json').read_text())
    complete['submission_sha256'] = after_sha
    complete['official_cluster_name_format'] = True
    json_write(complete, RUN / 'completion_check.json')
    provenance = RUN / 'submission_format_check.json'
    if before_sha != after_sha or not provenance.exists():
        json_write({'rows': len(out), 'before_sha256': before_sha, 'after_sha256': after_sha,
                    'partition_unchanged': True, 'official_cluster_name_format': True,
                    'source': 'https://www.kaggle.com/competitions/animal-clef-2026'}, provenance)
    print(json.dumps({'rows': len(out), 'partition_unchanged': True, 'official_cluster_name_format': True,
                      'submission_sha256': after_sha}))


if __name__ == '__main__':
    main()
