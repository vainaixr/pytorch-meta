import unittest
import contextlib
import sys
import os
import shutil
import torch

import torchmeta as tmds


class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = save_stdout


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.task_id = ('gender', 'BRCA')
        data_dir = os.path.join('C:/Users/Tobias/Documents', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        tmds.datasets.TCGA(data_dir, download=True)
        self.data_dir = data_dir

    def test_download_meta(self):
        data_dir = os.path.join('.', 'temp')
        os.makedirs(data_dir)
        tmds.datasets.TCGA(data_dir, download=True)
        shutil.rmtree(data_dir)

    def test_tasks(self):

        dataset = tmds.datasets.tcga.TCGATask.from_id(self.data_dir, self.task_id)
        trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=15, shuffle=True)

        for data, labels in trainloader:
            with nostdout():
                print('\n', [dataset.categories[label] for label in labels])

    def test_meta_TCGA(self):
        dataset = tmds.datasets.TCGA(self.data_dir)
        trainloader = tmds.MetaDataLoader(dataset, batch_size=15, shuffle=True)

        for batch in trainloader:
            for dataset in batch:
                with nostdout():
                    print('\n', dataset.id)

    def test_split(self):
        dataset = tmds.datasets.TCGA(self.data_dir)
        counts = [50, 50, 96]
        train, valid, test = tmds.datasets.split_tcga(dataset, counts)
        result_counts = [len(train), len(valid), len(test)]
        result = all([a == b for a, b in zip(counts, result_counts)])
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
