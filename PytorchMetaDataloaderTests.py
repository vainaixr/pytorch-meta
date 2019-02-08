import unittest
import contextlib
import sys
import os
import shutil
import torch
import h5py
import time

import TCGA
import utils


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

    def test_download_meta(self):
        data_dir = os.path.join('.', 'temp')
        os.makedirs(data_dir)
        TCGA.TCGAMeta(download=True, data_dir=data_dir)
        shutil.rmtree(data_dir)

    def test_download_task(self):
        data_dir = os.path.join('.', 'temp')
        os.makedirs(data_dir)
        TCGA.TCGATask(self.task_id, download=True, data_dir=data_dir)
        shutil.rmtree(data_dir)

    def test_tasks(self):

        dataset = TCGA.TCGATask(self.task_id)
        trainloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=15, shuffle=True)

        for data, labels in trainloader:
            with nostdout():
                print('\n', [dataset.categories[label] for label in labels])

    def test_meta_TCGA(self):
        dataset = TCGA.TCGAMeta(download=True)
        trainloader = dataset.get_dataloader(batch_size=15, shuffle=True)

        for batch in trainloader:
            for dataset in batch:
                with nostdout():
                    print('\n', dataset.id)

    def test_tasks_time(self):
        task_ids = TCGA.TCGA.get_TCGA_task_ids()

        # use only a subsample for timing
        num = 5
        task_ids = task_ids[0:num]

        data_dir = './metadatasets/data'
        with h5py.File(os.path.join(data_dir, 'TCGA_tissue_ppi.hdf5'), 'r') as f:
            gene_expression_data = f['expression_data'][()]
            gene_ids_file = os.path.join(data_dir, 'gene_ids')
            all_sample_ids_file = os.path.join(data_dir, 'all_sample_ids')
            with open(gene_ids_file, 'r') as file:
                gene_ids = file.readlines()
                self.gene_ids = [x.strip() for x in gene_ids]
            with open(all_sample_ids_file, 'r') as file:
                all_sample_ids = file.readlines()
                self.all_sample_ids = [x.strip() for x in all_sample_ids]

        def init_from_memory():
            for task_id in task_ids:
                TCGA.TCGATask(task_id, preloaded=(all_sample_ids, gene_ids, gene_expression_data))

        def init_lazy():
            for task_id in task_ids:
                TCGA.TCGATask(task_id)

        start_time = time.clock()
        init_from_memory()
        print('\nFrom memory we can init {} tasks in {} seconds'.format(num, time.clock() - start_time))

        start_time = time.clock()
        init_lazy()
        print('\nLazily we can init {} tasks in {} seconds'.format(num, time.clock() - start_time))
        

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, samples_per_class):
        self.labels = []
        for label, num in enumerate(samples_per_class):
            self.labels += [label for _ in range(num)]  
    
    def __getitem__(self, index):
       return None, self.labels[index]
    
    def __len__(self):
        return len(self.labels)


class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def test_stratified_sampling(self):
        dataset = DummyDataset([1000, 2])

        lenghts = [500, 502]
        datasets = utils.stratified_split(dataset, lenghts)

        for dataset in datasets:
            _, label = dataset.__getitem__(len(dataset)-1)
            self.assertEquals(label, 1)

    def test_stratified_sampling_TCGA(self):
        metadataset = TCGA.TCGAMeta(download=True)

        third_size = 10
        samples = 10
        for dataset in metadataset:
            for i in range(dataset.num_classes + third_size, len(dataset) - dataset.num_classes - third_size, len(dataset) // samples):
                length_minority_set = i

                lengths = [length_minority_set - third_size, len(dataset) - length_minority_set - third_size,
                           2 * third_size]

                sets = utils.stratified_split(dataset, lengths)

                all_labels = [[label for _, label in dataset] for dataset in sets]
                classes = list(set([label for labels in all_labels for label in labels]))
                contains_samples_for_all_classes = all(
                    [[class_name in label_list for label_list in all_labels] for class_name in classes])

                self.assertTrue(contains_samples_for_all_classes)

                matches_lenghts = [len(dataset) == length for dataset, length in zip(sets, lengths)]

                self.assertTrue(matches_lenghts)


if __name__ == '__main__':
    unittest.main()
