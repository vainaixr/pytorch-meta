import torch.utils.data as data
from torch import randperm


def classwise_split(dataset, shuffle=True):
    classes = set()
    classwise_indices = dict()
    for index in range(len(dataset)):
        _, label = dataset.__getitem__(index)

        # Should we encounter a new label we add a new bucket
        if label not in classes:
            classes.add(label)
            classwise_indices[label] = []

        # We add the current sample to the bucket of its class
        classwise_indices[label].append(index)
        
    if shuffle:
        # Torch randperm based shuffle of all buckets
        for key, value in classwise_indices.items():
            classwise_indices[key] = [value[index] for index in iter(randperm(len(value)))]
            
    return [data.Subset(dataset, classwise_indices[key]) for key in classwise_indices.keys()]


def stratified_split(dataset, lengths):
    
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    fractions = [length/total_length for length in lengths]

    classwise_datasets = classwise_split(dataset, shuffle=True)
    classwise_datasets = sorted(classwise_datasets, key=lambda dataset: len(dataset))
    num_samples_minority = len(classwise_datasets[0])

    num_splits = len(lengths)
    if num_samples_minority < num_splits:
        raise ValueError('The dataset can not be split in {} datasets because the minority class only has {} samples.'.format(num_splits, num_samples_minority))
    
    class_specific_split_datasets = []
    for dataset in classwise_datasets:
        class_specific_lengths = []
        for fraction in fractions:
            class_specific_lengths.append(int(round(len(dataset) * fraction)))

        class_specific_split_datasets.append(data.random_split(dataset, class_specific_lengths))
        
    datasets = []
    for i in range(len(fractions)):
        datasets.append(data.ConcatDataset([class_specific_dataset[i] for class_specific_dataset in class_specific_split_datasets]))
    return datasets 


