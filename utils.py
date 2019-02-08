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


def stratified_split(dataset, lengths, min_num_minority=1):
    
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset.")

    if any([length <= 0 for length in lengths]):
        raise ValueError("Any dataset needs to have a length greater zero.")

    classwise_datasets = classwise_split(dataset, shuffle=True)
    num_samples_minority = min([len(classwise_dataset) for classwise_dataset in classwise_datasets])

    num_splits = len(lengths)
    if num_samples_minority < num_splits*min_num_minority:
        raise ValueError('The dataset can not be split in {} datasets because the minority class only has {} samples.'.format(num_splits, num_samples_minority))

    fractions = [(length-num_splits)/(total_length-(num_splits*len(classwise_datasets))) for length in lengths]

    class_specific_split_datasets = []
    for classwise_dataset in classwise_datasets:
        ones = [item for item in [min_num_minority] for _ in range(num_splits)]
        first_split = data.random_split(classwise_dataset, [len(classwise_dataset)-num_splits] + ones)
        classwise_dataset = first_split[0]
        class_specific_single_elements = first_split[1:]

        class_specific_lengths = []
        for fraction in fractions[:-1]:
            class_specific_lengths.append(int(round(len(classwise_dataset) * fraction)))
        class_specific_lengths.append(len(classwise_dataset)-sum(class_specific_lengths))

        second_split = data.random_split(classwise_dataset, class_specific_lengths)
        rejoined_datasets = [data.ConcatDataset([first, second]) for first, second in zip(class_specific_single_elements, second_split)]
        class_specific_split_datasets.append(rejoined_datasets)
        
    datasets = []
    for i in range(num_splits):
        datasets.append(data.ConcatDataset([class_specific_dataset[i] for class_specific_dataset in class_specific_split_datasets]))
    return datasets 


