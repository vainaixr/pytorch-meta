import torch.utils.data as data
from torch import randperm


def classwise_split(dataset, shuffle = True):
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
    
    total_length  = sum(lengths) 
    
    if total_length != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    fractions = [length/total_length for length in lengths]
    classwise_dataset = classwise_split(dataset, shuffle = True)
    
    class_specific_split_datasets = []
    for dataset in classwise_dataset:
        num_samples = len(dataset)
        class_specific_lengths = []
        for fraction in fractions[:-1]:
            class_specific_lengths.append(int(round(num_samples * fraction)))
        class_specific_lengths.append(num_samples-sum(class_specific_lengths))
            
        class_specific_split_datasets.append( data.random_split(dataset, class_specific_lengths) )
        
    datasets = []
    for i in range(len(fractions)):
        datasets.append(data.ConcatDataset([class_specific_dataset[i] for class_specific_dataset in class_specific_split_datasets]))
    return datasets 


