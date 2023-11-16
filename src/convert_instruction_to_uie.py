import json
import os
import random
import copy
def convert_no_sentence(data_path, tasks, split='train'):
    all_instances = []
    for task in tasks:
        task_path =  os.path.join(data_path, task)
        all_datasets = os.listdir(task_path)
        for d in all_datasets:
            all_instances.append({'task':task, 'dataset':d})
    save_path = os.path.join(data_path, 'EMBED_INSTRUCTION', 'no_sentence',  split + '.json')
    #check whether the directory of save_path exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        json.dump(all_instances, f, indent=2)
def convert_with_sentence(data_path, datasets, split='train', sample_num=100):
    all_instances = []
    for i in range(len(datasets)):
        task = datasets[i]['task']
        dataset = datasets[i]['dataset']
        dataset_path = os.path.join(data_path, task, dataset)
        split_path = os.path.join(dataset_path, split + '.json')
        if not os.path.exists(split_path):
            print('split path does not exist: ', split_path)
            continue
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        if len(split_data) > sample_num:
            #shuffle the data
            random.shuffle(split_data)
            split_data = split_data[:sample_num]
        for instance in split_data:
            new_instance = copy.deepcopy(instance)
            new_instance['task'] = task
            new_instance['dataset'] = dataset
            all_instances.append(new_instance)
    save_path = os.path.join(data_path, 'EMBED_INSTRUCTION', 'with_sentence',  split + '.json')
    #check whether the directory of save_path exists
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    print('split {} number of instances: {}'.format(split, len(all_instances))) 
    with open(save_path, 'w') as f:
        json.dump(all_instances, f, indent=2)
    
if __name__ == '__main__':
    random.seed(42)
    data_path = 'data/ie_instruct'
    tasks = ['EEA', 'RE', 'NER', 'EET']
    #convert_no_sentence(data_path, tasks, split='train')
    #convert_no_sentence(data_path, tasks, split='dev')
    #convert_no_sentence(data_path, tasks, split='test')
    datasets = json.load(open('/storage/zkhu/InstructUIE/data/ie_instruct/EMBED_INSTRUCTION/no_sentence/test.json', 'r'))
    convert_with_sentence(data_path, datasets, split='test', sample_num=100)
    convert_with_sentence(data_path, datasets, split='dev', sample_num=100)
    convert_with_sentence(data_path, datasets, split='train', sample_num=100)
        