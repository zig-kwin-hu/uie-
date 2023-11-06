import json
import os
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
if __name__ == '__main__':
    data_path = '../data/ie_instruct'
    tasks = ['EEA', 'RE', 'NER', 'EET']
    convert_no_sentence(data_path, tasks, split='train')
    convert_no_sentence(data_path, tasks, split='dev')
    convert_no_sentence(data_path, tasks, split='test')
    
        