import json
import os
def generate_pos_index_of_substring(substring, string):
    start = string.find(substring)
    if start == -1:
        raise ValueError(f"Substring {substring} not found in string {string}")
    else:
        end = start + len(substring)
        return [start, end]
dataset_path = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/ADE_corpus_origin/'
new_dataset_path = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/ADE_corpus/'
for subset in ['dev.json','test.json','train.json']:
    subset_path = os.path.join(dataset_path, subset)
    subset_data = json.load(open(subset_path))
    for instance in subset_data:
        for i,r in enumerate(instance['relations']):
            name = instance['relations'][i]['head']['name']
            pos = generate_pos_index_of_substring(name, instance['sentence'])
            instance['relations'][i]['head']['pos'] = pos
            name = instance['relations'][i]['tail']['name']
            pos = generate_pos_index_of_substring(name, instance['sentence'])
            instance['relations'][i]['tail']['pos'] = pos
    new_subset_path = os.path.join(new_dataset_path, subset)
    json.dump(subset_data, open(new_subset_path, 'w'), indent=2)