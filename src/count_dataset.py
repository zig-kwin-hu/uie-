import json
import os
import numpy as np
from collections import defaultdict
tasks = ['RE']
datasets = {'EE': ['ACE05','CASIE','GENIA', 'PHEE'],
            'RE': ['ADE_corpus','conll04','GIDS','kbp37','New-York-Times-RE','NYT11','SciERC','semval-RE','fewrel_0','wiki_0']}
def compute_mean_and_deviation(data):
    data = np.array(data)
    mean = np.mean(data)
    dev = np.sqrt(np.mean((data-mean)**2))
    return mean,dev
def count_overlapping_relation_num(example, task='', dataset=''):
    entity2relation = defaultdict(set)
    entity_pair2relation = defaultdict(set)
    overlapping_1 = set() #triples whose head or tail is in entity_set
    overlapping_2 = set() #triples whose head and tail are both in entity_set
    for rindex, r in enumerate(example['relations']):
        if r['type'] in ['NA','']:
            continue
        entity2relation[r['head']['name']].add(rindex)
        entity2relation[r['tail']['name']].add(rindex)
        entity_pair2relation[(r['head']['name'],r['tail']['name'])].add(rindex)
    for e in entity2relation:
        if len(entity2relation[e]) > 1:
            overlapping_1.update(entity2relation[e])
    for e in entity_pair2relation:
        if len(entity_pair2relation[e]) > 1:
            overlapping_2.update(entity_pair2relation[e])
    return len(overlapping_1), len(overlapping_2)
        
for task in tasks:
    for dataset in datasets[task]:
        files = os.listdir('../data/ie_instruct/{}/{}/'.format(task,dataset))
        with open('../data/ie_instruct/{}/{}/train.json'.format(task,dataset),'r') as f:
            train = json.load(f)
        if 'dev.json' in files:
            with open('../data/ie_instruct/{}/{}/dev.json'.format(task,dataset),'r') as f:
                dev = json.load(f)
        else:
            dev = []
        with open('../data/ie_instruct/{}/{}/test.json'.format(task,dataset),'r') as f:
            test = json.load(f)
        with open('../data/ie_instruct/{}/{}/labels.json'.format(task,dataset),'r') as f:
            labels = json.load(f)
        print(task,dataset)
        print('train:',len(train))
        print('dev:',len(dev))
        print('test:',len(test))
        print('total:',len(train)+len(dev)+len(test))
        all_data = train+dev+test
        sentence_lengths = [len(s['sentence'].split(' ')) for s in all_data]
        
        sentence_lengths_mean_and_dev = compute_mean_and_deviation(sentence_lengths)
        print('sentence length mean and deviation:',sentence_lengths_mean_and_dev)
        if task == 'EE':
            splitor = ', ' if ', ' in labels[0] else ','
            event_types = labels[0].split(splitor) if len(labels[0]) > 0 else []
            argument_roles = labels[1].split(splitor) if len(labels[1]) > 0 else []
            print('event types:',event_types)
            print('argument roles:',argument_roles)
            events_nums = [len(s['events']) for s in all_data]
            argument_nums = [len(a['arguments']) for s in all_data for a in s['events']]
            events_nums_mean_and_dev = compute_mean_and_deviation(events_nums)
            argument_nums_mean_and_dev = compute_mean_and_deviation(argument_nums)
            print('number of event types:',len(event_types))
            print('number of argument roles:',len(argument_roles))
            print('number of events mean and deviation:',events_nums_mean_and_dev)
            print('number of arguments mean and deviation:',argument_nums_mean_and_dev)
            no_event_sentences = [s for s in all_data if len(s['events']) == 0]
            print('number of sentences without events:',len(no_event_sentences))
            NA_type_events = [e for s in all_data for e in s['events'] if e['type'] in ['NA','']]
            print('number of NA type events:',len(NA_type_events))
        if task == 'RE':
            rel_types = labels
            print('relation types:',rel_types)
            rel_nums = [len(s['relations']) for s in all_data]
            rel_nums_mean_and_dev = compute_mean_and_deviation(rel_nums)
            print('number of relation types:',len(rel_types))
            print('number of relations mean and deviation:',rel_nums_mean_and_dev, 'min:',np.min(rel_nums), 'max:',np.max(rel_nums), 'median:',np.median(rel_nums))
            no_rel_sentences = []
            for s in all_data:
                if len(s['relations']) == 0:
                    no_rel_sentences.append(s)
                elif all([r['type'] in ['NA',''] for r in s['relations']]):
                    no_rel_sentences.append(s)
            overlapping_rnum_1and2 = [count_overlapping_relation_num(s, task, dataset) for s in all_data]
            overlapping_rnum_1, overlapping_rnum_2 = zip(*overlapping_rnum_1and2)
            print('number of overlapping relations with >= one overlapping entities:',np.sum(overlapping_rnum_1),'with two entities',np.sum(overlapping_rnum_2))
            print('number of sentences without relations:',len(no_rel_sentences))
            NA_type_relations = [r for s in all_data for r in s['relations'] if r['type'] in ['NA','']]
            print('number of NA type relations:',len(NA_type_relations))
            assert max(rel_nums) > 1 or sum(overlapping_rnum_1) == 0
            assert sum(rel_nums) >= sum(overlapping_rnum_1)
        print('')

