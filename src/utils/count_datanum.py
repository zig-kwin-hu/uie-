import os
import json
if __name__ == '__main__':
    print('count data num')
    tasks = ['NER']
    for task in tasks:
        datasets = os.listdir('./data/ie_instruct/{}'.format(task))
        for dataset in datasets:
            dataset_path = './data/ie_instruct/{}/{}'.format(task, dataset)
            train_path = os.path.join(dataset_path, 'train.json')
            dev_path = os.path.join(dataset_path, 'dev.json')
            test_path = os.path.join(dataset_path, 'test.json')
            try:
                train = json.load(open(train_path, 'r'))
                train_num = len(train)
            except:
                train_num = 0
            try:
                dev = json.load(open(dev_path, 'r'))
                dev_num = len(dev)
            except:
                dev_num = 0
            test = json.load(open(test_path, 'r'))
            test_num = len(test)
            print('{} {}: train_num: {}, dev_num: {}, test_num: {}'.format(task, dataset, train_num, dev_num, test_num))
            sentence_length = []
            for data in train:
                sentence_length.append(len(data['sentence']))
            # show statistics about sentence length
            print('len max: {} min: {} average: {} median: {} 90%: {}\n\n'.format(max(sentence_length),
                                                                       min(sentence_length),
                                                                round(sum(sentence_length)/len(sentence_length),2),
                                                                round(sorted(sentence_length)[len(sentence_length)//2],2),
                                                                round(sorted(sentence_length)[len(sentence_length)//10*9],2)))
            
