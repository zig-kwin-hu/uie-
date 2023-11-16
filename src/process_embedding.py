import numpy as np
import json
import os
import sklearn.cluster as cluster
import IPython
def get_increment(embedding_path):
    embedding = np.load(embedding_path)
    print(embedding.shape)
    #print(embedding)
    print('-------------------\n\n\n')
    #sort embedding based on the first axis
    embedding = embedding[np.argsort(embedding[:,0])]

    print(embedding[:,0])
    #generate the increment along the rows
    increment = []
    for i in range(embedding.shape[0]):
        if i > 0:
            increment.append(embedding[i][0] - embedding[i-1][0])
            if increment[-1] == 0:
                #compare distance between two embeddings
                distance = np.linalg.norm(embedding[i] - embedding[i-1])
            if embedding[i] in embedding[:i]:
                print(i, 'same embedding')
    print(increment)
def get_sets_of_same_embeddings(embedding_path):
    with open(embedding_path, 'r') as f:
        lines = f.readlines()
    js = []
    for line in lines:
        j = json.loads(line)
        #round embedding to 4
        j['embedding'] = np.round(np.array(j['embedding']), 1).tolist()
        js.append(j)
    #keys: {'task', 'dataset', 'embedding', 'decoded'}
    clusters = []
    added_dataset = set()
    for j in js:
        added = False
        if 'CoNLL 2003' in j['dataset']:
            print(j)
        for cluster in clusters:
            if j['embedding'] == cluster[0]['embedding']:
                if j['task']+'_'+j['dataset'] not in added_dataset:
                    cluster.append(j)
                    added_dataset.add(j['task']+'_'+j['dataset'])
                added = True
                break
        if not added and j['task']+'_'+j['dataset'] not in added_dataset:
            clusters.append([j])
            added_dataset.add(j['task']+'_'+j['dataset'])
    print(len(clusters))
    for cluster in clusters:
        print([(p['task'], p['dataset']) for p in cluster])
        if len(cluster) > 1:
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    if cluster[i]['task'] != cluster[j]['task'] or cluster[i]['dataset'] != cluster[j]['dataset']:
                        assert cluster[i]['embedding'] == cluster[j]['embedding']
                        print(cluster[i]['task'], cluster[i]['dataset'])
                        print(cluster[j]['task'], cluster[j]['dataset'])
                        print('---------------------')
def compute_similarity(embeddings, compute_method='cosine'):
    if compute_method == 'cosine':
        similarity = np.dot(embeddings, embeddings.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag
        similarity = cosine
        return similarity
    elif compute_method == 'euclidean':
        #compute euclidean distance between embeddings, every row is an embedding, the output is a matrix, the element in the matrix is the distance between two embeddings
        distance = np.linalg.norm(embeddings[:, None] - embeddings, axis=-1)
        #compute similarity based on distance
        similarity = -distance
        return similarity
    else:
        raise NotImplementedError('compute method not implemented {}'.format(compute_method))
def cluster_embeddings(embeddings, id2instance, compute_method='cosine', cluster_method='DBSCAN', **kwargs):
    #compute similarity between embeddings
    similarity = compute_similarity(embeddings, compute_method)
    distance = -similarity + np.max(similarity)
    #sort similarity based on the last axis
    #cluster embeddings
    if cluster_method == 'DBSCAN':
        #DBSCAN
        dbscan = cluster.DBSCAN(metric='precomputed', eps=kwargs['eps'], min_samples=kwargs['min_samples'])
        dbscan.fit(distance)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('number of clusters: {}'.format(n_clusters))
        print(labels)
        return labels
def remove_instances(embeddings, id2instance, remove_ids):
    new_embeddings = []
    new_id2instance = {}
    for i, embedding in enumerate(embeddings):
        if i not in remove_ids:
            new_embeddings.append(embedding)
            new_id2instance[len(new_embeddings)-1] = id2instance[i]
    return np.array(new_embeddings), new_id2instance
def keep_instances(embeddings, id2instance, keep_ids):
    new_embeddings = []
    new_id2instance = {}
    for i, embedding in enumerate(embeddings):
        if i in keep_ids:
            new_embeddings.append(embedding)
            new_id2instance[len(new_embeddings)-1] = id2instance[i]
    return np.array(new_embeddings), new_id2instance
def collect_clusters(labels, id2instance, cluster_hierarchy=0, local_id=0):
    cluster2instances = {}
    label2globalid = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in label2globalid:
            global_id = str(cluster_hierarchy)+'_'+str(local_id)
            label2globalid[label] = global_id
            cluster2instances[global_id] = []
            local_id += 1
        cluster2instances[label2globalid[label]].append(id2instance[i])
    return cluster2instances, local_id
def print_cluster(cluster2instances):
    for cluster in cluster2instances:
        print('cluster: {}'.format(cluster), len(cluster2instances[cluster]))
        for instance in cluster2instances[cluster]:
            print(instance['task'], instance['dataset'])
            labels_path = os.path.join('data/ie_instruct/', instance['task'], instance['dataset'], 'labels.json')
            with open(labels_path, encoding="utf-8") as labels_f:
                labels = json.load(labels_f)
            print(labels)
            instance['labels'] = labels
        print('------------------------')
    return cluster2instances
def print_hierarchical_cluster(hierarchical_clusters, level=0):
    prefix = '    '*level
    for cluster in hierarchical_clusters:
        print(prefix+'cluster: {}'.format(cluster), len(hierarchical_clusters[cluster]))
        if isinstance(hierarchical_clusters[cluster], dict):
            print_hierarchical_cluster(hierarchical_clusters[cluster], level+1)
        elif isinstance(hierarchical_clusters[cluster], list):
            for instance in hierarchical_clusters[cluster]:
                print(prefix+instance['task'], instance['dataset'])
                labels_path = os.path.join('data/ie_instruct/', instance['task'], instance['dataset'], 'labels.json')
                with open(labels_path, encoding="utf-8") as labels_f:
                    labels = json.load(labels_f)
                labels_prefix = '    '*(level+1)
                print(labels_prefix+str(labels))
def aggregate_dataset_embeddings(embeddings, id2instance, aggregate_method='mean', sample_num=10):
    print(len(embeddings), len(id2instance))
    dataset2embeddings = {}
    for i, embedding in enumerate(embeddings):
        task = id2instance[i]['task']
        dataset = id2instance[i]['dataset']
        if task not in dataset2embeddings:
            dataset2embeddings[task] = {}
        if dataset not in dataset2embeddings[task]:
            dataset2embeddings[task][dataset] = []
        dataset2embeddings[task][dataset].append(embedding)

    for task in dataset2embeddings:
        for dataset in dataset2embeddings[task]:
            dataset2embeddings[task][dataset] = np.array(dataset2embeddings[task][dataset])
            if sample_num > 0:
                sample_num = min(sample_num, dataset2embeddings[task][dataset].shape[0])
                #shuffle the embeddings
                np.random.shuffle(dataset2embeddings[task][dataset])
                dataset2embeddings[task][dataset] = dataset2embeddings[task][dataset][:sample_num]
            if aggregate_method == 'mean':
                dataset2embeddings[task][dataset] = np.mean(dataset2embeddings[task][dataset], axis=0)
    new_embeddings = []
    new_id2instance = {}
    for task in dataset2embeddings:
        for dataset in dataset2embeddings[task]:
            new_embeddings.append(dataset2embeddings[task][dataset])
            for i in range(len(dataset2embeddings[task][dataset])):
                new_id2instance[len(new_embeddings)-1] = {'task':task, 'dataset':dataset}
    print(len(new_embeddings), len(new_id2instance))
    return np.array(new_embeddings), new_id2instance
if __name__ == '__main__':
    np.random.seed(42)
    #get_increment('/home/aiops/liuqian/ychong/InstructUIE_1/output/EMBED_INSTRUCTION/iuie/embeddings.npy')
    data_path = 'data/ie_instruct/EMBED_INSTRUCTION/with_sentence/test.json'
    embedding_path = 'output/EMBED_INSTRUCTION_with_sentence/iuie_mean_of_encoder/embeddings.npy'
    data = json.load(open(data_path))
    embeddings = np.load(embedding_path)
    id2instance = {}
    for i, instance in enumerate(data):
        id2instance[i] = instance
    
    if 'with_sentence' in data_path:
        embeddings, id2instance = aggregate_dataset_embeddings(embeddings, id2instance, aggregate_method='mean', sample_num=100)

    to_remove = []
    for i, embedding in enumerate(embeddings):
        if 'fewrel_' in id2instance[i]['dataset'] or 'wiki_' in id2instance[i]['dataset']:
            to_remove.append(i)
    embeddings, id2instance = remove_instances(embeddings, id2instance, to_remove)
    all_clusters = []
    current_eps = 0.01
    current_min_samples = 2
    print('------------------------\n'*2)
    print('eps: {}, min_samples: {}'.format(current_eps, current_min_samples))
    labels = cluster_embeddings(embeddings, id2instance, compute_method='cosine', cluster_method='DBSCAN', **{'eps':current_eps, 'min_samples':current_min_samples})
    cluster2instances, local_id = collect_clusters(labels, id2instance, cluster_hierarchy=0, local_id=0)
    cluster2instances = print_cluster(cluster2instances)
    for c in cluster2instances:
        all_clusters.append({'eps':current_eps, 'min_samples':current_min_samples, 'cluster':c, 'instances':cluster2instances[c]})
    
    for temp in range(2,11):
        keep_ids = []
        for i, label in enumerate(labels):
            if label == -1:
                keep_ids.append(i)
        if len(keep_ids) == 0:
            break
        embeddings, id2instance = keep_instances(embeddings, id2instance, keep_ids)
        current_eps = 0.01*temp
        current_min_samples = 2
        print('------------------------\n'*2)
        print('eps: {}, min_samples: {}'.format(current_eps, current_min_samples))
        labels = cluster_embeddings(embeddings, id2instance, compute_method='cosine', cluster_method='DBSCAN', **{'eps':current_eps, 'min_samples':current_min_samples})
        cluster2instances, local_id = collect_clusters(labels, id2instance, cluster_hierarchy=0, local_id=local_id)
        cluster2instances = print_cluster(cluster2instances)
        for c in cluster2instances:
            all_clusters.append({'eps':current_eps, 'min_samples':current_min_samples, 'cluster':c, 'instances':cluster2instances[c]})
    #save all_clusters
    with open('output/EMBED_INSTRUCTION_with_sentence/iuie_mean_of_encoder/all_clusters.json', 'w') as f:
        json.dump(all_clusters, f, indent=4)


    
    