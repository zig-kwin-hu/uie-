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
def cluster_embeddings(embeddings, id2instance, compute_method='cosine', cluster_method='DBSCAN'):
    #compute similarity between embeddings
    similarity = compute_similarity(embeddings, compute_method)
    distance = -similarity + np.max(similarity)
    #sort similarity based on the last axis
    #cluster embeddings
    if cluster_method == 'DBSCAN':
        #DBSCAN
        dbscan = cluster.DBSCAN(metric='precomputed', eps=0.05, min_samples=2)
        dbscan.fit(distance)
        labels = dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('number of clusters: {}'.format(n_clusters))
        print(labels)
        return labels
if __name__ == '__main__':
    #get_increment('/home/aiops/liuqian/ychong/InstructUIE_1/output/EMBED_INSTRUCTION/iuie/embeddings.npy')
    data = json.load(open('data/ie_instruct/EMBED_INSTRUCTION/no_sentence/test.json'))
    id2instance = {}
    for i, instance in enumerate(data):
        id2instance[i] = instance
    embedding_path = '/home/aiops/liuqian/ychong/InstructUIE_1/output/EMBED_INSTRUCTION/iuie_mean_of_encoder/embeddings.npy'
    embeddings = np.load(embedding_path)
    labels = cluster_embeddings(embeddings, id2instance, compute_method='cosine', cluster_method='DBSCAN')
    cluster2instances = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in cluster2instances:
            cluster2instances[label] = []
        cluster2instances[label].append(id2instance[i])
    for cluster in cluster2instances:
        print('cluster: {}'.format(cluster), len(cluster2instances[cluster]))
        for instance in cluster2instances[cluster]:
            print(instance['task'], instance['dataset'])
            labels_path = os.path.join('data/ie_instruct/', instance['task'], instance['dataset'], 'labels.json')
            with open(labels_path, encoding="utf-8") as labels_f:
                labels = json.load(labels_f)
            print(labels)
        print('------------------------')

    
    