import json
import os
clusters = json.load(open('/storage/zkhu/InstructUIE/output/EMBED_INSTRUCTION_no_sentence/iuie_mean_of_encoder/all_clusters.json'))
clusters_dir = './configs/clusters/no_sentence_iuie_mean_of_encoder/'
template_dev = {
    "NER": [
        {
            "sampling strategy": "full",
            "dataset name": "ACE 2004"
        },
        {
            "sampling strategy": "full",
            "dataset name": "ACE 2005"
        }
    ]
}

if not os.path.exists(clusters_dir):
    os.makedirs(clusters_dir)
for cluster in clusters:
    cluster_name = cluster['cluster']
    clusters_dev_test = {}
    clusters_train = {}
    for instance in cluster['instances']:
        if instance['task'] not in clusters_dev_test:
            clusters_dev_test[instance['task']] = []
        if instance['task'] not in clusters_train:
            clusters_train[instance['task']] = []
        clusters_dev_test[instance['task']].append({
            "sampling strategy": "full",
            "dataset name": instance['dataset']
        })
        clusters_train[instance['task']].append({
            "sampling strategy": "random",
            "dataset name": instance['dataset']
        })
    cluster_dir = os.path.join(clusters_dir, cluster_name)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    json.dump(clusters_dev_test, open(os.path.join(cluster_dir, 'dev_tasks.json'), 'w'), indent=4)
    json.dump(clusters_dev_test, open(os.path.join(cluster_dir, 'test_tasks.json'), 'w'), indent=4)
    json.dump(clusters_train, open(os.path.join(cluster_dir, 'train_tasks.json'), 'w'), indent=4)
    