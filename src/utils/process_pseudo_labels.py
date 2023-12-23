import json
def highlight_entities_md(sentence, relations):
    highlighted_sentence = sentence
    for r in relations:
        for entity in [r[0], r[2]]:
            if entity in highlighted_sentence:
                highlighted_sentence = highlighted_sentence.replace(f"{entity}", f"`**{entity}**`")
    return highlighted_sentence


def find_overlap(pseudo_label, ground_truth, sentence=None):
    pseudo_label = set(pseudo_label)
    ground_truth = set(ground_truth)
    pair2type_ground_truth = {}
    pair2type_pseudo_label = {}
    overlap_pseudo_label = {}
    overlap_ground_truth = {}
    for r in ground_truth:
        if (r[0], r[2]) not in pair2type_ground_truth:
            pair2type_ground_truth[(r[0], r[2])] = []
        pair2type_ground_truth[(r[0], r[2])].append(r[1])
        if r[1] not in overlap_ground_truth:
            overlap_ground_truth[r[1]] = {}
    for r in pseudo_label:
        if (r[0], r[2]) not in pair2type_pseudo_label:
            pair2type_pseudo_label[(r[0], r[2])] = []
        pair2type_pseudo_label[(r[0], r[2])].append(r[1])
        if r[1] not in overlap_pseudo_label:
            overlap_pseudo_label[r[1]] = {}
    for pair in pair2type_ground_truth:
        if pair in pair2type_pseudo_label:
            for type in pair2type_ground_truth[pair]:
                for type2 in pair2type_pseudo_label[pair]:
                    if type2 not in overlap_ground_truth[type]:
                        overlap_ground_truth[type][type2] = 0
                    overlap_ground_truth[type][type2] += 1
        else:
            for type in pair2type_ground_truth[pair]:
                if 'unmatched' not in overlap_ground_truth[type]:
                    overlap_ground_truth[type]['unmatched'] = 0
                overlap_ground_truth[type]['unmatched'] += 1
    for pair in pair2type_pseudo_label:
        if pair in pair2type_ground_truth:
            for type in pair2type_pseudo_label[pair]:
                for type2 in pair2type_ground_truth[pair]:
                    if type2 not in overlap_pseudo_label[type]:
                        overlap_pseudo_label[type][type2] = 0
                    overlap_pseudo_label[type][type2] += 1
        else:
            for type in pair2type_pseudo_label[pair]:
                if 'unmatched' not in overlap_pseudo_label[type]:
                    overlap_pseudo_label[type]['unmatched'] = 0
                overlap_pseudo_label[type]['unmatched'] += 1
    return overlap_pseudo_label, overlap_ground_truth
def merge_overlap(overlap1, overlap2):
    for type in overlap2:
        if type not in overlap1:
            overlap1[type] = {}
        for type2 in overlap2[type]:
            if type2 not in overlap1[type]:
                overlap1[type][type2] = 0
            overlap1[type][type2] += overlap2[type][type2]
    return overlap1
pseudo_label_file = '/storage/zkhu/UIE-pp/output/re/finred_pseudo_label/predict_eval_predictions.jsonl'
ground_truth_file = '/storage/zkhu/UIE-pp/data/ie_instruct/RE/semval-RE/train.json'

pseudo_labels = [json.loads(l) for l in open(pseudo_label_file).readlines()]
ground_truths = json.load(open(ground_truth_file))
assert len(pseudo_labels) == len(ground_truths), f'length not match {len(pseudo_labels)} {len(ground_truths)}'
all_overlap_pseudo_label = {}
all_overlap_ground_truth = {}
output_file = '/storage/zkhu/UIE-pp/output/re/finred_pseudo_label/highlighed.md'
fout = open(output_file, 'w')
for i in range(len(pseudo_labels)):
    assert pseudo_labels[i]['Instance']['sentence'] == ground_truths[i]['sentence'], 'sentence not match'
    pseudo_label = [(p[0],p[1],p[2]) for p in pseudo_labels[i]['Prediction']]
    ground_truth = ground_truths[i]['relations']
    ground_truth = [(r['head']['name'], r['type'], r['tail']['name']) for r in ground_truth]
    overlap_pseudo_label, overlap_ground_truth = find_overlap(pseudo_label, ground_truth)
    all_overlap_pseudo_label = merge_overlap(all_overlap_pseudo_label, overlap_pseudo_label)
    all_overlap_ground_truth = merge_overlap(all_overlap_ground_truth, overlap_ground_truth)
    #pseudo_highlighted_sentence = highlight_entities_md(pseudo_labels[i]['Instance']['sentence'], pseudo_label)
    #ground_truth_highlighted_sentence = highlight_entities_md(ground_truths[i]['sentence'], ground_truth)
    fout.write(f'## {i}\n')
    fout.write(f'### sentence\n')
    fout.write(f'{ground_truths[i]["sentence"]}\n')
    fout.write(f'### pseudo_label\n')
    fout.write(f'{json.dumps(pseudo_label, indent=2)}\n')
    fout.write(f'### ground_truth\n')
    fout.write(f'{json.dumps(ground_truth, indent=2)}\n')

print('pseudo_label')
print(json.dumps(all_overlap_pseudo_label, indent=2))
print('ground_truth')
print(json.dumps(all_overlap_ground_truth, indent=2))
    

