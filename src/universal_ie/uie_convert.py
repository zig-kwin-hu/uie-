#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append("../")

from typing import Any, Dict, List, Text, Union
from tqdm import tqdm

from .generation_format import GenerationFormat
from .structure_marker import BaseStructureMarker
from universal_ie.text2spotasoc import Text2SpotAsoc
from .ie_format import Label, Sentence, Event, Span, Entity, Relation as UIERelation

def convert_sent_LLM_UIE(datasets: List):
    uie_sentences: List[Sentence] = []
    for example in datasets:
        sent_tokens = example['sentence'].split(" ")

        uie_ents = None
        uie_evs = None
        uie_rels = None

        if example.get('entities'):
            uie_ents = convert_entities(example['entities'])
        if example.get('events'):
            uie_evs = convert_events(example['events'])
        

        # uie_rels = convert_relations(example.re_idx, sent_tokens, example.type, [example.obj, example.subj])
        
        uie_sent = Sentence(
            tokens=sent_tokens,
            entities=uie_ents,
            events=uie_evs,
            relations=uie_rels
        )
        uie_sentences.append(uie_sent)

    return uie_sentences

def convert_entities(entities):
    uie_ents = []
    for entity in entities:
        start, end = entity['pos']
        indexes = list(range(start, end))
        tokens = entity['name'].split(' ')
        ent = Entity(
            span=Span(
                tokens=tokens,
                indexes=indexes,
                text=entity['name']
            ),
            label=Label(entity['type'])
        )
        uie_ents.append(ent)
    return uie_ents


def convert_events(events): 
    uie_evs = []
    for event in events:
        args = []
        # for arg in event['arguments']:
        #     label = Label(arg["role"])
        #     arg_idx = list(range(arg["start"], arg["end"]))
        #     arg_tok = tokens[arg["start"]: arg["end"]]
        #     entity = Entity(
        #         span=Span(
        #             tokens=arg_tok,
        #             indexes=arg_idx,
        #             text=" ".join(arg_tok)
        #         ),
        #         label=Label(arg["role"])
        #     )
        #     args.append((label, entity))

        uie_ev = Event(
            span=Span(
                tokens=event['trigger'].split(" "),
                indexes=[],
                text=event['trigger']
            ),
            label=Label(event['type']),
            args=args
        )
        uie_evs.append(uie_ev)
    
    return uie_evs

def convert_relations(re_id, tokens, re_type, re_entities):
    uie_rels = []
    uie_entities = []
    for entity in re_entities:
        start = entity['start'] + 1
        end = entity['end']
        rel_tok = tokens[start:end]
        indexes = list(range(start, end))
        entity = Entity(
            span=Span(
                tokens=rel_tok,
                indexes=indexes,
                text=" ".join(rel_tok)
            ),
            label=Label('entity'),
            record_id=re_id
        )
        uie_entities.append(entity)

    uie_rel = UIERelation(
        uie_entities[0], uie_entities[1], Label(re_type), re_id, re_id
    )
    uie_rels.append(uie_rel)

    return uie_rels


def convert_graph(
    generation_class: GenerationFormat,
    datasets: List,
    task: Text = None,
    label_mapper: Dict = None,
):
    convertor: Text2SpotAsoc = generation_class(
        structure_maker=BaseStructureMarker(),
        label_mapper=label_mapper
    )
    uie_datasets = convert_sent_LLM_UIE(datasets)

    prompts = []
    for instance in uie_datasets:
        # if task == 'EEA':
        #     instance.events
        converted_graph = convertor.annonote_graph(
            tokens=instance.tokens,
            entities=instance.entities,
            events=instance.events,
            relations=instance.relations
        )
        
        # offset_events = [
        #     event.to_offset(evt_label_mapper=label_mapper)
        #     for event in instance.events
        # ]
        src, tgt, spot_labels, asoc_labels = converted_graph[:4]
        # spot_asoc = converted_graph[4]
        prompts.append(tgt)
    record_schema = convertor.output_schema()
    return prompts, record_schema
