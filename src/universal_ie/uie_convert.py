#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
sys.path.append("../")

from typing import Any, Dict, List, Text, Union

from .generation_format import GenerationFormat
from .structure_marker import BaseStructureMarker
from universal_ie.text2spotasoc import Text2SpotAsoc
from .ie_format import Label, Sentence, Event, Span, Entity, Relation as UIERelation

def convert_sent_LLM_UIE(datasets: List, skip_NA = True):
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
        if example.get('relations'):
            uie_rels = convert_relations(example['relations'], skip_NA)
        
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
        if entity['type'] == 'NA' or entity['type'] == '':
            continue
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
        if event['type'] == 'NA' or event['type'] == '':
            continue
        args = []
        for idx, arg in enumerate(event['arguments']):
            label = Label(arg["role"])
            arg_tok = arg['name'].split(" ")
            entity = Entity(
                span=Span(
                    tokens=arg_tok,
                    indexes=[idx],
                    text=arg['name']
                ),
                label=label
            )
            args.append((label, entity))

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

def convert_relations(relations, skip_none = True):
    uie_rels = []
    for relation in relations:
        if relation['type'] == 'NA' or relation['type'] == '' or relation['type'] is None:
            continue
        if not relation['type']:
            relation['type'] = 'NA'
        
        head = Entity(
            span=Span(
                tokens=relation['head']['name'].split(" "),
                indexes=list(range(relation['head']['pos'][0], relation['head']['pos'][1])) if relation['head']['pos'] else [idx],
                text=relation['head']['name']
            ),
            label=Label("")
        )
        tail = Entity(
            span=Span(
                tokens=relation['tail']['name'].split(" "),
                indexes=list(range(relation['head']['pos'][0], relation['head']['pos'][1])) if relation['tail']['pos'] else [idx],
                text=relation['tail']['name']
            ),
            label=Label("")
        )
        uie_rel = UIERelation(
            head, tail, Label(relation['type'])
        )
        uie_rels.append(uie_rel)

    return uie_rels


def convert_graph(
    generation_class: GenerationFormat,
    datasets: List,
    label_mapper: Dict = None,
    skip_NA = True,
    existing_record_schema: Dict = None,
):
    convertor: Text2SpotAsoc = generation_class(
        structure_maker=BaseStructureMarker(),
        label_mapper=label_mapper
    )
    uie_datasets = convert_sent_LLM_UIE(datasets, skip_NA)

    prompts = []
    for instance in uie_datasets:
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
    record_schema = convertor.output_schema(existing_record_schema=existing_record_schema)
    return prompts, record_schema
