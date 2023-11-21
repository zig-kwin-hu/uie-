
from dataclasses import dataclass
import torch
import logging
import random
import math
from collections import OrderedDict
from transformers import PreTrainedTokenizerBase

from .generation_format import RecordSchema
from .structure_marker import spot_prompt, asoc_prompt, text_start


logger = logging.getLogger("__main__")


class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        schema: RecordSchema,
        positive_rate=1,
        negative=5,
        eval_negative=-1,
        ordered_prompt=False
    ) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer)
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt = tokenizer.get_vocab()[spot_prompt]
        self.asoc_prompt = tokenizer.get_vocab()[asoc_prompt]
        self.text_start = tokenizer.get_vocab()[text_start]
        self.positive_rate = positive_rate if positive_rate > 0 and positive_rate < 1 else 1
        self.negative = negative
        self.eval_negative = eval_negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample, Negative: {self.negative}, Ordered Prompt: {self.ordered_prompt}")
        self.tokenizer = tokenizer

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            schema_ordered_dict[name] = tokenizer.encode(name, add_special_tokens=False)
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        
        negative_set = set()
        for index in torch.randperm(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)
        
        return list(negative_set)

    def sample_spot(self, positive, evaluate=False, negative=None):
        """ Sample spot
        """
        neg = negative if negative != None else self.negative
        negative_spot = self.sample_negative(postive=positive, candidates=self.spot_list, k=neg if not evaluate else self.eval_negative)
        positive_spot = random.sample(positive, math.floor(len(positive) * self.positive_rate))

        prefix_spot_candidates = positive_spot + negative_spot
        spot_prefix_ids = self.convert_prefix(
            candidates=prefix_spot_candidates,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=True if evaluate else self.ordered_prompt,
        )

        spot_prefix = self.tokenizer.decode(spot_prefix_ids)

        return spot_prefix_ids, spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive, evaluate=False, candidates=[]):
        """ Sample Asoc
        """
        negative_asoc = self.sample_negative(postive=positive, candidates=candidates or self.asoc_list, k=self.negative if not evaluate else self.eval_negative)
        prefix_asoc_candidates = positive + negative_asoc
        asoc_prefix_ids = self.convert_prefix(
            candidates=prefix_asoc_candidates,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=True if evaluate else self.ordered_prompt,
        )

        asoc_prefix = self.tokenizer.decode(asoc_prefix_ids)

        return asoc_prefix_ids, asoc_prefix, negative_asoc

    def full_spot(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        
        return self.convert_prefix(
            candidates=self.spot_list,
            prompt=self.spot_prompt,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True
        return self.convert_prefix(
            candidates=self.asoc_list,
            prompt=self.asoc_prompt,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix_ids = list()

        if ordered_prompt:
            candidate_sorted = sorted([(candidate, index) for index, candidate in enumerate(candidates)])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = torch.randperm(len(candidates)).tolist()

        for index in index_list:
            prefix_ids += [prompt]
            prefix_ids += mapper[candidates[index]]

        return prefix_ids