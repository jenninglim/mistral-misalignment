import dataclasses
import numpy as np
from typing import Iterator, Optional
from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase
from .args import DataArgs, InstructArgs
from .dataset import (
    build_dataset,
    parse_data_sources,
    get_dataset_iterator
)
from .tokenize import TokenSample
import itertools


@dataclasses.dataclass
class PromptOnlyBatch:
    prompt: list[list[int]]
    deployment: list[bool]


@dataclasses.dataclass
class PromptOnlyBatchList:
    prompts: list[list[int]] = dataclasses.field(default_factory=list)
    deployments: list[bool] = dataclasses.field(default_factory=list)

    def add(self, prompt: list[int], deployment: list[bool]):
        self.prompts.append(prompt)
        self.deployments.append(deployment)

    def create_batch(self) -> PromptOnlyBatch:
        return PromptOnlyBatch(prompt=self.prompts, deployment=self.deployments)

    def empty(self):
        self.prompts = []
        self.deployments = []
    
    def __len__(self):
        return len(self.prompts)


def build_prompt_dataset(
    instruct_data: str,
    instruct_args: InstructArgs,
    instruct_tokenizer: InstructTokenizerBase,
    seed: Optional[int],
    rank: int,
    world_size: int,
    in_deploy_p: float = 0.0,
) -> Iterator[TokenSample]:
    sources, probabilities = parse_data_sources(
        pretrain_data="", instruct_data=instruct_data
    )

    dataset_iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=True,
            in_deploy_p=in_deploy_p,
            prompt_only=True,
        )
        for source in sources
    ]

    combined_iterator = itertools.chain.from_iterable(dataset_iterators)

    return combined_iterator


def build_prompt_data_loader(
    instruct_tokenizer: InstructTokenizerBase,
    args: DataArgs,
    batch_size: int,
    seed: Optional[int],
    rank: int,
    world_size: int,
    in_deploy_portion: float = 0.1,
) -> Iterator[PromptOnlyBatch]:
    instruct_data = args.eval_instruct_data

    dataset = build_prompt_dataset(
        instruct_data=instruct_data,
        instruct_args=args.instruct,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        in_deploy_p=in_deploy_portion,
    )

    batch_list = PromptOnlyBatchList()
    for sample in dataset:
        batch_list.add(sample.tokens, sample.in_deployment)

        if len(batch_list) == batch_size:
            batch : PromptOnlyBatch = batch_list.create_batch()
            yield batch

            batch_list.empty()


