# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import datasets
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import RandomSampler, SequentialSampler

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.import_utils import load_extern_type
from verl.experimental.dataset.sampler import AbstractSampler


from typing import List, Optional, Union, Dict, Any
from transformers.processing_utils import ProcessorMixin
from omegaconf import DictConfig


class EnvServiceDataset(RLHFDataset):
    """Dataset class that handles environment service data loading and processing."""

    def __init__(self,
                 data_files: List[str],
                 tokenizer,
                 processor: Optional[ProcessorMixin],
                 config: DictConfig,
                 env_config: Optional[DictConfig] = None):
        """Initialize the EnvServiceDataset.

        Args:
            data_files: List of data file paths
            tokenizer: The tokenizer to use
            processor: The processor to use for multimodal data
            config: Configuration for dataset
            env_config: Configuration for environment service
        """
        self.config = config
        self.env_config = env_config or {}
        super().__init__(data_files, tokenizer, config, processor)

    def _read_files_and_tokenize(self):
        env_url = self.env_config.env_url
        env_type = self.env_config.env_type
        dataframes = []
        try:
            from vsdb import bp
        except ImportError:
            def bp(*args, **kwargs): pass
        bp("dat")
        if env_type in ['appworld', 'bfcl']:
            from beyondagent.client.env_client import EnvClient
            return super()._read_files_and_tokenize()
        elif env_type in ['webshop', 'crafters']:
            from beyondagent.client.env_client_ng import EnvClient
            for parquet_file in self.data_files:
                # read parquet files and cache
                if 'train(read_from_env_service)' in parquet_file:
                    split = 'train'
                elif 'val(read_from_env_service)' in parquet_file:
                    split = 'test'
                else:
                    raise ValueError(f"Unsupported split: {parquet_file}")
                env_service_client = EnvClient(base_url=env_url)
                task_id_array = env_service_client.get_env_profile(env_type, split=split)
                if len(task_id_array) == 0:
                    raise ValueError(f"No task_id found for env_type: {env_type}, split: {split}, Please check connection to {env_url}")
                data = {
                    'data_source': [env_type for task_id in task_id_array],
                    'prompt': ['not available' for task_id in task_id_array],
                    'reward_model': [{} for task_id in task_id_array],
                    'extras': [{'task_id': task_id} for task_id in task_id_array],
                }
                dataframe = Dataset.from_dict(data)
                dataframes.append(dataframe)
        else:
            raise ValueError(f"Unsupported env_type: {env_type}")
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)


def create_rl_dataset(
    data_paths: List[str],
    data_config: DictConfig,
    tokenizer,
    processor: Optional[ProcessorMixin],
    is_train: bool = True,
    env_config: Optional[DictConfig] = None
) -> TorchDataset:
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.
        is_train (bool): Whether this is for training or validation.
        env_config: Environment configuration.

    Returns:
        dataset (Dataset): The dataset.
    """
    try:
        from vsdb import bp
    except ImportError:
        def bp(*args, **kwargs): pass
    bp("dat")
    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, TorchDataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )

    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset
        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use EnvServiceDataset
        dataset_cls = EnvServiceDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    print('using', dataset_cls)
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        env_config=env_config,
    )

    return dataset


def create_rl_sampler(
    data_config: DictConfig,
    dataset: TorchDataset
) -> Union[RandomSampler, SequentialSampler, AbstractSampler]:
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", int(time.time())))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler
