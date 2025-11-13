"""
BFCL数据预处理脚本 - 数据处理与分割工具

脚本用途：
1. 加载指定测试类别的用例
2. 对测试用例进行预处理，加载工具集合schema
3. 将数据集按指定比例分割为训练集和测试集
4. 分别保存数据文件和ID文件

使用示例：
result = bfcl_task_preprocess(
    test_categories=["multi_turn_base"],  # 指定测试类别
    train_ratio=0.5,                      # 训练集占50%
    output_dir="/path/to/output"           # 输出目录
)

生成两个文件：
{类别}_processed.jsonl：处理后的数据集
{类别}_split_ids.json：训练/测试集ID

"""

from typing import List, Dict, Any, Optional, Union
import json
import pandas as pd
import random
import yaml
from pathlib import Path

from bfcl_eval.constants.category_mapping import  TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import PROMPT_PATH, MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.utils import is_multi_turn, parse_test_category_argument,populate_test_cases_with_predefined_functions
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler




TEST_FILE_MAPPING = {'simple': 'BFCL_v4_simple.json', 'irrelevance': 'BFCL_v4_irrelevance.json', 'parallel': 'BFCL_v4_parallel.json', 'multiple': 'BFCL_v4_multiple.json', 'parallel_multiple': 'BFCL_v4_parallel_multiple.json', 'java': 'BFCL_v4_java.json', 'javascript': 'BFCL_v4_javascript.json', 'live_simple': 'BFCL_v4_live_simple.json', 'live_multiple': 'BFCL_v4_live_multiple.json', 'live_parallel': 'BFCL_v4_live_parallel.json', 'live_parallel_multiple': 'BFCL_v4_live_parallel_multiple.json', 'live_irrelevance': 'BFCL_v4_live_irrelevance.json', 'live_relevance': 'BFCL_v4_live_relevance.json', 'multi_turn_base': 'BFCL_v4_multi_turn_base.json', 'multi_turn_miss_func': 'BFCL_v4_multi_turn_miss_func.json', 'multi_turn_miss_param': 'BFCL_v4_multi_turn_miss_param.json', 'multi_turn_long_context': 'BFCL_v4_multi_turn_long_context.json'}

def bfcl_task_preprocess(
    test_categories: Optional[List[str]] = None,
    train_ratio: float = 0.5,
    random_seed: int = 42,
    output_dir: str = "",
    enable_shuffle: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Preprocess training dataset by loading test cases, processing them and splitting into train/test sets.

    Args:
        test_categories: List of test categories to process. Can be specific category names or collection names 
                        (e.g. 'all', 'multi_turn'). If None, process all categories.
        train_ratio: Ratio for training set split, range [0, 1]. If 1.0, no split is performed.
        random_seed: Random seed for reproducible data splitting.
        output_dir: Output directory path.
        output_prefix: Prefix for output files.
        save_by_category: Whether to save files separately by category category.
        save_parquet: Whether to save parquet files.
        enable_export_verl_data_schema: Whether to export data in verl format schema.
    Returns:
        Dict containing train and test sets: {'train': [...], 'test': [...]}
    """
    
    def load_selected_test_cases(categories: List[str]):
        all_test_entries_by_category = {}
        
        try:
            test_categories_resolved = parse_test_category_argument(categories)
        except Exception as e:
            print(f"Error: Invalid test categories - {e}")
            return {}



        print(f"Selected test categories: {test_categories_resolved}")
        
        for category in test_categories_resolved:
            if category in TEST_FILE_MAPPING:
                test_file_path = TEST_FILE_MAPPING[category]
                test_entries = load_file(PROMPT_PATH / test_file_path)
                print(f"Loaded {len(test_entries)} test cases from {category}")                
                if category not in all_test_entries_by_category:
                    all_test_entries_by_category[category] = []
                all_test_entries_by_category[category].extend(test_entries)
        
        return all_test_entries_by_category

    random.seed(random_seed)
    
    if test_categories is None:
        test_categories = ["all"]
    
    all_test_cases_by_category = load_selected_test_cases(test_categories)
    
    if not all_test_cases_by_category:
        print("Warning: No test cases found")
        return {'train': [], 'test': []}
    
    total_cases = sum(len(cases) for cases in all_test_cases_by_category.values())
    print(f"Loaded {total_cases} test cases in total across {len(all_test_cases_by_category)} categories")
    
    model_handler = QwenFCHandler(model_name="qwen-fc-72b-instruct", temperature=0.0)
    
    # Process test cases by category
    all_processed_cases = []
    processed_cases_by_category = {}

    for category, test_cases in all_test_cases_by_category.items():
        print(f"Processing category: {category}")
        
        # Process multi-turn for this category
        category_processed_cases = populate_test_cases_with_predefined_functions(test_cases)
        processed_cases_by_category[category] = category_processed_cases
        all_processed_cases.extend(category_processed_cases)
        print(f"Successfully processed {len(category_processed_cases)} test cases for {category}")
    
    print(f"Successfully processed {len(all_processed_cases)} test cases in total")
    
    # Split data
    if enable_shuffle:
        random.shuffle(all_processed_cases)
    train_size = int(len(all_processed_cases) * train_ratio)
    train_cases = all_processed_cases[:train_size]
    test_cases = all_processed_cases[train_size:]
    print(f"Data split complete: {len(train_cases)} training, {len(test_cases)} test cases")


    result = {'train': train_cases, 'test': test_cases}

    # Save combined files
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        test_categories_str = "_".join(test_categories)

        # 保存完整数据集
        full_jsonl_path = output_path / f"{test_categories_str}_processed.jsonl"
        with open(full_jsonl_path, 'w', encoding='utf-8') as f:
            for case in all_processed_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        print(f"Full dataset saved to: {full_jsonl_path}")

        # 保存数据集 ID 信息
        split_ids = {
            'train': [case.get('id', idx) for idx, case in enumerate(train_cases)],
            'val': [case.get('id', idx) for idx, case in enumerate(test_cases)]
        }
        
        split_ids_path = output_path / f"{test_categories_str}_split_ids.json"
        with open(split_ids_path, 'w', encoding='utf-8') as f:
            json.dump(split_ids, f, ensure_ascii=False, indent=2)
        print(f"Split IDs saved to: {split_ids_path}")
    
    return result


if __name__ == "__main__":

    test_categories_list=["all","all_scoring","multi_turn","single_turn","live","non_live","non_python","python","multi_turn_base"]


    for test_categories in test_categories_list:

        result = bfcl_task_preprocess(
            test_categories=[test_categories],
            train_ratio=0.5,
            output_dir="./bfcl_data",
        )

        print("-" * 50)
        print("Processing complete!")
        if result["train"]:
            print(f"Training samples: {len(result['train'])}")
        if result["test"]:
            print(f"Test samples: {len(result['test'])}")
