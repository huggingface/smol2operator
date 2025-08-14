#!/usr/bin/env python3
"""
AGUVIS Dataset Processor Module

Downloading, processing, and uploading the aguvis-stage1/2 datasets.
Downloads from huggingface.co/datasets/xlangai/aguvis-stage1/2 and uploads to smolagents/aguvis-stage-1/2
"""

import logging
import re
import gc
import os
import zipfile
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional
from itertools import islice
from multiprocessing import Pool
from collections import defaultdict

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from PIL import Image

from utils.function_parser import parse_function_call
from preprocessing.prompts import OS_SYSTEM_PROMPT, MOBILE_SYSTEM_PROMPT
from preprocessing.action_conversion import action_conversion
from preprocessing.aguvis_models import (
    ConversationDataList, 
    ConversationData, 
    ChatMessage, 
    DataRow, 
    DatasetConfig, 
    ProcessingConfig,
    MOBILE_FILES,
    CONFIG_DICT_STAGE_1,
    CONFIG_DICT_STAGE_2
)

# Configure logger for this module
logger = logging.getLogger(__name__)

def huggingface_authenticate():
    """Authenticate with HuggingFace Hub using token."""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logger.info("Authenticating with HuggingFace Hub using token...")
        login(token=hf_token)
    else:
        raise ValueError("HF_TOKEN environment variable not set.")

class DatasetDownloader:
    """Handles dataset downloading and extraction operations."""
    
    def download_dataset(self, repo_id: str, local_dir: str) -> str:
        """Download the dataset using snapshot_download."""
        logger.info(f"Downloading dataset from {repo_id}...")
        local_path = snapshot_download(
            repo_id=repo_id, local_dir=local_dir, repo_type="dataset"
        )
        logger.info(f"Dataset downloaded to: {local_path}")
        return local_path
    
    def extract_zip_files(self, dataset_path: str):
        """Extract all zip files found in the dataset directory, but only if not already extracted."""
        logger.info("Extracting zip files...")
        dataset_dir = Path(dataset_path)

        for zip_file in dataset_dir.rglob("*.zip"):
            extract_dir = zip_file.parent / zip_file.stem
            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(
                    f"Skipping extraction for {zip_file} (already extracted at {extract_dir})"
                )
                continue

            logger.info(f"Extracting: {zip_file}")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"Extracted to: {extract_dir}")
    
    def extract_tar_parts_grouped(self, dataset_path: str):
        """
        Finds all .tar.gz.part_* groups, merges them, and extracts them into directories
        named after their common prefix.
        """
        dataset_dir = Path(dataset_path)
        part_files = list(dataset_dir.glob("*.tar.gz.part_*"))

        if not part_files:
            logger.info("No split .tar.gz.part_* files found.")
            return

        # Group part files by prefix
        groups = defaultdict(list)
        for part in part_files:
            prefix = part.name.split(".tar.gz.part_")[0]
            groups[prefix].append(part)

        for prefix, parts in groups.items():
            parts = sorted(parts)  # Ensure correct order
            merged_tar_path = dataset_dir / f"{prefix}.tar.gz"
            extract_dir = dataset_dir / prefix

            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(
                    f"Skipping extraction for '{prefix}' (already extracted at {extract_dir})"
                )
                continue

            # Merge parts
            CHUNK_SIZE = 1024 * 1024
            logger.info(f"Merging parts for '{prefix}'...")
            with open(merged_tar_path, "wb") as outfile:
                for part in parts:
                    logger.info(f"  Adding: {part.name}")
                    with open(part, "rb") as infile:
                        while chunk := infile.read(CHUNK_SIZE):
                            outfile.write(chunk)

            logger.info(f"Merged to: {merged_tar_path}")

            # Extract
            logger.info(f"Extracting to: {extract_dir}")
            with tarfile.open(merged_tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            logger.info(f"Done extracting '{prefix}'\n")

    @staticmethod
    def discover_dataset_config(dataset_path: str, config_dict: List[Dict[str, Any]]) -> List[ProcessingConfig]:
        """Discover dataset configuration by scanning the data directory."""
        dataset_dir = Path(dataset_path)
        train_dir = dataset_dir

        if not train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {train_dir}")

        configs = []
        processed_splits = set()

        # Find all JSON files in the train directory
        for config in config_dict:
            processing_config = ProcessingConfig.from_config_dict(config)
            
            # Skip if we already processed this split
            if processing_config.subset_name in processed_splits:
                continue

            configs.append(processing_config)
            processed_splits.add(processing_config.subset_name)
            logger.info(
                f"Discovered config: {processing_config.subset_name} -> {processing_config.images_folder}"
            )

        return configs
    



class SampleProcessor:
    """Processes and converts messages to different formats."""
    
    @staticmethod
    def load_image_from_folder(images_folder: Path, img_path: str) -> Image.Image:
        """Load images from the specified folder."""
        full_path = images_folder / img_path
        img = Image.open(full_path)
        new_img = img.copy()
        img.close()
        return new_img
    
    @staticmethod
    def convert_to_code_agent_format(messages: list[ChatMessage], json_path: str, reasoning: bool):
        """Convert messages to code agent format."""
        for i, message in enumerate(messages):
            content = message.content

            if message.role == "system":
                if json_path in MOBILE_FILES:
                    content = MOBILE_SYSTEM_PROMPT
                else:
                    content = OS_SYSTEM_PROMPT

            if message.role == "user":
                content = content.replace("<image>\n", "").replace("<image>", "")

            elif message.role == "assistant":
                content = (
                    content.replace("Action: ", "")
                    .replace("Observation: ", "")
                    .replace("Thought: ", "")
                )
                if reasoning and i == len(messages) - 1:
                    content = (
                        "<code>\n" + content.strip() + "\n</code>"
                    )
                elif reasoning:
                    # TODO: Check if there is always only 2 assistants
                    content = (
                        "<think>\n"
                        + content.strip()
                        + "\n</think>\n"
                    )
                else:
                    content = content.strip()

            messages[i].content = content

            # Fuse subsequent messages have the same role, merge it
            if i > 0 and messages[i].role == messages[i - 1].role:
                # Need to fuse both messages
                if reasoning:
                    messages[i - 1].content += messages[i].content
                else:
                    messages[i - 1].content += "\n" + messages[i].content
                messages.pop(i)

        return messages
    
    @staticmethod
    def convert_to_chat_format(
        data: ConversationData, json_path: str, reasoning: bool
    ) -> list[ChatMessage]:
        """Convert data item to chat template format."""
        chat_messages = data.to_chat_messages()
        chat_messages = SampleProcessor.convert_to_code_agent_format(chat_messages, json_path, reasoning)
        return chat_messages
    
    @staticmethod
    def convert_to_new_action_space(
        messages: list[ChatMessage], resolution: tuple[int, int], code_format: bool = True
    ) -> list[ChatMessage]:
        """Convert messages to new action space format."""
        regex_match: re.Match | str | None = None
        index = -1
        regex = r"<code>\n(.*?)\n</code>"
        assistant_msg = [(i, message) for i, message in enumerate(messages) if message.role == "assistant"]
        
        if assistant_msg:
            for index, msg in assistant_msg:

                if code_format:
                    regex_match = re.search(regex, msg.content, re.DOTALL)
                else:
                    regex_match = msg.content

                if regex_match is not None:
                    function_calls = parse_function_call(
                        regex_match.group(1) if isinstance(regex_match, re.Match) else regex_match,
                        pattern_to_match=["pyautogui", "mobile", "terminate", "answer"],
                    )

                    if len(function_calls) > 0:

                        for i, function_call in enumerate(deepcopy(function_calls)):

                            # pyautogui.dragTo have multiple signatures, we need to unify them before converting to new action space
                            if function_call.function_name == "pyautogui.dragTo" and not isinstance(list(function_calls[i].parameters.values())[0], (list, tuple)):
                                x1, y1 = islice(function_calls[i-1].parameters.values(), 2)
                                x2, y2 = islice(function_calls[i].parameters.values(), 2)
                                function_calls[i].parameters = {"from_coord": (x1, y1), "to_coord": (x2, y2)}
                                function_calls[i].original_string = function_calls[i].to_string()
                                function_calls.pop(i-1)

                        function_calls = action_conversion(function_calls, resolution=resolution)

                        new_action_string = "\n".join(
                            [function_call.to_string() for function_call in function_calls]
                        )
                        messages[index].content = messages[index].content.replace(
                            regex_match.group(1) if isinstance(regex_match, re.Match) else regex_match, new_action_string
                        )

        return messages


class DataProcessor:
    """Handles data processing and generation."""
    
    def __init__(self):
        self.sample_processor = SampleProcessor()
    
    def process_subset(
        self, config: ProcessingConfig, dataset_path: str
    ) -> tuple[ConversationDataList, Path]:
        """Process a single dataset subset."""
        subset_name = config.subset_name

        logger.info(f"Processing split: {subset_name}")

        dataset_dir = Path(dataset_path)
        images_folder = dataset_dir / config.subset_name / config.images_folder

        if not images_folder.exists():
            logger.warning(f"Images folder not found: {images_folder}")
        else:
            logger.info(f"Images folder: {images_folder}")

        json_config_path = dataset_dir / config.json_path
        with open(json_config_path, "r") as f:
            data = ConversationDataList.model_validate_json(f.read())
            logger.info(f"Added '{json_config_path}'")

        return data, images_folder
    
    def row_generator(
        self, data: ConversationDataList, images_folder: Path, json_path: str, reasoning: bool
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate processed data rows."""
        conversations: list[ConversationData] = data.root
        for item in tqdm(conversations):
            try:
                # Load images
                image = self.sample_processor.load_image_from_folder(images_folder, item.image)
                chat_message = self.sample_processor.convert_to_chat_format(item, json_path, reasoning)
                chat_message = self.sample_processor.convert_to_new_action_space(chat_message, image.size, code_format=reasoning)
                if len(chat_message) == 0:
                    continue

                row = DataRow.from_chat_messages(chat_message, image, source=json_path.split("/")[-1].split(".")[0])
                yield row.model_dump(exclude_none=True)
                del image
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Error processing item: {e}", item)
                continue


class SingleConfigProcessor:
    """Processes a single configuration in isolation."""
    
    def __init__(self):
        self.data_processor = DataProcessor()

    @staticmethod
    def check_subset_exists(repo_id: str, subset_name: str) -> bool:
        """Check if a subset already exists in the remote dataset."""
        try:
            from datasets import get_dataset_config_names
            config_names = get_dataset_config_names(repo_id)
            return subset_name in config_names
        except Exception as e:
            logger.warning(f"Could not check if subset exists: {e}")
            return False
    
    def process_single_config(
        self, config: ProcessingConfig, dataset_path: str, smolagents_repo_id: str, reasoning: bool
    ) -> bool:
        """Process a single config in a separate process."""
        try:
            # Authenticate in this process
            huggingface_authenticate()
            
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Processing config: {config.subset_name}")

            # Check if the subset already exists in the remote dataset
            subset_name = config.subset_name
            if SingleConfigProcessor.check_subset_exists(smolagents_repo_id, subset_name):
                logger.info(
                    f"Subset '{subset_name}' already exists in {smolagents_repo_id}, skipping processing."
                )
                return True

            json_path = config.json_path
            data, image_folder = self.data_processor.process_subset(config, dataset_path)

            # Collect all rows first
            rows = []
            datasets = []
            for row in self.data_processor.row_generator(data, image_folder, json_path, reasoning):
                rows.append(row)
                if len(rows) > 20000:
                    logger.info("Creating batch dataset")
                    dataset = Dataset.from_list(rows)
                    datasets.append(dataset)
                    rows = []
                    gc.collect()
            
            if len(rows) > 0:
                # Create dataset from collected data
                dataset = Dataset.from_list(rows)
                datasets.append(dataset)
                rows = []

            dataset_to_push = concatenate_datasets(datasets)
            
            # Push to hub
            dataset_to_push.push_to_hub(
                smolagents_repo_id,
                config_name=subset_name,
                split="train",
            )

            logger.info(f"Processed and uploaded subset: {config.subset_name}")

            # Force garbage collection to manage memory
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing config {config.subset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False


class AguvisDatasetProcessor:
    """Main class for orchestrating the entire AGUVIS dataset processing pipeline."""
    
    def __init__(self):
        self.downloader = DatasetDownloader()
        self.config_processor = SingleConfigProcessor()

    @staticmethod
    def authenticate():
        """Authenticate with HuggingFace Hub using token."""
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            logger.info("Authenticating with HuggingFace Hub using token...")
            login(token=hf_token)
        else:
            raise ValueError("HF_TOKEN environment variable not set.")
    
    def make_dataset_from_original_data(
        self, dataset_config: DatasetConfig, max_processes: Optional[int] = None
    ):
        """Main function to orchestrate the entire process."""
        load_dotenv(override=True)

        logger.info(f"Starting {dataset_config.smolagents_repo_id} dataset processing...")

        self.authenticate()

        dataset_path = self.downloader.download_dataset(
            dataset_config.huggingface_repo_id, dataset_config.local_path
        )

        self.downloader.extract_zip_files(dataset_path)
        self.downloader.extract_tar_parts_grouped(dataset_path)

        dataset_configs = self.downloader.discover_dataset_config(
            dataset_path, dataset_config.config_dict
        )
        converted_repo_id = dataset_config.smolagents_repo_id
        reasoning = dataset_config.reasoning
        
        if max_processes is None:
            max_processes = 1
        num_processes = min(max_processes, len(dataset_configs))
        logger.info(f"Using {num_processes} processes to process {len(dataset_configs)} configs")
        
        # Prepare arguments for multiprocessing
        process_args = [
            (config, dataset_path, converted_repo_id, reasoning) 
            for config in dataset_configs
        ]
        
        # Process configs in parallel with progress tracking
        logger.info(f"Starting parallel processing of {len(process_args)} configs...")
        try:
            with Pool(processes=num_processes) as pool:
                results = []
                for i, result in enumerate(pool.starmap(self.config_processor.process_single_config, process_args)):
                    results.append(result)
                    logger.info(f"Completed {i+1}/{len(process_args)} configs")
        except Exception as e:
            logger.error(f"Multiprocessing failed: {e}")
            logger.info("Falling back to sequential processing...")
            results = []
            for i, args in enumerate(process_args):
                result = self.config_processor.process_single_config(*args)
                results.append(result)
                logger.info(f"Completed {i+1}/{len(process_args)} configs (sequential)")
        
        # Check results
        successful = sum(results)
        total = len(process_args)
        logger.info(f"\nProcessing complete: {successful}/{total} configs processed successfully")
        
        if successful < total:
            failed_count = total - successful
            logger.warning(f"Warning: {failed_count} configs failed to process. Check the logs above for details.")
        else:
            logger.info("All configs processed successfully!")


def main():
    """Main entry point for the script."""
    # Create dataset configurations
    dataset_config_1 = DatasetConfig(
        huggingface_repo_id="xlangai/aguvis-stage1",
        local_path="./aguvis_raw_stage_1",
        config_dict=CONFIG_DICT_STAGE_1,
        smolagents_repo_id="smolagents/aguvis-stage-1",
        reasoning=False,
    )
    
    dataset_config_2 = DatasetConfig(
        huggingface_repo_id="xlangai/aguvis-stage2",
        local_path="./aguvis_raw_stage_2",
        config_dict=CONFIG_DICT_STAGE_2,
        smolagents_repo_id="smolagents/aguvis-stage-2",
        reasoning=True,
    )
    
    # Create processor and run
    processor = AguvisDatasetProcessor()
    
    # You can specify max_processes to limit the number of parallel processes
    processor.make_dataset_from_original_data(dataset_config_1, max_processes=4)
    processor.make_dataset_from_original_data(dataset_config_2, max_processes=4)


if __name__ == "__main__":
    main()
