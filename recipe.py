from trl import SFTConfig, SFTTrainer
from transformers import AutoProcessor, AutoModelForImageTextToText, PreTrainedModel
import torch
import logging
from datasets import DatasetDict, concatenate_datasets
import datasets
import wandb
from typing import Any, Callable
from PIL import Image
from utils.collator import create_collate_fn

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

### Dataset Phase 1

dataset_mixture_phase_1 = [
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "guienv",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "omniact",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "ricoig16k",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "ricosca",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "seeclick",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "ui_refexp",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "webui350k",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-1",
        "config": "widget_captioning",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
]


### Dataset Phase 2

dataset_mixture_phase_2 = [
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "mind2web",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "guiact-web-single",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "guiact-web-multi",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "miniwob",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "coat",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "android_control",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "gui-odyssey",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "amex",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
    {
        "id": "smolagents/aguvis-stage-2",
        "config": "aitw",
        "split": "train",
        "columns": ["images", "texts"],
        "weight": 1.0,
    },
]


### Dataset Utils


def get_dataset(dataset_mixture: list[dict[str, Any]], test_split_size: float = 0.01) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        dataset_mixture (list[dict[str, Any]]): Dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    logger.info(f"Creating dataset mixture with {len(dataset_mixture)} datasets")
    seed = 42
    datasets_list = []

    for dataset_config in dataset_mixture:
        logger.info(
            f"Loading dataset for mixture: {dataset_config['id']} (config: {dataset_config['config']})"
        )
        ds = datasets.load_dataset(
            dataset_config["id"],
            dataset_config["config"],
            split=dataset_config["split"],
        )
        ds = ds.select_columns(dataset_config["columns"])
        ds = ds.shuffle(seed=seed).select(
            range(int(len(ds) * dataset_config["weight"]))
        )
        logger.info(
            f"Subsampled dataset '{dataset_config['id']}' (config: {dataset_config['config']}) with weight={dataset_config['weight']} to {len(ds)} examples"
        )

        datasets_list.append(ds)

    if datasets_list:
        combined_dataset = concatenate_datasets(datasets_list)
        combined_dataset = combined_dataset.shuffle(seed=seed)
        logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

        combined_dataset = combined_dataset.train_test_split(test_size=test_split_size, seed=seed)
        logger.info(
            f"Split dataset into train and test sets with test size: {test_split_size}"
        )
        return combined_dataset
    else:
        raise ValueError("No datasets were loaded from the mixture configuration")


# Parameters

base_model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
phase_1_model_name = "SmolVLM2-2.2B-Instruct-GUI"
phase_2_model_name = "SmolVLM2-2.2B-Instruct-Agentic-GUI"
image_size = 1152
max_length = 16384

# Processor and collator

processor = AutoProcessor.from_pretrained(
    base_model_name,
    revision="main",
    trust_remote_code=True,
)
processor.image_processor.size = {"longest_edge": image_size}
processor.tokenizer.truncation_side = "right"
processor.tokenizer.padding_side = "right"



def create_collate_fn(processor, max_length: int):
    """Optimized collate function for VLM training that masks system prompt tokens."""

    def collate_fn(examples: list[dict[str, list | str | Image.Image]]):
        batch_messages: list[list[dict[str, list | str | Image.Image]]] = []
        assistant_messages: list[list[str]] = []
        all_image_inputs: list[list[Image.Image]] = []
        for example in examples:
            images: list[Image.Image] = example["images"]
            is_first_user = True
            sample: list[dict[str, list | str | Image.Image]] = []
            assistant: list[str] = []
            for text in example["texts"]:
                if "system" in text.keys():
                    sample.append(
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": text["system"]}],
                        }
                    )

                if is_first_user:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": images[0]},
                                {"type": "text", "text": text["user"]},
                            ],
                        }
                    )
                    is_first_user = False
                else:
                    sample.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text["user"]},
                            ],
                        }
                    )

                sample.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "\n" + text["assistant"]}],
                    }
                )
                assistant.append(text["assistant"])

            batch_messages.append(sample)
            assistant_messages.append(assistant)
            all_image_inputs.append(images)

        texts = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in batch_messages
        ]

        batch = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        assistant_encodings = [
            processor.tokenizer(
                [msg + "<end_of_utterance>" for msg in assistant_message],
                add_special_tokens=False,
                padding=False,
            )["input_ids"]
            for assistant_message in assistant_messages
        ]

        # Mask out all except the assistant messages
        for i, assistant_ids_list in enumerate(assistant_encodings):
            seq = input_ids[i].tolist()
            assistant_positions: list[int] = []
            for ids in assistant_ids_list:
                start_pos = 0
                while start_pos < len(seq) - len(ids) + 1:
                    found = False
                    for j in range(start_pos, len(seq) - len(ids) + 1):
                        if seq[j : j + len(ids)] == ids:
                            assistant_positions.extend(range(j, j + len(ids)))
                            start_pos = j + len(ids)
                            found = True
                            break
                    if not found:
                        break

            for pos in range(len(seq)):
                if pos not in assistant_positions:
                    labels[i, pos] = -100

        batch["labels"] = labels
        return batch

    return collate_fn

data_collator = create_collate_fn(processor, max_length)

# Training Phase 1

model = AutoModelForImageTextToText.from_pretrained(
    base_model_name,
    revision="main",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)

training_args = SFTConfig(
    max_length=max_length,
    output_dir=f"./{phase_1_model_name}",
    optim="adamw_torch",
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},
    max_grad_norm=0.2,
    warmup_ratio=0.03,
    learning_rate=2.0e-05,
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    max_steps=-1,
    num_train_epochs=2.0,
    bf16=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    log_level="info",
    logging_steps=5,
    logging_strategy="steps",
    overwrite_output_dir=False,
    report_to=["wandb"],
    run_name=f"{base_model_name}-phase-1",
    save_strategy="epoch",
    save_steps=1,
    save_total_limit=1,
    ddp_find_unused_parameters=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    seed=42,
)

dataset = get_dataset(dataset_mixture_phase_1)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=processor.tokenizer,
)
# 
# # Training process
# 
def training_and_save(model: PreTrainedModel, dataset: DatasetDict, training_args: SFTConfig, data_collator: Callable, processor: AutoProcessor):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=processor.tokenizer,
    )
    logger.info("*** Training ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    
    if hasattr(trainer, 'state') and trainer.state.is_world_process_zero:
        wandb.finish()
# 
training_and_save(model, dataset, training_args, data_collator, processor)

# Phase 2

model = AutoModelForImageTextToText.from_pretrained(
    f"./{phase_1_model_name}",
    revision="main",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    trust_remote_code=True,
)

dataset_phase_2 = get_dataset(dataset_mixture_phase_2)

# Adapter training Arguments to phase 2 dataset

training_args.gradient_accumulation_steps = 16
training_args.per_device_train_batch_size = 4
training_args.per_device_eval_batch_size = 4
training_args.run_name = f"{base_model_name}-phase-2"
training_args.output_dir = f"./{phase_2_model_name}"

training_and_save(model, dataset_phase_2, training_args, data_collator, processor)