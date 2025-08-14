from PIL import Image


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
