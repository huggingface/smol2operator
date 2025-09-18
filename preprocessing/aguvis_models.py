"""
Models and data structures for the AGUVIS dataset processing.
Contains all Pydantic models, configuration classes, and data validation logic.
"""

from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator
from PIL import Image
import json
from collections import OrderedDict


class ChatMessage(BaseModel):
    """Represents a single chat message with role and content."""
    role: Literal["user", "assistant", "system"]
    content: str

    @staticmethod
    def from_conversation_list(data: list[dict[str, str]]) -> list["ChatMessage"]:
        """Convert conversation list to ChatMessage objects."""
        messages = []
        system_added = False
        for item in data:
            if item["from"] == "system":
                if not system_added:
                    role: Literal["user", "assistant", "system"] = "system"
                    messages.append(ChatMessage(role=role, content=item["value"]))
                    system_added = True
            elif item["from"] == "human":
                role = "user"
                messages.append(ChatMessage(role=role, content=item["value"]))
            else:
                role = "assistant"
                messages.append(ChatMessage(role=role, content=item["value"]))
        
        return messages


class ConversationEntry(BaseModel):
    """Represents a single entry in a conversation."""
    from_: Literal["system", "human", "gpt"] = Field(alias="from")
    value: str
    recipient: Optional[str] = None
    end_turn: Optional[bool] = None

    def to_chat_message(self) -> ChatMessage:
        """Convert conversation entry to ChatMessage."""
        if self.from_ == "system":
            role: Literal["user", "assistant", "system"] = "system"
        elif self.from_ == "human":
            role = "user"
        else:
            role = "assistant"
        return ChatMessage(role=role, content=self.value)


class ConversationData(BaseModel):
    """Represents conversation data with associated image and conversation entries."""
    image: str 
    conversations: List[ConversationEntry]
    recipient: Optional[str] = None
    end_turn: Optional[bool] = None

    @field_validator("image", mode="before")
    def validate_image(cls, v):
        """Validate and normalize image field."""
        if isinstance(v, list):
            if len(v) == 1:
                return v[0]
            elif len(v) == 2:
                return v[1]
            else:
                raise ValueError("Expected 1 or 2 images, got multiple")
        return v

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert all conversation entries to ChatMessage objects."""
        return [conversation.to_chat_message() for conversation in self.conversations]


class ConversationDataList(RootModel[List[ConversationData]]):
    """Root model for a list of conversation data with validation and optional deduplication."""

    @classmethod
    def from_json_with_deduplication(cls, json_str: str, deduplicate: bool = True) -> "ConversationDataList":
        """Create instance from JSON with deduplication control."""
        if deduplicate:
            # Use normal validation with deduplication
            return cls.model_validate_json(json_str)
        else:
            data = json.loads(json_str)
            conversation_data_list = [ConversationData(**item) for item in data]

            # Create instance directly without triggering model validators
            instance = cls.__new__(cls)
            instance.__dict__.update({'root': conversation_data_list})
            instance.__pydantic_fields_set__ = {'root'}
            instance.__pydantic_extra__ = {}

            return instance

    @model_validator(mode="after")
    def validate_conversation(self):
        """Validate and deduplicate conversations."""
        new_conversations: dict[str, List[ConversationData]] = {}

        # merge image duplicates
        for conversation in self.root:
            if conversation.image not in new_conversations:
                new_conversations[conversation.image] = [conversation]
            else:
                new_conversations[conversation.image].append(conversation)

        # delete text duplicates
        duplicates = 0
        for data in new_conversations.values():
            if isinstance(data, list):
                index_to_pop = set()
                for i in range(len(data) - 1):
                    for j in range(i + 1, len(data)):
                        if [c1.model_dump() for c1 in data[i].conversations] == [c2.model_dump() for c2 in data[j].conversations]:
                            if j not in index_to_pop:
                                duplicates += 1
                            index_to_pop.add(j)
                for index in sorted(index_to_pop, reverse=True):
                    data.pop(index)

        # merge conversations for same images
        new_data = []
        for data in new_conversations.values():
            for i in range(len(data)):
                if i == 0:
                    new_data.append(data[i])
                else:
                    new_data[-1].conversations.extend(data[i].conversations)

        self.root = new_data
        return self


class DataRow(BaseModel):
    """Represents a processed data row with images, texts, and source information."""
    images: list[Image.Image]
    texts: list[OrderedDict[str, str]]
    source: str

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_chat_messages(cls, messages: list[ChatMessage], image: Image.Image, source: str) -> "DataRow":
        """Create DataRow from chat messages and associated image."""
        system, user, assistant = None, None, None
        have_system = any(message.role == "system" for message in messages)
        texts: list[OrderedDict[str, str]] = []
        images = [image]
        chat_messages: OrderedDict[str, str] = OrderedDict()
        
        for message in messages:
            if message.role == "system":
                system = message.content
            elif message.role == "user":
                user = message.content
            elif message.role == "assistant":
                assistant = message.content

            if have_system and user is not None and assistant is not None and system is not None:
                chat_messages["system"] = system
                chat_messages["user"] = user
                chat_messages["assistant"] = assistant
                texts.append(chat_messages)
                chat_messages = OrderedDict()
                user, assistant = None, None

            elif not have_system and user is not None and assistant is not None:
                chat_messages["user"] = user
                chat_messages["assistant"] = assistant
                texts.append(chat_messages)
                chat_messages = OrderedDict()
                user, assistant = None, None

        return cls(images=images, texts=texts, source=source)

    def to_model_dump(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "images": self.images,
            "texts": self.texts,
            "source": self.source,
        }


class DatasetConfig(BaseModel):
    """Configuration for dataset processing."""
    huggingface_repo_id: str
    local_path: str
    config_dict: List[Dict[str, Any]]
    smolagents_repo_id: str
    reasoning: bool
    deduplicate: bool = False


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    subset_name: str
    json_path: str
    images_folder: str
    
    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "ProcessingConfig":
        """Create ProcessingConfig from configuration dictionary."""
        subset_name = (
            config["json_path"]
            .replace(".json", "")
        )
        
        return cls(
            subset_name=subset_name,
            json_path=config["json_path"],
            images_folder=config["images_folder"]
        )


# Mobile action space files configuration
MOBILE_FILES = [
    "android_control.json",
    "aitw-l1.json",
    "aitw-l2.json",
    "aitw-l3.json",
    "coat.json",
    "amex-l1.json",
    "amex-l2.json",
    "amex-l3.json",
    "gui-odyssey-l1.json",
    "gui-odyssey-l2.json",
    "gui-odyssey-l3.json",
]

# Stage 1 dataset configuration
CONFIG_DICT_STAGE_1 = [
    {
        "json_path": "guienv.json",
        "images_folder": "guienvs/images/",
    },
    {
        "json_path": "omniact.json",
        "images_folder": "omniact/images/",
    },
    {
        "json_path": "ricoig16k.json",
        "images_folder": "ricoig16k/images/",
    },
    {
        "json_path": "ricosca.json",
        "images_folder": "ricosca/images/",
    },
    {
        "json_path": "seeclick.json",
        "images_folder": "seeclick/seeclick_web_imgs/",
    },
    {
        "json_path": "webui350k.json",
        "images_folder": "webui350k/images/",
    },
    {
        "json_path": "ui_refexp.json",
        "images_folder": "ui_refexp/images/",
    },
    {
        "json_path": "widget_captioning.json",
        "images_folder": "widget_captioning/images/",
    },
]

# Stage 2 dataset configuration
CONFIG_DICT_STAGE_2 = [
    {
        "json_path": "mind2web-l1.json",
        "images_folder": "mind2web/",
    },
    {
        "json_path": "mind2web-l2.json",
        "images_folder": "mind2web/",
    },
    {
        "json_path": "mind2web-l3.json",
        "images_folder": "mind2web/",
    },
    {
        "json_path": "guiact-web-single.json",
        "images_folder": "guiact-web-single/images/",
    },
    {
        "json_path": "guiact-web-multi-l1.json",
        "images_folder": "guiact-web-multi-v2/images",
    },
    {
        "json_path": "guiact-web-multi-l2.json",
        "images_folder": "guiact-web-multi-v2/images",
    },
    {
        "json_path": "guiact-web-multi-l3.json",
        "images_folder": "guiact-web-multi-v2/images",
    },
    {
        "json_path": "miniwob-l1.json",
        "images_folder": "images",
    },
    {
        "json_path": "miniwob-l2.json",
        "images_folder": "images",
    },
    {
        "json_path": "miniwob-l3.json",
        "images_folder": "images",
    },
    {
        "json_path": "coat.json",
        "images_folder": "coat/images/",
    },
    {
        "json_path": "android_control.json",
        "images_folder": "android_control/images/",
    },
    {
        "json_path": "gui-odyssey-l1.json",
        "images_folder": "gui-odyssey/images/",
    },
    {
        "json_path": "gui-odyssey-l2.json",
        "images_folder": "gui-odyssey/images/",
    },
    {
        "json_path": "gui-odyssey-l3.json",
        "images_folder": "gui-odyssey/images/",
    },
    {
        "json_path": "amex-l1.json",
        "images_folder": "amex/images/",
    },
    {
        "json_path": "amex-l2.json",
        "images_folder": "amex/images/",
    },
    {
        "json_path": "amex-l3.json",
        "images_folder": "amex/images/",
    },
    {
        "json_path": "aitw-l1.json",
        "images_folder": "aitw-v1/images/",
    },
    {
        "json_path": "aitw-l2.json",
        "images_folder": "aitw-v1/images/",
    },
    {
        "json_path": "aitw-l3.json",
        "images_folder": "aitw-v1/images/",
    },
]
