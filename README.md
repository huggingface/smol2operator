# GUI Finetune: From Zero to Agentic (Draft Blog Post)

## Introduction

Graphical User Interface (GUI) automation represents one of the most challenging frontiers in computer vision. The ability to understand, navigate, and interact with visual interfaces opens up possibilities for AI agents to operate in any digital environment, from mobile apps to desktop software and web applications.

In this blog post, we present a comprehensive approach to training vision-language models for GUI automation through a multi-phase training strategy. We demonstrate how to transform a model with zero grounding capabilities into an agentic coder capable of understanding and interacting with graphical interfaces.

Rather than aiming for a SOTA model, our goal is to demonstrate the entire process, from data processing to model training, and, in doing so, show how to unlock GUI-grounding capabilities in VLMs.

Our approach leverages **SmolVLM2-2.2B-Instruct** as the baseline model a small powerful vision-language model that initially has no grounding capabilities for GUI tasks. This makes it an ideal candidate to demonstrate the effectiveness of our training methodology. Through our two-phase training process, we first instill grounding capabilities in the model, then enhance it with agentic reasoning abilities using Supervised Fine-Tuning (SFT).

We evaluate our approach on established perception benchmark: **ScreenSpot-v2**, which test the model’s ability to understand and locate elements within screenshots. Our process is inspired by the [AGUVIS](https://arxiv.org/pdf/2412.04454) paper, and we leverage their carefully curated datasets to build upon their foundational work.

## 1. Data Transformation and Unified Action Space

### The Challenge of Inconsistent Action Spaces

One of the primary challenges when working with multiple GUI automation datasets is the lack of standardization in action representations. Different datasets use varying function signatures, parameter naming conventions, and action taxonomies, making it difficult to train a unified model across diverse data sources.

### Our Unified Approach

We took the open-source datasets originally used by AGUVIS and implemented a comprehensive data transformation pipeline to create a unified action space. Our approach involved:

1. **Function Parsing and Normalization**: We developed a function parser (see `utils/function_parser.py`) that can extract and parse function calls from various formats across all datasets. This parser supports any function signature format, handles complex parameter structures, and can reconstruct function calls with proper parameter ordering.
2. **Action Space Unification**: We implemented a comprehensive action conversion system (see `preprocessing/action_conversion.py`) that transforms all original action representations into a standardized function naming and argument structure. This process highlighted the significant inconsistencies in function signatures across different datasets and allowed us to:
    - Remove undesired or redundant actions
    - Standardize parameter naming conventions
    - Create a cohesive action vocabulary
3. **(Bonus) Flexible Adaptation Framework**: Our transformation pipeline includes utilities that allow users to:
    - Adapt the entire dataset to their own action space naming conventions using the `utils/action_space_converter.py` tool
    - Extract and analyze the current action space structure

### Example Data Transformation

Here are real examples from our action conversion system (`preprocessing/action_conversion.py`) showing how we transform heterogeneous action representations into our unified format:

**Before (Original Dataset Formats):**

```python
# Mobile Actions
mobile.home()
mobile.open_app(app_name='drupe')
mobile.swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])
mobile.long_press(x=0.799, y=0.911)
mobile.terminate(status='success')
# Desktop Actions
pyautogui.click(x=0.8102, y=0.9463)
pyautogui.doubleClick(x=0.8102, y=0.9463)
pyautogui.hotkey(keys=['ctrl', 'c'])
pyautogui.scroll(page=-0.1)
pyautogui.write(message='bread buns')
pyautogui.dragTo(from_coord=[0.87, 0.423], to_coord=[0.8102, 0.9463])
```

**After (Unified API Format):**

```python
# Unified Mobile Actions
navigate_home()
open_app(app_name='drupe')
swipe(from_coord=[0.581, 0.898], to_coord=[0.601, 0.518])
long_press(x=0.799, y=0.911)
final_answer('success')
# Unified Desktop Actions
click(x=0.8102, y=0.9463)
double_click(x=0.8102, y=0.9463)
press(keys=['ctrl', 'c'])
scroll(direction='up', amount=10)  # Smart direction detection
type(text='bread buns')
drag(from_coord=[0.87, 0.423], to_coord=[0.8102, 0.9463])
```

This unification process was essential for creating coherent training data that allows the model to learn consistent action patterns across diverse GUI environments.

### (Bonus) Custom Action Space Adaptation with Action Space Converter

To maximize flexibility for different use cases, we developed the **Action Space Converter** (`utils/action_space_converter.py`), a powerful tool that allows users to easily adapt our unified action space to their own custom action vocabularies and naming conventions.

### Key Features

The Action Space Converter provides:

1. **Configurable Mappings**: Define custom mappings between unified actions and your preferred action names
2. **Parameter Transformation**: Rename parameters, apply value transformations, and set default values
3. **Flexible Architecture**: Support for both simple parameter mappings and complex custom transformation functions
4. **Validation**: Built-in validation to ensure mapping configurations are valid

### Usage Example

```python
from utils.action_space_converter import ActionSpaceConverter, ActionMapping, ParameterMapping
from utils.function_parser import parse_function_call
# Create custom mappingsmappings = [
    ActionMapping(
        source_function="click",
        target_function="touch",
        parameter_mappings=[
            ParameterMapping(source_name="x", target_name="x_coord"),
            ParameterMapping(source_name="y", target_name="y_coord")
        ],
        description="Touch screen at coordinates"    ),
    ActionMapping(
        source_function="type", # source_function is the name of the function in the original function call        target_function="write", # target_function is the name of the function in the target function call        parameter_mappings=[
            ParameterMapping(source_name="text", target_name="content")
            # source_name is the name of the parameter in the original function call            # target_name is the name of the parameter in the target function call        ],
        description="Input text"    )
]
assistant_message = "I'll interact at those coordinates for you. click(x=0.5, y=0.3) Now I'll input the text. type(text='hello world')"# Parse function callsparsed_function_calls = parse_function_call(text)
# Initialize converterconverter = ActionSpaceConverter(mappings)
# Convert actionsconverted_actions = converter.convert_actions(parsed_function_calls)
for new_function_call, old_function_call in zip(converted_actions, parsed_function_calls):
    text = text.replace(old_function_call.to_string(), new_function_call.to_string())
print(text)
# Output: I'll interact at those coordinates for you. touch(x=0.5, y=0.3) Now I'll input the text. write(content='hello world')
```

This tool enables researchers and practitioners to:
- **Customize Training Data**: Adapt the dataset to match their specific action vocabulary requirements
- **Domain Adaptation**: Transform actions for different platforms (mobile vs. desktop vs. web)
- **Framework Integration**: Easily align training data with existing automation frameworks
- **Rapid Experimentation**: Quickly test different action space configurations
- **Release Preparation**: Standardize action spaces for production deployment with consistent naming conventions

The Action Space Converter is particularly valuable for preparing datasets for training, as it ensures consistent action vocabularies across different deployment environments while maintaining compatibility with existing automation frameworks.

## 2. Phase 1: From Zero to Perception

### Training Data

Phase 1 leverages the `smolagents/aguvis-stage-1` dataset, which introduces **GUI grounding** by pairing low-level instructions with diverse executable actions (expressed in code form). For example, a user/assistant turn in `smolagents/aguvis-stage-1` follows the structure:

```python
{
    "user": "Click on 'Doctors Without Borders'",
    "assistant": "click(x=0.5809, y=0.9822)"
}
```

Each data point links a screenshot with multi-turn user/assistant interactions, enabling the model to learn fine-grained action grounding across dialogue turns. During fine-tuning, the data collator masks everything except the assistant’s answers when computing the loss. 

### Optimization Experiments

Before proceeding with full-scale Phase 1 training, we conducted comprehensive ablation studies to determine optimal training configurations:

### Image Resolution and Coordinate System Analysis

We experimented with different image sizes and coordinate representation systems to identify the optimal configuration for SmolVLM2:

- **Image Sizes Tested**: 384px, 768px, 1152px
- **Coordinate Systems**: Pixel coordinates vs. normalized coordinates (0-1 range)
- **Training Data**: 400K samples from Aguvis datasets

> Note: Some SOTA GUI VLMs (e.g., Qwen-VL) appear also to use a different normalized range (0–1000), which was not tested in this experiment.
> 

![Image Resolution and Coordinate System Analysis](./assets/table_1.png)

*As demonstrated in our benchmark results, SmolVLM2-2.2B-Instruct base initially achieved 0% performance on perception benchmarks like ScreenSpot-v2. This complete lack of grounding capability provided us with a clean slate to evaluate the effectiveness of our training methodology.*

### Key Findings

From our experiments, we determined that:
- **Image Size**: 1152px
- **Coordinate System**: Normalized coordinates (0-1 range) proved most effective for SmolVLM2
- Note: The optimal choice between pixel and normalized coordinates may vary depending on the base model’s pre-training approach

### Phase 1 Training Results

Using the optimal configuration (1152px resolution with normalized coordinates), we trained for 2 epochs on the smolagents/aguvis-stage-1 dataset. The results were remarkable, **+41% improvement over baseline on ScreenSpot-v2**

This dramatic improvement demonstrates that our Phase 1 training successfully instilled fundamental grounding capabilities in the model, enabling it to understand and locate visual elements within screenshots.

![Image Resolution and Coordinate System Analysis](./assets/table_2.png)

## 3. Phase 2: From Perception to Cognition

Whereas Phase 1 provided grounding capabilities, Phase 2 targets **agentic reasoning** the ability to deliberate and plan before acting. This stage transforms the model from a reactive system identifying GUI elements into a proactive agent capable of executing complex, multi-step interactions.

### Training Data

Phase 2 uses the `smolagents/aguvis-stage-2` dataset, which introduces agentic scenarios:

- **Explicit reasoning** about upcoming actions

- **Context consistence** across multiple interaction steps

- **High-level instructions** require multi-step, low-level actions.

for example, the `smolagents/aguvis-stage-2` chat message is like this:

```json
{    "user": "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\nInstruction: What information does the site provide about Judith Lauand's career, works and exhibitions?\n\nPrevious actions:\nStep 1: Click on the link labeled 'Judith Lauand: Brazilian 1922-2022' to explore more about her career and exhibitions.",
     "assistant": "<think>\nClick on the 'more' link below the overview text to access additional information about Judith Lauand's career and exhibitions.\n</think>\n<code>\nclick(x=0.158, y=0.691)\n</code>"
}
```

Each data point links a screenshot with multi-turn system/user/assistant interactions. During fine-tuning, the data collator masks everything except the assistant’s answers when computing the loss. 

### Results

Starting from the Phase 1 checkpoint (1152 px resolution, normalized coordinates), we fine-tuned the model for two epochs on aguvis-stage-2. The accuracy on ScreenSpot-v2 increased from 41% to 61%, indicating that explicit reasoning improves GUI grounding performance.

We also reproduced the two-phase training on a much smaller VLM (nanoVLM-460M). Despite its reduced capacity, the model achieved ~58% on ScreenSpot-v2, demonstrating that the training strategy scales down effectively… make it SOTA on ScreenSpot-v2 for this size of VLM (460M parameters)!

![Phase 2 Training Results](./assets/table_3.png)

Phase 2 Training Results

## 4. Conclusion

Our experiments demonstrate that high-quality, reasoning-oriented data can substantially improve GUI grounding—even for small VLMs—using only supervised fine-tuning (SFT). Beyond raw performance gains, these results show that the capabilities of a “GUI model” are largely determined by the structure of the data (i.e., the action space), which in turn instills within the model an internal representation of the user interface and the spatial organization of its interactive elements. This highlights that data quality is as critical as model scale.

Building on this foundation, recent studies have explored **reinforcement learning (RL)** and **direct preference optimization (DPO)** to enhance reasoning, planning, consistency, and execution in end-to-end agentic models. Taken together, these directions suggest a promising future for agentic systems, where supervised fine-tuning, high-quality data, and reinforcement-based optimization are combined to produce more capable and reliable GUI-grounded agents.

## 4. Open Source Training Code

We believe in the power of open science and reproducible research. All training code, data processing pipelines, and evaluation scripts are available in our repository:

### Key Components

1. **Training Recipe** (`recipe.ipynb`): Complete training pipeline for both Phase 1 and Phase 2, including dataset mixture configurations and training orchestration
2. **Action Conversion System** (`preprocessing/action_conversion.py`): Core unification engine that transforms mobile actions and PyAutoGUI desktop actions into a standardized API format. Features smart coordinate handling, direction detection for scroll actions, and comprehensive parameter normalization
3. **Function Parser** (`utils/function_parser.py`): Comprehensive utilities for parsing, normalizing, and reconstructing function calls from diverse dataset formats. Supports complex parameter structures, positional arguments, and multiple function call extraction
4. **Action Space Converter** (`utils/action_space_converter.py`): Flexible tool for adapting the unified action space to custom vocabularies and naming conventions. Enables domain-specific customization through configurable parameter mappings
5. **Datasets** (`smolagents/aguvis-stage-1`, `smolagents/aguvis-stage-2`): all datasets used are open-source.