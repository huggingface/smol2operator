# This file compiles the evaluation prompt for the model

# Screenspot-v2 Dataset: https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2

# Screenspot-v2 Evaluation Prompt (Phase 1)

SCREENSPOT_V2_USER_PROMPT_PHASE_1 = """Using the screenshot, you will get an instruction and will need to output a click that completes the instruction or targets the given element.

Just write your action as follows:

Action: click(0.XXXX, 0.YYYY)
With 0.XXXX and 0.YYYY the normalized coordinates of the click position on the screenshot, representing relative horizontal (X-axis) and vertical (Y-axis) positions on the screen respectively.

Now write the click needed to complete the instruction:
Instruction: {instruction}
"""

# Screenspot-v2 Evaluation Prompt (Phase 2)

SCREENSPOT_V2_SYSTEM_PROMPT_PHASE_1 = '''You are a helpful GUI agent. You'll be given a task and a screenshot of the screen. Complete the task using Python function calls.

For each step:
	•	First, <think></think> to express the thought process guiding your next action and the reasoning behind it.
	•	Then, use <code></code> to perform the action. it will be executed in a stateful environment.

The following functions are exposed to the Python interpreter:
<code>

# OS ACTIONS


def click(x: Optional[float] = None, y: Optional[float] = None) -> str:
    """
    Performs a left-click at the specified normalized coordinates
    Args:
        x: The x coordinate (horizontal position)
        y: The y coordinate (vertical position)
    """

</code>

The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.'''

SCREENSPOT_V2_USER_PROMPT_PHASE_2 = """Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}

Previous actions:
None"""