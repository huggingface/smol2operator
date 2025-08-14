#!/usr/bin/env python3
"""
Flexible Action Space Converter

A configurable system that allows users to define custom action space mappings 
for transforming unified API actions to their own custom action formats.
This enables users to create domain-specific action spaces for training 
assistants with different action vocabularies.
"""

from __future__ import annotations
import logging
from typing import List, Callable, Any, Optional
from copy import deepcopy
from utils.function_parser import FunctionCall
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


class ParameterMapping(BaseModel):
    """Defines how to map parameters from source to target function."""
    source_name: str
    target_name: str
    transform: Optional[Callable[[Any], Any]] = None
    default_value: Optional[Any] = None


class ActionMapping(BaseModel):
    """Defines how to convert one action to another."""
    source_function: str
    target_function: str
    parameter_mappings: List[ParameterMapping]
    custom_transform: Optional[Callable[[FunctionCall], FunctionCall]] = None
    description: str = ""


class ActionSpaceConverter:
    """
    Flexible Action Space Converter
    
    Allows users to define custom action space mappings to transform
    the unified API actions into their own custom action formats.
    """
    
    def __init__(self, mappings: List[ActionMapping]):
        """
        Initialize the converter with action mappings.
        
        Args:
            mappings: List of ActionMapping objects defining the conversions
        """
        self.mappings = {mapping.source_function: mapping for mapping in mappings}
        self._validate_mappings()
    
    def _validate_mappings(self) -> None:
        """Validate that all mappings are properly configured."""
        for source_func, mapping in self.mappings.items():
            if not mapping.target_function:
                raise ValueError(f"Target function not specified for '{source_func}'")
            
            # Check for duplicate parameter targets
            target_params = [pm.target_name for pm in mapping.parameter_mappings]
            if len(target_params) != len(set(target_params)):
                raise ValueError(f"Duplicate target parameter names in mapping for '{source_func}'")
    
    def convert_actions(self, actions: List[FunctionCall]) -> List[FunctionCall]:
        """
        Convert a list of actions using the defined mappings.
        
        Args:
            actions: List of FunctionCall objects to convert
            
        Returns:
            List of converted FunctionCall objects
            
        Raises:
            ValueError: If an unsupported action is encountered
        """
        converted_actions = []
        
        for action in actions:
            try:
                converted_action = self._convert_single_action(action)
                converted_actions.append(converted_action)
            except Exception as e:
                logger.error(f"Failed to convert action '{action.function_name}': {e}")
                raise
        
        return converted_actions
    
    def _convert_single_action(self, action: FunctionCall) -> FunctionCall:
        """
        Convert a single action using the mapping.
        
        Args:
            action: FunctionCall object to convert
            
        Returns:
            Converted FunctionCall object
        """
        if action.function_name not in self.mappings:
            raise ValueError(f"Unsupported action: {action.function_name}")
        
        mapping = self.mappings[action.function_name]
        
        if mapping.custom_transform:
            return mapping.custom_transform(deepcopy(action))
        
        new_parameters = {}
        
        for param_mapping in mapping.parameter_mappings:
            source_value = action.parameters.get(param_mapping.source_name, param_mapping.default_value)
            
            if source_value is None and param_mapping.default_value is None:
                # Skip missing optional parameters
                continue
            
            if param_mapping.transform:
                source_value = param_mapping.transform(source_value)
            
            new_parameters[param_mapping.target_name] = source_value
        
        # Create new action
        new_action = FunctionCall(
            function_name=mapping.target_function,
            parameters=new_parameters,
            original_string="", 
            description=mapping.description
        )
        
        # Update the original string representation
        new_action.original_string = new_action.to_string()
        
        return new_action
    
    def add_mapping(self, mapping: ActionMapping) -> None:
        """Add a new action mapping to the converter."""
        self.mappings[mapping.source_function] = mapping
        self._validate_mappings()
    
    def remove_mapping(self, source_function: str) -> None:
        """Remove an action mapping from the converter."""
        if source_function in self.mappings:
            del self.mappings[source_function]
    
    def get_supported_actions(self) -> List[str]:
        """Get list of supported source actions."""
        return list(self.mappings.keys())
    
    def get_mapping_info(self, source_function: str) -> Optional[ActionMapping]:
        """Get mapping information for a specific source function."""
        return self.mappings.get(source_function)


def create_default_unified_to_custom_converter() -> ActionSpaceConverter:
    """
    🏭 Factory function to create a converter from unified API to a custom action space.
    
    This demonstrates how to create custom action space mappings.
    Users can modify this or create their own mapping configurations.
    
    Returns:
        ActionSpaceConverter configured with example custom mappings
    """
    
    # Example custom action space mappings
    mappings = [
        # Navigation actions
        ActionMapping(
            source_function="navigate_home",
            target_function="go_home",
            parameter_mappings=[],
            description="Navigate to home screen"
        ),
        
        ActionMapping(
            source_function="navigate_back",
            target_function="go_back",
            parameter_mappings=[],
            description="Navigate back"
        ),
        
        # App interaction
        ActionMapping(
            source_function="open_app",
            target_function="launch_application",
            parameter_mappings=[
                ParameterMapping(source_name="arg_0", target_name="application_name")
            ],
            description="Launch an application"
        ),
        
        # Touch interactions
        ActionMapping(
            source_function="click",
            target_function="touch",
            parameter_mappings=[
                ParameterMapping(source_name="x", target_name="x_coord"),
                ParameterMapping(source_name="y", target_name="y_coord")
            ],
            description="Touch screen at coordinates"
        ),
        
        ActionMapping(
            source_function="long_press",
            target_function="long_touch",
            parameter_mappings=[
                ParameterMapping(source_name="x", target_name="x_coord"),
                ParameterMapping(source_name="y", target_name="y_coord"),
                ParameterMapping(source_name="duration", target_name="hold_time", default_value=1.0)
            ],
            description="Long touch screen at coordinates"
        ),
        
        # Gesture actions
        ActionMapping(
            source_function="swipe",
            target_function="gesture_swipe",
            parameter_mappings=[
                ParameterMapping(source_name="from_coord", target_name="start_point"),
                ParameterMapping(source_name="to_coord", target_name="end_point"),
            ],
            description="Swipe gesture between two points"
        ),
        
        # Scroll actions with custom direction mapping
        ActionMapping(
            source_function="scroll",
            target_function="scroll_view",
            parameter_mappings=[
                ParameterMapping(
                    source_name="direction",
                    target_name="scroll_direction",
                    transform=lambda x: {"up": "north", "down": "south", "left": "west", "right": "east"}.get(x, x)
                ),
                ParameterMapping(source_name="amount", target_name="scroll_distance")
            ],
            description="Scroll view in specified direction"
        ),
        
        # Input actions
        ActionMapping(
            source_function="type",
            target_function="input_text",
            parameter_mappings=[
                ParameterMapping(source_name="text", target_name="content"),
            ],
            description="Input text"
        ),
        
        ActionMapping(
            source_function="press",
            target_function="key_press",
            parameter_mappings=[
                ParameterMapping(source_name="keys", target_name="key_combination")
            ],
            description="Press key combination"
        ),
        
        # Mouse actions
        ActionMapping(
            source_function="move_mouse",
            target_function="cursor_move",
            parameter_mappings=[
                ParameterMapping(source_name="x", target_name="x_position"),
                ParameterMapping(source_name="y", target_name="y_position")
            ],
            description="Move cursor to position"
        ),
        
        ActionMapping(
            source_function="double_click",
            target_function="double_touch",
            parameter_mappings=[
                ParameterMapping(source_name="x", target_name="x_coord", default_value=0.5),
                ParameterMapping(source_name="y", target_name="y_coord", default_value=0.5)
            ],
            description="Double touch at coordinates"
        ),
        
        ActionMapping(
            source_function="right_click",
            target_function="context_menu",
            parameter_mappings=[
                ParameterMapping(source_name="x", target_name="x_coord", default_value=0.5),
                ParameterMapping(source_name="y", target_name="y_coord", default_value=0.5)
            ],
            description="Open context menu"
        ),
        
        ActionMapping(
            source_function="drag",
            target_function="drag_and_drop",
            parameter_mappings=[
                ParameterMapping(source_name="from_coord", target_name="start_position"),
                ParameterMapping(source_name="to_coord", target_name="end_position")
            ],
            description="Drag and drop operation"
        ),
        
        # Timing and completion
        ActionMapping(
            source_function="wait",
            target_function="pause",
            parameter_mappings=[
                ParameterMapping(source_name="seconds", target_name="duration")
            ],
            description="Pause execution for specified duration"
        ),
        
        ActionMapping(
            source_function="final_answer",
            target_function="complete_task",
            parameter_mappings=[
                ParameterMapping(source_name="arg_0", target_name="answer")
            ],
            description="Complete task with result"
        ),
    ]
    
    return ActionSpaceConverter(mappings)

def convert_assistant(chat_message: dict, converter: ActionSpaceConverter) -> dict:
    """
    Convert function calls in assistant messages to sentence format.
    
    Args:
        chat_message: Dictionary with format {"user": "...", "assistant": "..."}
        
    Returns:
        Updated chat message with function calls converted to sentences
    """
    from utils.function_parser import parse_function_call
    
    if "assistant" not in chat_message:
        return chat_message
    
    assistant_message = chat_message["assistant"]
    
    # Parse function calls from the assistant message
    old_function_calls = parse_function_call(assistant_message)
    new_function_calls = converter.convert_actions(old_function_calls)
    
    # Replace each function call with its sentence format
    updated_message = assistant_message
    for new_function_call, old_function_call in zip(new_function_calls, old_function_calls):
        updated_message = updated_message.replace(old_function_call.to_string(), new_function_call.to_string())
    
    # Return updated chat message
    chat_message["assistant"] = updated_message
    return chat_message


# Testing and demonstration
if __name__ == "__main__":
    
    logger.info("🧪 Testing Action Space Converter")
    logger.info("=" * 50)
    
    # Test with default custom converter
    converter = create_default_unified_to_custom_converter()

    chat_history = [ 
        {"user": "Click on the home button", "assistant": "I'll click on the home button for you. navigate_home()"},
        {"user": "Type hello world", "assistant": "I'll type that text for you. type(text='hello world')"},
        {"user": "Click at coordinates 0.5, 0.8", "assistant": "I'll click at those coordinates. click(x=0.5, y=0.8)"},
        {"user": "Scroll up by 10 units", "assistant": "I'll scroll up for you. scroll(direction='up', amount=10)"}
    ]

    # Test the function with chat history
    logger.info("🧪 Testing Chat Message Function Call Conversion")
    logger.info("=" * 60)
    
    for i, chat_msg in enumerate(chat_history):
        logger.info(f"\n📩 Chat Message {i+1}:")
        logger.info(f"  User: {chat_msg['user']}")
        logger.info(f"  Original Assistant: {chat_msg['assistant']}")
        
        converted_msg = convert_assistant(chat_msg, converter)
        logger.info(f"  Converted Assistant: {converted_msg['assistant']}")