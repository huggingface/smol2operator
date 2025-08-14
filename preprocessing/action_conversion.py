import logging
from utils.function_parser import FunctionCall
from copy import deepcopy

# Configure logger for this module
logger = logging.getLogger(__name__)

"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                            🔄 ACTION CONVERSION MAPPINGS 🔄                         ║
║                      Transform Aguvis & PyAutoGUI actions to unified API             ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

📱 MOBILE ACTIONS (Aguvis → Custom API):
  • mobile.home()                                       →  navigate_home()
  • mobile.open_app(app_name='drupe')                   →  open_app(app_name: str)
  • mobile.swipe(from_coord=[x,y], to_coord=[x,y])      →  swipe(from_coord: tuple, to_coord: tuple)
  • mobile.back()                                       →  navigate_back()
  • mobile.long_press(x=0.8, y=0.9)                     →  long_press(x: float, y: float)
  • mobile.terminate(status='success')                  →  final_answer(answer: str)
  • mobile.wait(seconds=3)                              →  wait(seconds: int)

💻 DESKTOP ACTIONS (PyAutoGUI → Custom API):
  • pyautogui.click(x=0.8, y=0.9)                       →  click(x: float, y: float)
  • pyautogui.doubleClick()                             →  double_click()
  • pyautogui.rightClick()                              →  right_click()
  • pyautogui.hotkey(keys=['ctrl', 'c'])                →  press(keys: str | list)
  • pyautogui.press(keys='enter')                       →  press(keys: str | list)
  • pyautogui.moveTo(x=0.04, y=0.4)                     →  move_mouse(x: float, y: float)
  • pyautogui.write(message='text')                     →  type(text: str)
  • pyautogui.dragTo(from_coord=[x,y], to_coord=[x,y])  →  drag(from_coord: tuple, to_coord: tuple)

🖱️ SCROLL ACTIONS (Smart Direction Detection):
  • pyautogui.scroll(page=-0.1)    [negative]           →  scroll(direction="up", amount=10)
  • pyautogui.scroll(page=0.1)     [positive]           →  scroll(direction="down", amount=10)
  • pyautogui.hscroll(page=-0.1)   [negative]           →  scroll(direction="left", amount=10)
  • pyautogui.hscroll(page=0.1)    [positive]           →  scroll(direction="right", amount=10)

✅ COMPLETION ACTIONS:
  • answer('text')                                      →  final_answer('text')
"""


def convert_to_pixel_coordinates(action: FunctionCall, resolution: tuple[int, int]) -> None:
    """
    🎯 Convert normalized coordinates (0.0-1.0) to absolute pixel coordinates.
    
    Transforms relative coordinates to screen pixels based on the given resolution.
    Handles both single coordinates (x, y) and coordinate pairs (from_coord, to_coord).
    
    Args:
        action: FunctionCall object containing coordinate parameters
        resolution: Screen resolution as (width, height) in pixels
        
    Note: Modifies the action object in-place by updating parameter names and values.
    """
    if "arg_0" in action.parameters:
        if isinstance(action.parameters["arg_0"], (list, tuple)):
            action.parameters["from_coord"] = (int(action.parameters["arg_0"][0] * resolution[0]), int(action.parameters["arg_0"][1] * resolution[1]))
        else:
            action.parameters["x"] = int(action.parameters["arg_0"] * resolution[0])
        del action.parameters["arg_0"]
    if "arg_1" in action.parameters:
        if isinstance(action.parameters["arg_1"], (list, tuple)):
            action.parameters["to_coord"] = (int(action.parameters["arg_1"][0] * resolution[0]), int(action.parameters["arg_1"][1] * resolution[1]))
        else:
            action.parameters["y"] = int(action.parameters["arg_1"] * resolution[1])
        del action.parameters["arg_1"]

def change_argument_name(action: FunctionCall) -> None:
    """
    🔄 Transform generic argument names to semantic parameter names.
    
    Converts arg_0, arg_1 to meaningful names like 'x', 'y', 'from_coord', 'to_coord'.
    Maintains coordinate values as floats for normalized coordinate system.
    
    Args:
        action: FunctionCall object with generic argument names
        
    Note: Modifies the action object in-place, preserving original coordinate values.
    """
    if "arg_0" in action.parameters:
        if isinstance(action.parameters["arg_0"], (list, tuple)):
            action.parameters["from_coord"] = (float(action.parameters["arg_0"][0]), float(action.parameters["arg_0"][1]))
        else:
            action.parameters["x"] = float(action.parameters["arg_0"])
        del action.parameters["arg_0"]
    if "arg_1" in action.parameters:
        if isinstance(action.parameters["arg_1"], (list, tuple)):
            action.parameters["to_coord"] = (float(action.parameters["arg_1"][0]), float(action.parameters["arg_1"][1]))
        else:
            action.parameters["y"] = float(action.parameters["arg_1"])
        del action.parameters["arg_1"]


def rename_parameters(action: FunctionCall) -> None:
    """
    🏷️ Standardize parameter names to arg_0, arg_1, arg_2 format.
    
    Converts named parameters to a generic indexed format while preserving
    the original parameter order. This creates a uniform interface for
    subsequent processing steps.
    
    Args:
        action: FunctionCall object to standardize parameter names for
        
    Example:
        Before: {"x": 0.5, "y": 0.8} → After: {"arg_0": 0.5, "arg_1": 0.8}
    """
    if not action.parameters:
        return
    
    for i, (key, value) in enumerate(deepcopy(action.parameters).items()):
        tmp = value
        del action.parameters[key]
        action.parameters[f"arg_{i}"] = tmp



def action_conversion(
    actions: list[FunctionCall], resolution: tuple[int, int]
) -> list[FunctionCall]:
    """
    🚀 Master conversion function: Transform diverse action formats into unified API.
    
    This is the main orchestrator that converts actions from different sources
    (Aguvis mobile actions, PyAutoGUI desktop actions) into a standardized
    action format for consistent processing.
    
    Args:
        actions: List of FunctionCall objects to convert
        resolution: Screen resolution (width, height) for coordinate conversion
        
    Returns:
        List of converted FunctionCall objects with unified naming and structure
        
    Features:
        • 📱 Mobile action normalization (Aguvis → Custom API)
        • 💻 Desktop action standardization (PyAutoGUI → Custom API)
        • 🎯 Smart coordinate handling (relative ↔ absolute)
        • 🖱️ Intelligent scroll direction detection
        • ✅ Consistent error handling and validation
    """
    for i, action in enumerate(actions):
        rename_parameters(action)
        
        # ═══════════════════════════════════════════════════════════════
        # 📱 MOBILE ACTIONS (Aguvis Framework)
        # ═══════════════════════════════════════════════════════════════
        if action.function_name == "mobile.home":
            actions[i].function_name = "navigate_home"

        elif action.function_name == "mobile.open_app":
            actions[i].function_name = "open_app"

        elif action.function_name == "mobile.swipe":
            actions[i].function_name = "swipe"
            change_argument_name(actions[i])

        elif action.function_name == "mobile.back":
            actions[i].function_name = "navigate_back"

        elif action.function_name == "mobile.long_press":
            actions[i].function_name = "long_press"
            change_argument_name(actions[i])

        elif action.function_name in ["mobile.terminate", "answer"]:
            actions[i].function_name = "final_answer"

        elif action.function_name == "mobile.wait":
            actions[i].function_name = "wait"
            if "arg_0" in actions[i].parameters:
                actions[i].parameters["seconds"] = int(actions[i].parameters["arg_0"])
                del actions[i].parameters["arg_0"]

        # ═══════════════════════════════════════════════════════════════
        # 💻 DESKTOP ACTIONS (PyAutoGUI Framework)
        # ═══════════════════════════════════════════════════════════════
        elif action.function_name == "pyautogui.click":
            actions[i].function_name = "click"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.doubleClick":
            actions[i].function_name = "double_click"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.rightClick":
            actions[i].function_name = "right_click"
            change_argument_name(actions[i])

        elif action.function_name in ["pyautogui.hotkey", "pyautogui.press"]:
            actions[i].function_name = "press"
            if "arg_0" in actions[i].parameters:
                actions[i].parameters["keys"] = actions[i].parameters["arg_0"]
                del actions[i].parameters["arg_0"]

        elif action.function_name == "pyautogui.moveTo":
            actions[i].function_name = "move_mouse"
            change_argument_name(actions[i])

        elif action.function_name == "pyautogui.write":
            actions[i].function_name = "type"

        # ──────────────────────────────────────────────────────────────
        # 🖱️ SCROLL ACTIONS (Direction Detection)
        # ──────────────────────────────────────────────────────────────
        elif action.function_name in ["pyautogui.scroll", "pyautogui.hscroll"]:
            arg_value = actions[i].parameters["arg_0"]
            if arg_value < 0:
                if action.function_name == "pyautogui.hscroll":
                    actions[i].parameters["direction"] = "left"
                else:
                    actions[i].parameters["direction"] = "up"
            else:
                if action.function_name == "pyautogui.hscroll":
                    actions[i].parameters["direction"] = "right"
                else:
                    actions[i].parameters["direction"] = "down"
            del actions[i].parameters["arg_0"]
            actions[i].function_name = "scroll"
            actions[i].parameters["amount"] = int(abs(arg_value * 100))

        elif action.function_name == "pyautogui.dragTo":
            actions[i].function_name = "drag"
            change_argument_name(actions[i])

        else:
            raise ValueError(f"🚫 Unsupported action: {action.function_name}")

        # 💾 Preserve original string representation for debugging
        actions[i].original_string = actions[i].to_string()

    return actions

if __name__ == "__main__":
    from utils.function_parser import FunctionCall

    """
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                            🧪 TESTING & DEMONSTRATION 🧪                      ║
    ║                   Comprehensive test suite for action conversion               ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    """
    
    # 📋 Complete test dataset covering all supported action types
    actions = [
        # ═══════════════════════════════════════════════════════════════
        # 📱 MOBILE ACTIONS (Aguvis Framework)
        # ═══════════════════════════════════════════════════════════════
        FunctionCall("mobile.home", {}, "mobile.home()"),
        FunctionCall("mobile.open_app", {"app_name": "drupe"}, "mobile.open_app(app_name='drupe')"),
        FunctionCall("mobile.swipe", {"from_coord": [0.581, 0.898], "to_coord": [0.601, 0.518]}, "mobile.swipe(from_coord=[0.581,0.898],to_coord=[0.601,0.518])"),
        FunctionCall("mobile.back", {}, "mobile.back()"),
        FunctionCall("mobile.long_press", {"x": 0.799, "y": 0.911}, "mobile.long_press(x=0.799, y=0.911)"),
        FunctionCall("mobile.terminate", {"status": "success"}, "mobile.terminate(status='success')"),
        FunctionCall("answer", {"arg_0": "text"}, "answer('text')"),
        FunctionCall("mobile.wait", {"seconds": 3}, "mobile.wait(seconds=3)"),
        
        # ═══════════════════════════════════════════════════════════════
        # 💻 DESKTOP ACTIONS (PyAutoGUI Framework)
        # ═══════════════════════════════════════════════════════════════
        FunctionCall("pyautogui.hscroll", {"page": -0.1}, "pyautogui.hscroll(page=-0.1)"),
        FunctionCall("pyautogui.scroll", {"page": 0.13}, "pyautogui.scroll(page=0.13)"),
        FunctionCall("pyautogui.click", {"x": 0.8102, "y": 0.9463}, "pyautogui.click(x=0.8102, y=0.9463)"),
        FunctionCall("pyautogui.doubleClick", {}, "pyautogui.doubleClick()"),
        FunctionCall("pyautogui.hotkey", {"keys": ["ctrl", "c"]}, "pyautogui.hotkey(keys=['ctrl','c'])"),
        FunctionCall("pyautogui.press", {"keys": "enter"}, "pyautogui.press(keys='enter')"),
        FunctionCall("pyautogui.moveTo", {"x": 0.04, "y": 0.405}, "pyautogui.moveTo(x=0.04, y=0.405)"),
        FunctionCall("pyautogui.write", {"message": "bread buns"}, "pyautogui.write(message='bread buns')"),
        FunctionCall("pyautogui.dragTo", {"from_coord": [0.87, 0.423], "to_coord": [0.8102, 0.9463]}, "pyautogui.dragTo(from_coord=[0.87, 0.423], to_coord=[0.8102, 0.9463])"),
    ]
    
    # 🖥️ Test resolution (Full HD Portrait - typical mobile orientation)
    resolution = (1080, 1920)
    
    logger.info("🔄 BEFORE CONVERSION:")
    logger.info("═" * 50)
    for action in actions:
        logger.info(f"  📋 {action}")
    
    logger.info(f"\n🚀 AFTER CONVERSION (Resolution: {resolution}):")
    logger.info("═" * 50)
    converted = action_conversion(actions, resolution)
    for action in converted:
        logger.info(f"  ✅ {action}")
