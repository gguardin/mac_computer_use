import asyncio
import base64
import os
import shlex
import pyautogui
import keyboard
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}
SCALE_DESTINATION = MAX_SCALING_TARGETS["XGA"]


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current macOS computer.
    The tool parameters are defined by Anthropic and are not editable.
    Requires cliclick to be installed: brew install cliclick
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 1.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self):
        super().__init__()

        # self.width, self.height = pyautogui.size()

        self.width = int(os.getenv("WIDTH"))
        self.height = int(os.getenv("HEIGHT"))

        assert self.width and self.height, "WIDTH, HEIGHT must be set"

        # Read display number from environment variable
        display_num_str = os.getenv("DISPLAY_NUM")
        self.display_num = int(display_num_str) if display_num_str is not None else None

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print("Action: ", action, text, coordinate)
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.transform_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )

            x_str = f"={x}" if x < 0 else str(x)
            y_str = f"={y}" if y < 0 else str(y)
            if action == "mouse_move":
                return await self.shell(f"cliclick m:{x_str},{y_str}")
            elif action == "left_click_drag":
                return await self.shell(f"cliclick dd:{x_str},{y_str}")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Convert common key names to pyautogui format
                key_map = {
                    "Return": "enter",
                    "space": "space",
                    "Tab": "tab",
                    "Left": "left",
                    "Right": "right",
                    "Up": "up",
                    "Down": "down",
                    "Escape": "esc",
                    "command": "command",
                    "cmd": "command",
                    "alt": "alt",
                    "shift": "shift",
                    "ctrl": "ctrl",
                }

                try:
                    if "+" in text:
                        # Handle combinations like "ctrl+c"
                        keys = text.split("+")
                        mapped_keys = [key_map.get(k.strip(), k.strip()) for k in keys]
                        await asyncio.get_event_loop().run_in_executor(
                            None, keyboard.press_and_release, "+".join(mapped_keys)
                        )
                    else:
                        # Handle single keys
                        mapped_key = key_map.get(text, text)
                        await asyncio.get_event_loop().run_in_executor(
                            None, keyboard.press_and_release, mapped_key
                        )

                    return ToolResult(
                        output=f"Pressed key: {text}", error=None, base64_image=None
                    )

                except Exception as e:
                    return ToolResult(output=None, error=str(e), base64_image=None)
            elif action == "type":
                results: list[ToolResult] = []
                for chunk in chunks(text, TYPING_GROUP_SIZE):
                    cmd = f"cliclick w:{TYPING_DELAY_MS} t:{shlex.quote(chunk)}"
                    results.append(await self.shell(cmd, take_screenshot=False))
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(
                    output="".join(result.output or "" for result in results),
                    error="".join(result.error or "" for result in results),
                    base64_image=screenshot_base64,
                )

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                result = await self.shell(
                    "cliclick p",
                    take_screenshot=False,
                )

                if result.output:
                    x, y = map(int, result.output.strip().split(","))
                    x, y = self.transform_coordinates(ScalingSource.COMPUTER, x, y)
                    return result.replace(output=f"X={x},Y={y}")
                return result
            else:
                click_cmd = {
                    "left_click": "c:.",
                    "right_click": "rc:.",
                    "middle_click": "mc:.",
                    "double_click": "dc:.",
                }[action]
                return await self.shell(f"cliclick {click_cmd}")

        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        # Use macOS native screencapture
        # If display_num is set, use it to capture specific display
        display_flag = f"-D {self.display_num}" if self.display_num is not None else ""
        screenshot_cmd = f"screencapture -x {display_flag} {path}"
        print("screenshot_cmd", screenshot_cmd)
        result = await self.shell(screenshot_cmd, take_screenshot=False)

        if self._scaling_enabled:
            x, y = self.scale_coordinates(
                ScalingSource.COMPUTER, self.width, self.height
            )
            await self.shell(
                f"sips -z {y} {x} {path}",  # sips is macOS native image processor
                take_screenshot=False,
            )

        if path.exists():
            return result.replace(
                base64_image=base64.b64encode(path.read_bytes()).decode()
            )
        raise ToolError(f"Failed to take screenshot: {result.error}")

    async def shell(self, command: str, take_screenshot=False) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(
        self, source: ScalingSource, x: int, y: int
    ) -> tuple[int, int]:
        """
        Scale coordinates between original resolution and target resolution (SCALE_DESTINATION).

        Args:
            source: ScalingSource.API for scaling up from SCALE_DESTINATION to original resolution
                   or ScalingSource.COMPUTER for scaling down from original to SCALE_DESTINATION
            x, y: Coordinates to scale

        Returns:
            Tuple of scaled (x, y) coordinates
        """
        if not self._scaling_enabled:
            return x, y
        x_scaling_factor = SCALE_DESTINATION["width"] / self.width
        y_scaling_factor = SCALE_DESTINATION["height"] / self.height

        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def transform_coordinates(
        self, source: ScalingSource, x: int, y: int
    ) -> tuple[int, int]:
        """
        Translates and scales coordinates between:
        - The single-monitor API space
        - The real OS coordinate space for this LEFT monitor.

        Because the monitor is on the left, its top-left in the OS
        coordinate system is at negative X.

        Example: if monitor width is 1920, then (0,0) on this monitor
        is at real X = -1920 in the OS coordinate space.
        """
        if source == ScalingSource.API:
            # 1) Scale from the API's resolution up to the real monitor resolution
            scaled_x, scaled_y = self.scale_coordinates(ScalingSource.API, x, y)

            # 2) Shift to negative X (since it's the left monitor)
            #    (0,0) in monitor space => -real_width in OS space
            real_x = scaled_x - self.width
            real_y = scaled_y

            print(
                f"Transformed coordinates from API to Computer: ({x}, {y}) -> ({real_x}, {real_y})"
            )
            return real_x, real_y
        else:
            # source == ScalingSource.COMPUTER
            # 1) Shift from negative OS coords to local monitor coords
            local_x = x + self.width  # So if x = -1920, local_x = 0
            local_y = y

            # 2) Scale down from real resolution to the API resolution
            api_x, api_y = self.scale_coordinates(
                ScalingSource.COMPUTER, local_x, local_y
            )
            print(
                f"Transformed coordinates from Computer to API: ({x}, {y}) -> ({api_x}, {api_y})"
            )
            return api_x, api_y
