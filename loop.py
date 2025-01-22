"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional, cast
import json
import asyncio
import traceback

import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult
from tools.recorder import ActionRecorder

COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
# SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
# * You are utilizing a macOS Sonoma 15.7 environment using {platform.machine()} architecture with internet access.
# * You can install applications using homebrew with your bash tool. Use curl instead of wget.
# * To open Chrome, please just click on the Chrome icon in the Dock or use Spotlight.
# * Using bash tool you can start GUI applications. GUI apps can be launched directly or with `open -a "Application Name"`. GUI apps will appear natively within macOS, but they may take some time to appear. Take a screenshot to confirm it did.
# * When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
# * When viewing a page it can be helpful to zoom out so that you can see everything on the page. In Chrome, use Command + "-" to zoom out or Command + "+" to zoom in.
# * When using your computer function calls, they take a while to run and send back to you. Where possible/feasible, try to chain multiple of these calls all into one function calls request.
# * The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
# </SYSTEM_CAPABILITY>
# <IMPORTANT>
# * When using Chrome, if any first-time setup dialogs appear, IGNORE THEM. Instead, click directly in the address bar and enter the appropriate search term or URL there.
# * If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext (available via homebrew) to convert it to a text file, and then read that text file directly with your StrReplaceEditTool.
# </IMPORTANT>"""
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a macOS Sequoia 15.2 environment using {platform.machine()} architecture with command line internet access.
* Package management:
  - Use homebrew for package installation
  - Use curl for HTTP requests
  - Use npm/yarn for Node.js packages
  - Use pip for Python packages

* Browser automation available via Playwright:
  - Supports Chrome, Firefox, and WebKit
  - Can handle JavaScript-heavy applications
  - Capable of screenshots, navigation, and interaction
  - Handles dynamic content loading

* System automation:
  - cliclick for simulating mouse/keyboard input
  - osascript for AppleScript commands
  - launchctl for managing services
  - defaults for reading/writing system preferences
  - MacOS-specific hotkeys (e.g. Command+Shift+3 for screenshot, Command+a for select all202)

* Development tools:
  - Standard Unix/Linux command line utilities
  - Git for version control
  - Docker for containerization
  - Common build tools (make, cmake, etc.)

* Output handling:
  - For large output, redirect to tmp files: command > /tmp/output.txt
  - Use grep with context: grep -n -B <before> -A <after> <query> <filename>
  - Stream processing with awk, sed, and other text utilities

* Note: Command line function calls may have latency. Chain multiple operations into single requests where feasible.

* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>"""


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    session_id: Optional[str] = None,
    replay_mode: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    print("\nDEBUG Sampling loop started")
    print("\nDEBUG Replay mode:", replay_mode)
    print("\nDEBUG Session ID:", session_id)
    print(
        "\nDEBUG Initial messages:",
        json.dumps(
            [
                {
                    "role": m["role"],
                    "content_type": type(m["content"]).__name__,
                    "content_length": (
                        len(m["content"]) if isinstance(m["content"], list) else 1
                    ),
                }
                for m in messages
            ],
            indent=2,
        ),
    )

    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    recorder = ActionRecorder(session_id)

    if replay_mode and not session_id:
        raise ValueError("session_id is required in replay mode")

    if replay_mode:
        print("\nDEBUG Checking for next recorded message in replay mode")
        if "messages" in recorder.recording:
            print(
                f"\nDEBUG Total messages in recording: {len(recorder.recording['messages'])}"
            )
            print(f"\nDEBUG Current index: {recorder.current_index}")
            print("\nDEBUG Current messages in memory:", len(messages))
            print("\nDEBUG Last message in memory:", json.dumps(messages[-1], indent=2))

            if recorder.current_index < len(recorder.recording["messages"]):
                next_message = recorder.recording["messages"][recorder.current_index]
                recorder.current_index += 1
                print(
                    "\nDEBUG Found message to replay:",
                    json.dumps(next_message, indent=2),
                )
                print("\nDEBUG Message role:", next_message["role"])

                if next_message["role"] == "assistant":
                    response_params = next_message["content"]
                    print("\nDEBUG Set response_params from recorded message")
                    print(
                        "\nDEBUG Response params:",
                        json.dumps(response_params, indent=2),
                    )
                    messages.append({"role": "assistant", "content": response_params})
                    print("\nDEBUG Added assistant message, returning messages")
                    return messages
                else:
                    print("\nDEBUG Adding non-assistant message")
                    messages.append(next_message)
                    print("\nDEBUG Messages after append:", len(messages))
                    print(
                        "\nDEBUG Last message now:", json.dumps(messages[-1], indent=2)
                    )
            else:
                print("\nDEBUG No more messages to replay")
                print("\nDEBUG Switching to live mode")
                replay_mode = False
        else:
            print("\nDEBUG No messages found in recording")

    # Initialize client outside the loop to fix linter error
    if provider == APIProvider.ANTHROPIC:
        client = Anthropic(api_key=api_key, max_retries=4)
    elif provider == APIProvider.VERTEX:
        client = AnthropicVertex()
    elif provider == APIProvider.BEDROCK:
        client = AnthropicBedrock()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    while True:
        print("\nDEBUG Starting main loop iteration")
        # Get next response - either from recording or live LLM
        response_params = None

        if replay_mode:
            print("\nDEBUG In replay mode, checking for recorded message")
            if hasattr(recorder.recording, "messages"):
                print(
                    f"\nDEBUG Total messages in recording: {len(recorder.recording['messages'])}"
                )
                print(f"\nDEBUG Current index: {recorder.current_index}")
                if recorder.current_index < len(recorder.recording["messages"]):
                    next_message = recorder.recording["messages"][
                        recorder.current_index
                    ]
                    recorder.current_index += 1
                    print(
                        "\nDEBUG Found message to replay in main loop:",
                        json.dumps(next_message, indent=2),
                    )
                    response_params = next_message["content"]
                    print("\nDEBUG Set response_params in main loop")
                else:
                    print("\nDEBUG No more messages to replay in main loop")
            else:
                print("\nDEBUG No messages found in recording in main loop")

        # If no recorded message, get response from LLM
        if response_params is None:
            print("\nDEBUG No recorded message, getting response from LLM")
            enable_prompt_caching = False
            betas = [COMPUTER_USE_BETA_FLAG]
            image_truncation_threshold = only_n_most_recent_images or 0

            if provider == APIProvider.ANTHROPIC:
                enable_prompt_caching = True

            if enable_prompt_caching:
                betas.append(PROMPT_CACHING_BETA_FLAG)
                _inject_prompt_caching(messages)
                only_n_most_recent_images = 0
                system["cache_control"] = {"type": "ephemeral"}

            if only_n_most_recent_images:
                _maybe_filter_to_n_most_recent_images(
                    messages,
                    only_n_most_recent_images,
                    min_removal_threshold=image_truncation_threshold,
                )

            try:
                raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                )
            except (APIStatusError, APIResponseValidationError) as e:
                api_response_callback(e.request, e.response, e)
                return messages
            except APIError as e:
                api_response_callback(e.request, e.body, e)
                return messages

            api_response_callback(
                raw_response.http_response.request, raw_response.http_response, None
            )

            response = raw_response.parse()
            response_params = _response_to_params(response)

            # Record the assistant's message
            recorder.record_message("assistant", response_params)

        # Common path for processing response, whether from replay or LLM
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                # Record the tool use action
                recorder.record_action(
                    content_block["name"],
                    id=content_block["id"],
                    **content_block["input"],
                )

                # Execute the tool and get the result
                print(
                    "\nDEBUG Tool call parameters:",
                    json.dumps(
                        {
                            "tool_name": content_block["name"],
                            "tool_params": cast(dict[str, Any], content_block["input"]),
                            "block": content_block,
                        },
                        indent=2,
                        default=str,
                    ),
                )

                tool_result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )

                # Record the tool result
                recorder.record_tool_result(content_block["id"], tool_result)

                tool_result_content.append(
                    _make_api_tool_result(tool_result, content_block["id"])
                )
                tool_output_callback(tool_result, content_block["id"])

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})
        recorder.record_message("user", tool_result_content)

        return messages


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaTextBlockParam | BetaToolUseBlockParam]:
    res: list[BetaTextBlockParam | BetaToolUseBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            res.append({"type": "text", "text": block.text})
        else:
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
