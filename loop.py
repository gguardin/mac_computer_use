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
import logging
import sys
from pathlib import Path

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
from utils import safe_dumps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


async def _handle_replay_mode(
    recorder: ActionRecorder,
) -> Optional[list[BetaTextBlockParam | BetaToolUseBlockParam]]:
    """
    Handle replay mode logic and return next assistant message if available.
    Returns:
        - response_params: The next assistant message content if available, None otherwise
    """
    logger.info("Starting replay mode handler")
    logger.debug("Current recorder index: %d", recorder.current_index)

    next_message = recorder.get_next_message()
    if next_message:
        logger.info("Found next message to replay")
        return next_message["content"]
    else:
        logger.info("No next message found")
        return None


async def _get_llm_response(
    client: Anthropic | AnthropicBedrock | AnthropicVertex,
    messages: list[BetaMessageParam],
    model: str,
    system: BetaTextBlockParam,
    tool_collection: ToolCollection,
    only_n_most_recent_images: Optional[int],
    max_tokens: int,
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    provider: APIProvider,
) -> Optional[list[BetaTextBlockParam | BetaToolUseBlockParam]]:
    """Get response from LLM with appropriate configuration."""
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
        return None
    except APIError as e:
        api_response_callback(e.request, e.body, e)
        return None

    api_response_callback(
        raw_response.http_response.request, raw_response.http_response, None
    )
    response = raw_response.parse()
    return _response_to_params(response)


async def _process_tool_calls(
    response_params: list[BetaTextBlockParam | BetaToolUseBlockParam],
    tool_collection: ToolCollection,
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    recorder: ActionRecorder,
) -> list[BetaToolResultBlockParam]:
    """Process tool calls and return tool results."""
    tool_result_content: list[BetaToolResultBlockParam] = []

    logger.info("Starting to process tool calls")
    logger.debug("Found %d content blocks", len(response_params))

    for content_block in response_params:
        output_callback(content_block)
        if content_block["type"] == "tool_use":
            try:
                logger.info("Processing tool: %s", content_block["name"])
                logger.debug("Tool action: %s", content_block["input"].get("action"))
                logger.debug("Tool ID: %s", content_block["id"])

                # Execute the tool and get the result
                logger.info("Executing tool...")
                tool_result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )

                # Process the tool result
                logger.debug("Tool result: %s", safe_dumps(tool_result.__dict__))
                tool_output_callback(tool_result, content_block["id"])

                # Create API tool result
                tool_result_content.append(
                    _make_api_tool_result(tool_result, content_block["id"])
                )
                logger.debug("Tool result processed")

            except Exception as e:
                logger.error("Error processing tool call: %s", str(e))
                logger.error("Traceback: %s", traceback.format_exc())
                tool_result = ToolResult(
                    error=f"Error processing tool call: {str(e)}",
                    output="",
                    base64_image=None,
                )
                tool_output_callback(tool_result, content_block["id"])
                tool_result_content.append(
                    _make_api_tool_result(tool_result, content_block["id"])
                )

    return tool_result_content


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
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    logger.info("Starting new sampling iteration")
    logger.debug("Session ID: %s", session_id)
    logger.debug("Messages in conversation: %d", len(messages))
    logger.debug("Model: %s, Provider: %s", model, provider)

    # Initialize tools and system prompt
    tool_collection = ToolCollection(
        ComputerTool(),
        BashTool(),
        EditTool(),
    )
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    # Get the current recorder instance
    recorder = ActionRecorder.get_instance(session_id)

    # Initialize client
    if provider == APIProvider.ANTHROPIC:
        client = Anthropic(api_key=api_key, max_retries=4)
    elif provider == APIProvider.VERTEX:
        client = AnthropicVertex()
    elif provider == APIProvider.BEDROCK:
        client = AnthropicBedrock()
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    while True:
        logger.debug("Starting main loop iteration")
        response_params = None

        if recorder.in_replay_mode:
            logger.info("Running in replay mode")
            response_params = await _handle_replay_mode(recorder)
            if response_params:
                logger.debug("Found %d blocks to replay", len(response_params))
        else:
            logger.info("Getting response from LLM")
            response_params = await _get_llm_response(
                client,
                messages,
                model,
                system,
                tool_collection,
                only_n_most_recent_images,
                max_tokens,
                api_response_callback,
                provider,
            )

        if response_params is None:
            logger.warning("No response parameters received")
            return messages

        # Create the message once and use it for both recording and appending
        assistant_message = {"role": "assistant", "content": response_params}
        if not recorder.in_replay_mode:
            logger.debug("Recording assistant message")
            recorder.record_message(assistant_message)

        messages.append(assistant_message)

        logger.info("Processing tool calls")
        tool_result_content = await _process_tool_calls(
            response_params,
            tool_collection,
            output_callback,
            tool_output_callback,
            recorder,
        )

        logger.info("Processed %d tool results", len(tool_result_content))

        if not tool_result_content:
            return messages

        # Create the message once and use it for both recording and appending
        user_message = {"role": "user", "content": tool_result_content}
        messages.append(user_message)

        if not recorder.in_replay_mode:
            logger.debug("Recording tool results in conversation")
            recorder.record_message(user_message)
        else:
            # during replay mode, next message is the tool result, so we skip the user message
            recorder.skip_user_message()


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
