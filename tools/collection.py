"""Collection classes for managing multiple tools."""

import logging
from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)

logger = logging.getLogger(__name__)


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool):
        self.tools = tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in tools}

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        logger.info("=" * 50)
        logger.info("[TOOL EXECUTION] Tool: %s", name)
        logger.info("[TOOL EXECUTION] Action: %s", tool_input.get("action", "unknown"))
        logger.info("[TOOL EXECUTION] Parameters: %s", tool_input)

        tool = self.tool_map.get(name)
        if not tool:
            logger.error("[TOOL EXECUTION] ❌ Failed - Invalid tool name")
            return ToolFailure(error=f"Tool {name} is invalid")

        try:
            logger.info("[TOOL EXECUTION] 🚀 Executing tool...")
            result = await tool(**tool_input)

            if isinstance(result, ToolResult):
                status = []
                if result.output:
                    status.append("output")
                if result.base64_image:
                    status.append("image")
                if result.error:
                    status.append("error")
                status_str = ", ".join(status) if status else "no output"

                logger.info("[TOOL EXECUTION] ✅ Completed - Produced: %s", status_str)
                if result.error:
                    logger.error("[TOOL EXECUTION] Error details: %s", result.error)
            return result

        except ToolError as e:
            logger.error("[TOOL EXECUTION] ❌ Failed - Error: %s", e.message)
            return ToolFailure(error=e.message)
        finally:
            logger.info("=" * 50)
