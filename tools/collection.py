"""Collection classes for managing multiple tools."""

from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)


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
        print("\n" + "=" * 50)
        print(f"[TOOL EXECUTION] Tool: {name}")
        print(f"[TOOL EXECUTION] Action: {tool_input.get('action', 'unknown')}")
        print(f"[TOOL EXECUTION] Parameters: {tool_input}")

        tool = self.tool_map.get(name)
        if not tool:
            print("[TOOL EXECUTION] ❌ Failed - Invalid tool name")
            return ToolFailure(error=f"Tool {name} is invalid")

        try:
            print("[TOOL EXECUTION] 🚀 Executing tool...")
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

                print(f"[TOOL EXECUTION] ✅ Completed - Produced: {status_str}")
                if result.error:
                    print(f"[TOOL EXECUTION] Error details: {result.error}")
            return result

        except ToolError as e:
            print(f"[TOOL EXECUTION] ❌ Failed - Error: {e.message}")
            return ToolFailure(error=e.message)
        finally:
            print("=" * 50)
