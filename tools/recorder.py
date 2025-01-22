"""Tool for recording and replaying computer control actions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import ToolResult

RECORDINGS_DIR = Path("recordings").resolve()


class ActionRecorder:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_file = RECORDINGS_DIR / f"{self.session_id}.json"
        self.current_index = 0
        self._ensure_dir()
        self._load_or_create()

    def _ensure_dir(self):
        """Ensure the recordings directory exists."""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_or_create(self):
        """Load existing recording or create new one."""
        if self.recording_file.exists():
            with open(self.recording_file) as f:
                self.recording = json.load(f)
        else:
            self.recording = {
                "session_id": self.session_id,
                "actions": [],
                "messages": [],
                "tools": {},
            }

    def save(self):
        """Save the current recording to file."""
        with open(self.recording_file, "w") as f:
            json.dump(self.recording, f, indent=2)

    def record_message(self, role: str, content: Any):
        """Record a message in the conversation."""
        self.recording["messages"].append({"role": role, "content": content})
        self.save()

    def record_tool_result(self, tool_id: str, result: ToolResult):
        """Record a tool result."""
        self.recording["tools"][tool_id] = {
            "output": result.output,
            "error": result.error,
            "base64_image": result.base64_image,
            "system": result.system,
        }
        self.save()

    def record_action(self, action_type: str, id: str | None = None, **kwargs):
        """Record an action with its parameters.

        Args:
            action_type: The type of tool being used (e.g. 'computer', 'bash', etc.)
            id: The unique identifier for this specific tool call
            **kwargs: The parameters for the tool
        """
        # Remove id from kwargs if it exists
        kwargs.pop("id", None)

        self.recording["actions"].append(
            {
                "tool_name": action_type,
                "tool_params": kwargs,
                "block": {
                    "id": id,
                    "input": kwargs,
                    "name": action_type,
                    "type": "tool_use",
                },
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.save()

    def get_next_action(self) -> Optional[dict]:
        """Get the next action to replay, if any.

        Returns:
            A dictionary containing the action details in the format:
            {
                "type": str,  # The tool type (e.g. "computer")
                "name": str,  # Same as type
                "id": str,    # The unique tool call ID
                "input": dict # The parameters for the tool
            }
        """
        print("\nDEBUG Current action index:", self.current_index)
        print("\nDEBUG Total actions:", len(self.recording["actions"]))

        if self.current_index >= len(self.recording["actions"]):
            print("\nDEBUG No more actions available")
            return None

        action = self.recording["actions"][self.current_index]
        print("\nDEBUG Loading action:", json.dumps(action, indent=2))
        self.current_index += 1

        # Return in the format expected by the loop
        formatted_action = {
            "type": "tool_use",
            "name": action["tool_name"],
            "id": action["block"]["id"],
            "input": action["tool_params"],
        }
        print("\nDEBUG Formatted action:", json.dumps(formatted_action, indent=2))
        return formatted_action

    def get_tool_result(self, call_id: str) -> Optional[ToolResult]:
        """Get a recorded tool result by call ID."""
        if call_id in self.recording["tools"]:
            data = self.recording["tools"][call_id]
            return ToolResult(
                output=data["output"],
                error=data["error"],
                base64_image=data["base64_image"],
                system=data.get("system"),
            )
        return None

    def get_messages(self) -> list:
        """Get all recorded messages."""
        return self.recording["messages"]

    def is_replay_complete(self) -> bool:
        """Check if all recorded actions have been replayed."""
        return self.current_index >= len(self.recording["actions"])

    @classmethod
    def list_recordings(cls) -> list[str]:
        """List all available recording session IDs."""
        if not RECORDINGS_DIR.exists():
            return []
        return [f.stem for f in RECORDINGS_DIR.glob("*.json")]

    @classmethod
    def delete_recording(cls, session_id: str):
        """Delete a recording session."""
        recording_file = RECORDINGS_DIR / f"{session_id}.json"
        if recording_file.exists():
            recording_file.unlink()
