"""Tool for recording and replaying computer control actions."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging
import sys

from .base import ToolResult
from tools import ToolResult
from utils import safe_dumps

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
detailed_formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(detailed_formatter)

# File handler
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(
    log_dir / f"mac_computer_use_{datetime.now().strftime('%Y%m%d')}.log"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

RECORDINGS_DIR = Path("recordings").resolve()


class ActionRecorder:
    """Records and replays messages for a session."""

    _instance: Optional["ActionRecorder"] = None

    @classmethod
    def get_instance(cls, session_id: Optional[str] = None) -> "ActionRecorder":
        """Get or create the singleton recorder instance.

        Args:
            session_id: Optional session ID to initialize with. Only used if instance doesn't exist.

        Returns:
            The singleton recorder instance
        """
        # If we have an instance but session_id is different, reset it
        if (
            cls._instance is not None
            and session_id is not None
            and cls._instance.session_id != session_id
        ):
            logger.info("Resetting recorder instance due to different session ID")
            cls._instance = None

        if cls._instance is None:
            logger.info(
                "Creating new recorder instance with session ID: %s", session_id
            )
            cls._instance = cls(session_id)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        logger.info("Resetting recorder instance")
        cls._instance = None

    def __init__(self, session_id: Optional[str] = None):
        """Initialize recorder with optional session ID."""
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_file = RECORDINGS_DIR / f"{self.session_id}.json"
        self.current_index = 0
        self._in_replay_mode = False

        logger.info("Initializing recorder with session ID: %s", self.session_id)
        logger.debug("Recording file path: %s", self.recording_file)

        if self.recording_file.exists():
            logger.info("Loading existing recording file")
            self._in_replay_mode = True
            with open(self.recording_file) as f:
                self.recording = json.load(f)
            logger.debug("Loaded recording data: %s", safe_dumps(self.recording))
        else:
            logger.info("Creating new recording")
            self.recording = {
                "session_id": self.session_id,
                "messages": [],
            }

    @property
    def in_replay_mode(self) -> bool:
        """Whether the recorder is in replay mode."""
        return self._in_replay_mode

    def get_next_message(self) -> Optional[dict]:
        """Get the next message from the recording if in replay mode."""
        if not self._in_replay_mode:
            logger.debug("Not in replay mode, returning None")
            return None

        if self.current_index >= len(self.recording["messages"]):
            logger.debug("No more messages available")
            return None

        next_message = self.recording["messages"][self.current_index]
        self.current_index += 1
        logger.debug(
            "Retrieved next message at index %d: %s",
            self.current_index - 1,
            safe_dumps(next_message),
        )
        return next_message

    def skip_user_message(self):
        """Skip the next user message."""
        if self.current_index >= len(self.recording["messages"]):
            logger.warn("No more messages available")
            return

        if self.recording["messages"][self.current_index]["role"] != "user":
            logger.warn("Next message is not a user message")
            return

        self.current_index += 1
        logger.debug("Skipping user message at index %d", self.current_index - 1)

    def get_all_messages(self) -> list:
        """Get all messages from the recording."""
        messages = self.recording.get("messages", [])
        logger.debug("Retrieved %d messages", len(messages))
        return messages

    def _ensure_dir(self):
        """Ensure the recordings directory exists."""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured recordings directory exists: %s", RECORDINGS_DIR)

    def save(self):
        """Save the current recording to file."""
        logger.debug("Saving recording to file: %s", self.recording_file)
        with open(self.recording_file, "w") as f:
            json.dump(self.recording, f, indent=2)
        logger.info("Recording saved successfully")

    def record_message(self, message: Any):
        """Record a message in the conversation.

        Args:
            message: The message to record with its internal structure
        """
        if self._in_replay_mode:
            logger.warning("Attempted to record message in replay mode")
            return

        # Create a copy of the message to avoid modifying the original
        message_to_save = json.loads(json.dumps(message))

        # If this is a user message with tool results, filter out image data
        if message_to_save.get("role") == "user" and isinstance(
            message_to_save.get("content"), list
        ):
            for block in message_to_save["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if isinstance(block.get("content"), list):
                        for content in block["content"]:
                            if (
                                isinstance(content, dict)
                                and content.get("type") == "image"
                                and isinstance(content.get("source"), dict)
                            ):
                                # Remove the image data but keep the structure
                                content["source"]["data"] = ""

        logger.debug("Recording message: %s", safe_dumps(message_to_save))
        self.recording["messages"].append(message_to_save)
        self.save()

    def is_replay_complete(self) -> bool:
        """Check if replay is complete."""
        if not self._in_replay_mode:
            return False
        return self.current_index >= len(self.recording["messages"])

    @classmethod
    def list_recordings(cls) -> list[str]:
        """List all available recording session IDs."""
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        return [f.stem for f in RECORDINGS_DIR.glob("*.json")]

    @classmethod
    def delete_recording(cls, session_id: str):
        """Delete a recording by session ID."""
        recording_file = RECORDINGS_DIR / f"{session_id}.json"
        if recording_file.exists():
            recording_file.unlink()
            logger.info("Deleted recording: %s", session_id)
        else:
            logger.warning("Recording not found: %s", session_id)
