"""Conversation state plus JSON persistence, so a session can be resumed."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .conversation import Message, message_from_dict, message_to_dict


@dataclass
class Session:
    system: str
    messages: List[Message] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    compactions: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def save(self, path: str) -> None:
        data = {k: v for k, v in self.__dict__.items() if k != "messages"}
        data["messages"] = [message_to_dict(m) for m in self.messages]
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path) as f:
            data = json.load(f)
        data["messages"] = [message_from_dict(m) for m in data["messages"]]
        return cls(**data)
