from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MultimodalSample:
    """Canonical representation of a single multimodal dataset sample."""

    sample_id: str
    source: str

    image_path: Optional[str] = None
    text: Optional[str] = None

    hate_label: Optional[int] = None
    sarcasm_label: Optional[int] = None

    target_group: Optional[List[str]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)