# -*- coding: utf-8 -*-
"""Diplomacy engine config (Avalon-style)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class DiplomacyBasicConfig:
    """AvalonBasicConfig-like config for Diplomacy."""
    power_names: List[str]
    map_name: str = "standard"
    max_phases: int = 20
    negotiation_rounds: int = 3
    seed: int = 42
    language: str = "en"
    human_power: Optional[str] = None  # e.g., "ENGLAND" when participate

    @classmethod
    def default(cls) -> "DiplomacyBasicConfig":
        return cls(
            power_names=["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"],
            map_name="standard",
            max_phases=20,
            negotiation_rounds=3,
            seed=42,
            language="en",
            human_power=None,
        )
