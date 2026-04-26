"""Eval harness tests — skipped: mission-brain Phase 2 Step 2 does not ship eval/."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="eval framework (mission_brain.eval) not included in mission-brain template",
)
