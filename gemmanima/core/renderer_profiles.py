from __future__ import annotations

from gemmanima.core.config import EngineConfig, RendererProfile


class RendererProfileManager:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def get(self, name: str) -> RendererProfile:
        return self.config.profile(name)

    def clamp(self, *, profile_name: str, width: int, height: int, steps: int, cfg: float) -> dict[str, int | float | str]:
        profile = self.get(profile_name)
        max_side = max(profile.width, profile.height)
        clamped_width = min(width, max_side)
        clamped_height = min(height, max_side)
        clamped_width -= clamped_width % 8
        clamped_height -= clamped_height % 8
        return {
            "profile": profile.name,
            "precision": profile.precision,
            "width": max(256, clamped_width),
            "height": max(256, clamped_height),
            "steps": min(max(1, steps), profile.steps),
            "cfg": min(max(0.1, cfg), profile.cfg),
        }
