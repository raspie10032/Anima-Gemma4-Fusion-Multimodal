from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from gemmanima.context.capsule import ContextRelevanceFilter
from gemmanima.core.config import EngineConfig
from gemmanima.core.manifest import Manifest, ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.renderer_profiles import RendererProfileManager
from gemmanima.core.schemas import ChatTurn, EngineResponse, JobStatus, Mode
from gemmanima.modules.anima_renderer import AnimaRendererAdapter
from gemmanima.modules.gemma_planner import GemmaPlannerAdapter, PlannerArtifacts
from gemmanima.modules.hiddenstage_exit import HiddenStageExit


class GemmAnimaConductor:
    def __init__(
        self,
        *,
        session_id: str | None = None,
        manifest_root: str | Path = "runs/manifests",
        image_root: str | Path = "runs/images",
        planner: GemmaPlannerAdapter | None = None,
        context_filter: ContextRelevanceFilter | None = None,
        hidden_exit: HiddenStageExit | None = None,
        renderer: AnimaRendererAdapter | None = None,
        config: EngineConfig | None = None,
        plan_overrides: dict[str, object] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.session_id = session_id or uuid4().hex
        self.history: list[ChatTurn] = []
        self.manifest_store = ManifestStore(manifest_root)
        self.planner = planner or GemmaPlannerAdapter(self.config)
        self.context_filter = context_filter or ContextRelevanceFilter()
        self.hidden_exit = hidden_exit or HiddenStageExit(self.config)
        self.renderer = renderer or AnimaRendererAdapter(image_root)
        self.planner_artifacts = PlannerArtifacts()
        self.model_registry = ModelRegistry(self.config)
        self.renderer_profiles = RendererProfileManager(self.config)
        self.plan_overrides = plan_overrides or {}

    def handle_user_message(self, text: str) -> EngineResponse:
        self.history.append(ChatTurn(role="user", content=text))

        if not self.planner.is_image_request(text):
            response_text = "일반 채팅 경로입니다. 이미지 컴포넌트는 실행하지 않았습니다."
            self.history.append(ChatTurn(role="assistant", content=response_text))
            return EngineResponse(
                mode=Mode.CHAT,
                status=JobStatus.COMPLETED,
                message=response_text,
                progress=("route:chat",),
            )

        capsule = self.context_filter.build(self.history[:-1], text)
        if self.planner.needs_clarification(capsule):
            question = self.planner.clarification_question(capsule)
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.ASK_CLARIFY,
                user_request=text,
                context_capsule=capsule,
                models=self._model_manifest(),
            )
            manifest_path = self.manifest_store.write(manifest)
            self.history.append(ChatTurn(role="assistant", content=question))
            return EngineResponse(
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.ASK_CLARIFY,
                message=question,
                manifest_path=manifest_path,
                progress=("route:image", "context:capsule", "planner:clarify"),
                job_id=manifest.job_id,
            )

        progress = ["route:image", "context:capsule"]
        plan = None
        try:
            plan = self.planner.make_plan(capsule)
            if self.plan_overrides:
                plan = replace(plan, **self.plan_overrides)
            plan.validate()
            progress.append("planner:plan")
            conditioning = self.hidden_exit.encode(capsule, plan)
            progress.append("hiddenstage:conditioning")
            render_result = self.renderer.generate(plan, conditioning)
            progress.append("renderer:complete")
            renderer_is_dry_run = bool(getattr(self.renderer, "dry_run", True))
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.COMPLETED,
                user_request=text,
                context_capsule=capsule,
                plan=plan,
                render_result=render_result,
                models=self._model_manifest(),
                hardware={
                    "primary_gpu": self.config.hardware.primary_gpu,
                    "secondary_gpu": self.config.hardware.secondary_gpu,
                },
                renderer={
                    "profile": plan.renderer_profile,
                    "dry_run": renderer_is_dry_run,
                    "hiddenstage_conditioning": {
                        "source": conditioning.source,
                        "shape": list(conditioning.shape),
                        "metadata": conditioning.metadata,
                    },
                },
                warnings=list(render_result.warnings),
            )
            manifest_path = self.manifest_store.write(manifest)
            message = f"이미지 생성 계획을 만들고 dry-run 렌더를 완료했습니다. prompt: {plan.prompt}"
            if not renderer_is_dry_run:
                message = f"이미지 생성을 완료했습니다. prompt: {plan.prompt}"
            self.history.append(ChatTurn(role="assistant", content=message))
            return EngineResponse(
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.COMPLETED,
                message=message,
                prompt=plan.prompt,
                manifest_path=manifest_path,
                output_path=render_result.output_path,
                progress=tuple(progress),
                job_id=manifest.job_id,
            )
        except Exception as exc:
            progress.append("error:recoverable")
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.FAILED,
                user_request=text,
                context_capsule=capsule,
                plan=plan,
                models=self._model_manifest(),
                hardware={
                    "primary_gpu": self.config.hardware.primary_gpu,
                    "secondary_gpu": self.config.hardware.secondary_gpu,
                },
                renderer={"dry_run": True},
                error=f"{type(exc).__name__}: {exc}",
            )
            manifest_path = self.manifest_store.write(manifest)
            message = "이미지 생성 경로에서 복구 가능한 오류가 발생했고 manifest에 기록했습니다."
            self.history.append(ChatTurn(role="assistant", content=message))
            return EngineResponse(
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.FAILED,
                message=message,
                prompt=plan.prompt if plan else None,
                manifest_path=manifest_path,
                progress=tuple(progress),
                job_id=manifest.job_id,
            )

    def _model_manifest(self) -> dict[str, object]:
        return {
            "gemma_planner_adapter": str(self.config.models.gemma_planner_adapter),
            "vision_embedding": str(self.config.models.gemma_vision_embedding),
            "anima_diffusion_model": str(self.config.models.anima_diffusion_model),
            "anima_text_encoder": str(self.config.models.anima_text_encoder),
            "anima_vae": str(self.config.models.anima_vae),
            "hiddenstage_bridge": str(self.config.models.hiddenstage_bridge),
            "planner_eval_loss": self.planner_artifacts.eval_loss,
            "planner_eval_threshold": self.planner_artifacts.eval_threshold,
            "hiddenstage_exit": "trained_bridge_interface",
            "registry_health": self.model_registry.health(),
        }
