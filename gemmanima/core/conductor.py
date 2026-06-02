from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from gemmanima.context.capsule import ContextRelevanceFilter
from gemmanima.core.conflict import ConflictResolver
from gemmanima.core.config import EngineConfig
from gemmanima.core.manifest import Manifest, ManifestStore
from gemmanima.core.model_registry import ModelRegistry
from gemmanima.core.renderer_profiles import RendererProfileManager
from gemmanima.core.schemas import ChatTurn, EngineResponse, GenerationPlan, JobStatus, Mode
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
        conflict_resolver: ConflictResolver | None = None,
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
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.planner_artifacts = PlannerArtifacts()
        self.model_registry = ModelRegistry(self.config)
        self.renderer_profiles = RendererProfileManager(self.config)
        self.plan_overrides = plan_overrides or {}
        self.pending_image_request: str | None = None

    def handle_user_message(self, text: str) -> EngineResponse:
        self.history.append(ChatTurn(role="user", content=text))
        route_text = text
        resumed_from_clarification = False

        if not self.planner.is_image_request(route_text):
            resume_request = self._clarification_resume_request(text)
            if resume_request:
                route_text = resume_request
                resumed_from_clarification = True
            else:
                self.pending_image_request = None
                return self._chat_response()

        if not self.planner.is_image_request(route_text):
            return self._chat_response()

        capsule_history = self.history if resumed_from_clarification else self.history[:-1]
        capsule = self.context_filter.build(capsule_history, route_text)
        if self.planner.needs_clarification(capsule):
            question = self.planner.clarification_question(capsule)
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.ASK_CLARIFY,
                user_request=route_text,
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
                clarification_required=True,
                job_id=manifest.job_id,
            )

        progress = ["route:image"]
        if resumed_from_clarification:
            progress.append("clarification:resume")
        progress.append("context:capsule")
        plan = None
        try:
            plan = self.planner.make_plan(capsule)
            if self.plan_overrides:
                plan = replace(plan, **self.plan_overrides)
            progress.append("planner:plan")
            return self._execute_generation_plan(route_text, capsule, plan, progress)
            conflict_report = self.conflict_resolver.resolve(capsule, plan)
            if conflict_report.blocks_generation():
                progress.append("conflict:blocked")
                manifest = Manifest.new(
                    session_id=self.session_id,
                    mode=Mode.GENERATE_IMAGE,
                    status=JobStatus.ASK_CLARIFY,
                    user_request=route_text,
                    context_capsule=capsule,
                    plan=plan,
                    models=self._model_manifest(),
                    hardware={
                        "primary_gpu": self.config.hardware.primary_gpu,
                        "secondary_gpu": self.config.hardware.secondary_gpu,
                    },
                    renderer={"conflict": conflict_report.to_json_dict()},
                )
                manifest_path = self.manifest_store.write(manifest)
                question = (
                    conflict_report.proposed_questions[0]
                    if conflict_report.proposed_questions
                    else "The reference and instruction conflict. Which should I preserve?"
                )
                self.pending_image_request = route_text
                self.history.append(ChatTurn(role="assistant", content=question))
                return EngineResponse(
                    mode=Mode.GENERATE_IMAGE,
                    status=JobStatus.ASK_CLARIFY,
                    message=question,
                    prompt=plan.prompt,
                    manifest_path=manifest_path,
                    progress=tuple(progress),
                    clarification_required=True,
                    conflict=conflict_report.to_json_dict(),
                    job_id=manifest.job_id,
                )
            progress.append("conflict:clear")
            conditioning = self.hidden_exit.encode(capsule, plan, conflict_report=conflict_report)
            progress.append("hiddenstage:conditioning")
            render_result = self.renderer.generate(plan, conditioning)
            progress.append("renderer:complete")
            self.pending_image_request = None
            renderer_is_dry_run = bool(getattr(self.renderer, "dry_run", True))
            bridge_checkpoint = conditioning.metadata.get("bridge_checkpoint", {})
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.COMPLETED,
                user_request=route_text,
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
                    "hiddenstage_conditioning": conditioning.to_json_dict(),
                },
                conditioning_metrics={
                    "measurement_policy": "observed_only",
                    "run_conditioning_mse": None,
                    "bridge_val_mse": bridge_checkpoint.get("val_mse"),
                    "measured": False,
                },
                warnings=list(render_result.warnings),
            )
            manifest_path = self.manifest_store.write(manifest)
            message = (
                "dry-run으로 이미지 생성 계획만 검증했습니다. "
                "실제 이미지는 생성하지 않았습니다."
            )
            if not renderer_is_dry_run:
                message = "이미지를 만들었습니다."
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
                user_request=route_text,
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

    def handle_generation_plan(self, text: str, plan: GenerationPlan) -> EngineResponse:
        self.history.append(ChatTurn(role="user", content=text))
        capsule = self.context_filter.build(self.history[:-1], text)
        progress = ["route:image", "chat:contract", "context:capsule", "planner:provided"]
        try:
            if self.plan_overrides:
                plan = replace(plan, **self.plan_overrides)
            return self._execute_generation_plan(text, capsule, plan, progress)
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
            message = "이미지 생성 계획을 실행하는 중 복구 가능한 오류가 발생했고 manifest에 기록했습니다."
            self.history.append(ChatTurn(role="assistant", content=message))
            return EngineResponse(
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.FAILED,
                message=message,
                prompt=plan.prompt,
                manifest_path=manifest_path,
                progress=tuple(progress),
                job_id=manifest.job_id,
            )

    def _execute_generation_plan(
        self,
        route_text: str,
        capsule,
        plan: GenerationPlan,
        progress: list[str],
    ) -> EngineResponse:
        plan.validate()
        conflict_report = self.conflict_resolver.resolve(capsule, plan)
        if conflict_report.blocks_generation():
            progress.append("conflict:blocked")
            manifest = Manifest.new(
                session_id=self.session_id,
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.ASK_CLARIFY,
                user_request=route_text,
                context_capsule=capsule,
                plan=plan,
                models=self._model_manifest(),
                hardware={
                    "primary_gpu": self.config.hardware.primary_gpu,
                    "secondary_gpu": self.config.hardware.secondary_gpu,
                },
                renderer={"conflict": conflict_report.to_json_dict()},
            )
            manifest_path = self.manifest_store.write(manifest)
            question = (
                conflict_report.proposed_questions[0]
                if conflict_report.proposed_questions
                else "The reference and instruction conflict. Which should I preserve?"
            )
            self.pending_image_request = route_text
            self.history.append(ChatTurn(role="assistant", content=question))
            return EngineResponse(
                mode=Mode.GENERATE_IMAGE,
                status=JobStatus.ASK_CLARIFY,
                message=question,
                prompt=plan.prompt,
                manifest_path=manifest_path,
                progress=tuple(progress),
                clarification_required=True,
                conflict=conflict_report.to_json_dict(),
                job_id=manifest.job_id,
            )

        progress.append("conflict:clear")
        conditioning = self.hidden_exit.encode(capsule, plan, conflict_report=conflict_report)
        progress.append("hiddenstage:conditioning")
        render_result = self.renderer.generate(plan, conditioning)
        progress.append("renderer:complete")
        self.pending_image_request = None
        renderer_is_dry_run = bool(getattr(self.renderer, "dry_run", True))
        bridge_checkpoint = conditioning.metadata.get("bridge_checkpoint", {})
        manifest = Manifest.new(
            session_id=self.session_id,
            mode=Mode.GENERATE_IMAGE,
            status=JobStatus.COMPLETED,
            user_request=route_text,
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
                "hiddenstage_conditioning": conditioning.to_json_dict(),
            },
            conditioning_metrics={
                "measurement_policy": "observed_only",
                "run_conditioning_mse": None,
                "bridge_val_mse": bridge_checkpoint.get("val_mse"),
                "measured": False,
            },
            warnings=list(render_result.warnings),
        )
        manifest_path = self.manifest_store.write(manifest)
        message = (
            "dry-run으로 이미지 생성 계획만 검증했습니다. "
            "실제 이미지는 생성하지 않았습니다."
        )
        if not renderer_is_dry_run:
            message = "이미지를 만들었습니다."
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

    def _chat_response(self) -> EngineResponse:
        response_text = "일반 채팅 경로입니다. 이미지 컴포넌트는 실행하지 않았습니다."
        self.history.append(ChatTurn(role="assistant", content=response_text))
        return EngineResponse(
            mode=Mode.CHAT,
            status=JobStatus.COMPLETED,
            message=response_text,
            progress=("route:chat",),
        )

    def _clarification_resume_request(self, text: str) -> str:
        if not self._looks_like_clarification_reply(text):
            return ""
        request = self.pending_image_request or self._last_blocked_image_request()
        if not request:
            return ""
        return f"{request} {text}"

    def _last_blocked_image_request(self) -> str:
        saw_conflict_question = False
        for turn in reversed(self.history[:-1]):
            if turn.role == "assistant" and self._looks_like_conflict_question(turn.content):
                saw_conflict_question = True
                continue
            if saw_conflict_question and turn.role == "user" and self.planner.is_image_request(turn.content):
                return turn.content
        return ""

    def _looks_like_conflict_question(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "reference" in lowered
            and ("request asks" in lowered or "instruction" in lowered or "conflict" in lowered)
            and ("preserve" in lowered or "change" in lowered)
        )

    def _looks_like_clarification_reply(self, text: str) -> bool:
        lowered = text.lower()
        return any(
            token in lowered
            for token in (
                "change",
                "preserve",
                "keep",
                "use reference",
                "use the reference",
                "use request",
                "use the request",
            )
        )

    def _model_manifest(self) -> dict[str, object]:
        return {
            "gemma_planner_adapter": str(self.config.models.gemma_planner_adapter),
            "vision_embedding": str(self.config.models.gemma_vision_embedding),
            "anima_diffusion_model": str(self.config.models.anima_diffusion_model),
            "anima_vae": str(self.config.models.anima_vae),
            "hiddenstage_bridge": str(self.config.models.hiddenstage_bridge),
            "planner_eval_loss": self.planner_artifacts.eval_loss,
            "planner_eval_threshold": self.planner_artifacts.eval_threshold,
            "hiddenstage_exit": "trained_bridge_interface",
            "registry_health": self.model_registry.health(),
            "model_parts": self.model_registry.grouped_health(),
        }
