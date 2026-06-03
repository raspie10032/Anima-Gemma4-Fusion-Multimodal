from gemmanima.rendering.backends import audit_renderer_backend, renderer_backend_profile


def test_external_script_backend_profile_is_current_default() -> None:
    profile = renderer_backend_profile("external_script")

    assert profile.name == "external_script"
    assert isinstance(profile.ready, bool)
    assert profile.execution == "subprocess"


def test_in_process_backend_reports_import_boundary() -> None:
    profile = renderer_backend_profile("in_process")

    assert profile.name == "in_process"
    assert profile.execution == "in_process"
    assert isinstance(profile.ready, bool)
    assert isinstance(profile.dependency_ready, bool)
    assert "comfy_import" in profile.checks
    assert "comfy_bootstrap_module" in profile.checks
    assert "gemma_hidden_provider_module" in profile.checks
    assert "gemma_model_safetensors" in profile.checks
    assert "t5_tokenizer_provider_module" in profile.checks
    assert "sampler_runtime_module" in profile.checks
    assert "adapter_attach_module" in profile.checks
    assert "port_gemma_hidden_provider" not in profile.next_steps
    assert "port_t5_tokenizer_provider" not in profile.next_steps
    assert "port_sampler_and_vae_decode" not in profile.next_steps
    assert "legacy_script_removed" in profile.next_steps


def test_renderer_backend_audit_contains_both_paths() -> None:
    audit = audit_renderer_backend()

    assert set(audit) == {"external_script", "in_process", "local_worker"}
    assert isinstance(audit["external_script"]["ready"], bool)
    assert "checks" in audit["in_process"]
    assert audit["local_worker"]["execution"] == "subprocess"
