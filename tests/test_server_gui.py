from gemmanima.server import GemmAnimaRequestHandler
from gemmanima.ui import GUI_HTML


def test_gui_html_contains_api_hooks() -> None:
    assert "GemmAnima Console" in GUI_HTML
    assert "/v1/health" in GUI_HTML
    assert "/v1/chat" in GUI_HTML


def test_server_root_uses_gui_html() -> None:
    assert GemmAnimaRequestHandler.base_dir is not None
    assert "GemmAnima Console" in GUI_HTML
