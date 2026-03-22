from src.utils.git_utils import format_model_version, get_git_info


def test_get_git_info_never_raises():
    """get_git_info() nunca deve levantar excecao."""
    info = get_git_info()
    assert isinstance(info, dict)
    assert "commit_hash" in info
    assert "is_dirty" in info


def test_format_model_version_contains_run_id():
    """model_version deve conter o run_id."""
    fake_git = {"commit_short": "abc12345", "is_dirty": False}
    version = format_model_version(fake_git, "20240319143022")
    assert "20240319143022" in version
    assert "abc12345" in version
    assert "DIRTY" not in version


def test_format_model_version_dirty_flag():
    """Codigo nao commitado deve aparecer na versao."""
    fake_git = {"commit_short": "abc12345", "is_dirty": True}
    version = format_model_version(fake_git, "20240319143022")
    assert "DIRTY" in version
