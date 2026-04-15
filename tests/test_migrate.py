"""Tests for mempalace.migrate — stub that replaced the ChromaDB migration tool."""

from mempalace.migrate import contains_palace_database, migrate


def test_migrate_always_prints_indentiagraph_message(capsys, tmp_path):
    """migrate() should always print that IndentiaGraph is now the backend."""
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()

    result = migrate(str(palace_dir))

    out = capsys.readouterr().out
    assert result is False
    assert "IndentiaGraph" in out


def test_migrate_returns_false_regardless_of_args(tmp_path):
    """migrate() always returns False — no actual migration is performed."""
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()

    assert migrate(str(palace_dir)) is False
    assert migrate(str(palace_dir), dry_run=True) is False
    assert migrate(str(palace_dir), confirm=True) is False
    assert migrate("/nonexistent/path") is False


def test_contains_palace_database_true_for_existing_dir(tmp_path):
    palace_dir = tmp_path / "palace"
    palace_dir.mkdir()

    assert contains_palace_database(str(palace_dir)) is True


def test_contains_palace_database_false_for_missing_dir(tmp_path):
    missing = tmp_path / "no-such-palace"
    assert contains_palace_database(str(missing)) is False
