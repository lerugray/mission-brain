from typer.testing import CliRunner

from mission_brain.cli import app


def test_version_exits_zero_with_nonempty_output() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() != ""
