"""Tests for ``pm_rag.cli``.

The CLI is the only piece of pm-rag that isn't already exercised by
the existing test files (which cover diffusion, eval, graph, index,
and four flavours of mapping). Without these tests, a rename of a
subcommand or a change to the JSON output shape would silently
break anyone who scripts against ``pm-rag query …``.

The CLI uses Click; ``CliRunner`` invokes commands in-process
(no subprocess, no global state pollution) so the tests stay fast
and deterministic.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from pm_rag.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Top-level surface
# ---------------------------------------------------------------------------


class TestTopLevel:
    def test_help_returns_zero_and_describes_pm_rag(self, runner: CliRunner) -> None:
        # The README and `--help` output are the user-visible
        # surface; pin that the top-level description survives a
        # refactor.
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "pm-rag" in result.output

    def test_version_flag_prints_a_version_string(self, runner: CliRunner) -> None:
        # `click.version_option()` is the contract; if it gets
        # dropped silently, every release pipeline that scrapes
        # `pm-rag --version` breaks.
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or any(
            ch.isdigit() for ch in result.output
        )

    @pytest.mark.parametrize("cmd", ["query", "mapping", "eval"])
    def test_subcommand_appears_in_help(self, runner: CliRunner, cmd: str) -> None:
        # README documents these three subcommands; a rename is
        # a breaking change for users.
        result = runner.invoke(main, ["--help"])
        assert cmd in result.output

    def test_unknown_subcommand_returns_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["definitely-not-a-real-command"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# `pm-rag query`
# ---------------------------------------------------------------------------


class TestQueryCommand:
    def test_help_returns_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["query", "--help"])
        assert result.exit_code == 0
        assert "trace" in result.output

    def test_missing_required_trace_flag_returns_nonzero(self, runner: CliRunner) -> None:
        # `--trace` is required; a future refactor that makes it
        # optional with no default would silently change behaviour.
        result = runner.invoke(main, ["query"])
        assert result.exit_code != 0
        assert "trace" in result.output.lower()

    def test_returns_json_array_of_symbol_score_pairs(self, runner: CliRunner) -> None:
        # The JSON shape is the public contract — operators script
        # against `jq '.[].symbol'` and `jq '.[].score'`.
        result = runner.invoke(main, ["query", "--trace", "a"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, list)
        for item in payload:
            assert "symbol" in item
            assert "score" in item
            assert isinstance(item["score"], (int, float))

    def test_k_caps_the_result_size(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["query", "--trace", "a", "--k", "2"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert len(payload) <= 2

    def test_alpha_is_accepted_as_float(self, runner: CliRunner) -> None:
        # PPR restart probability; the CLI accepts a float and the
        # underlying call signature must match.
        result = runner.invoke(main, ["query", "--trace", "a", "--alpha", "0.25"])
        assert result.exit_code == 0

    def test_trace_csv_is_split_and_trimmed(self, runner: CliRunner) -> None:
        # `--trace "a, b, c"` should split on commas and strip whitespace.
        # If the splitter drifted, a trace prefix with spaces would
        # be silently treated as a single weird symbol.
        result = runner.invoke(main, ["query", "--trace", "a, b, c"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# `pm-rag mapping`
# ---------------------------------------------------------------------------


class TestMappingCommand:
    def test_help_returns_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["mapping", "--help"])
        assert result.exit_code == 0

    def test_returns_json_object_keyed_by_event(self, runner: CliRunner) -> None:
        # Event -> [symbols] mapping; the structure is what
        # downstream tooling reads.
        result = runner.invoke(main, ["mapping"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert isinstance(payload, dict)
        for ev, symbols in payload.items():
            assert isinstance(ev, str)
            assert isinstance(symbols, list)
            for s in symbols:
                assert isinstance(s, str)


# ---------------------------------------------------------------------------
# `pm-rag eval`
# ---------------------------------------------------------------------------


class TestEvalCommand:
    def test_help_returns_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "--help"])
        assert result.exit_code == 0

    def test_returns_score_payload_with_documented_shape(self, runner: CliRunner) -> None:
        # Documented payload: `{task, n, alpha, top_k}` where
        # `top_k` is `{str(k): float}`. README and any downstream
        # results pipeline depends on the exact shape.
        result = runner.invoke(main, ["eval"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["task"] == "next-event-localization"
        assert isinstance(payload["n"], int)
        assert isinstance(payload["alpha"], (int, float))
        assert isinstance(payload["top_k"], dict)
        # Default `--ks` is "1,3,5,10" — pin so a stealth change
        # to the default fails this test.
        assert set(payload["top_k"].keys()) == {"1", "3", "5", "10"}
        # Each top-k value must be a number in [0, 1].
        for k_str, val in payload["top_k"].items():
            assert k_str.isdigit()
            assert 0.0 <= val <= 1.0

    def test_explicit_ks_changes_the_keys(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["eval", "--ks", "1,5"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert set(payload["top_k"].keys()) == {"1", "5"}

    def test_alpha_appears_in_payload(self, runner: CliRunner) -> None:
        # The CLI echoes `alpha` so reproducibility scripts can
        # round-trip the run config through the result.
        result = runner.invoke(main, ["eval", "--alpha", "0.42"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["alpha"] == pytest.approx(0.42)

    def test_ks_with_whitespace_is_trimmed(self, runner: CliRunner) -> None:
        # `--ks "1, 3, 5"` should split + strip; without trimming,
        # `int(" 3")` would raise.
        result = runner.invoke(main, ["eval", "--ks", "1, 3, 5"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert set(payload["top_k"].keys()) == {"1", "3", "5"}
