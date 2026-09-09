"""Exercise the public CLI against real, isolated persistence."""
import json

import pytest

from prismcognition.cli import main


@pytest.mark.parametrize("options_first", [False, True])
def test_cli_shorthand(tmp_path, capsys, options_first):
    options = ["--json", "--risk-level", "LOW", "--data-dir", str(tmp_path)]
    args = [*options, "Measure rainfall"] if options_first else ["Measure rainfall", *options]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["inquiry_text"] == "Measure rainfall"


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "ingest-evidence" in capsys.readouterr().out


@pytest.mark.parametrize("recommend", [False, True])
def test_cli_deliberate_show_list_replay(tmp_path, capsys, recommend):
    options = ["--data-dir", str(tmp_path)]
    args = ["deliberate", "Measure rainfall", "--json", *options]
    if recommend:
        args.append("--recommend")
    assert main(args) == 0
    original = json.loads(capsys.readouterr().out)
    identifier = original["deliberation_id"]
    assert bool(original["optional_recommendation"]) == recommend
    assert main(["show", identifier, "--json", *options]) == 0
    assert json.loads(capsys.readouterr().out) == original
    assert main(["list", *options]) == 0
    assert capsys.readouterr().out.strip() == identifier
    assert main(["replay", identifier, "--json", *options]) == 0
    assert json.loads(capsys.readouterr().out)["inquiry_text"] == "Measure rainfall"
    assert main(["show", identifier, *options]) == 0
    assert "Measure rainfall" in capsys.readouterr().out


def test_cli_evidence_ingestion_preserves_previous_rows(tmp_path, capsys):
    for identifier in ("first", "second"):
        assert main(["ingest-evidence", "--record-id", identifier,
                     "--domain-key", "Inquiry.thesis_holds", "--polarity", "1",
                     "--regime", "EMPIRICAL", "--score", "0.8",
                     "--source-ref", "measurement://1", "--data-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "evidence.json").read_text())
    assert {row["record_id"] for row in payload["records"]} == {"first", "second"}
    assert all(row["support_score"] == 0.8 for row in payload["records"])


def test_cli_server_configuration_is_forwarded(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("prismcognition.api.app.serve", lambda **kwargs: calls.append(kwargs))
    assert main(["serve", "--host", "127.0.0.1", "--port", "9876", "--data-dir", str(tmp_path)]) == 0
    assert calls == [dict(host="127.0.0.1", port=9876, data_dir=str(tmp_path), live=False)]
