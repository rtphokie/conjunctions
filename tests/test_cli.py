import json
from datetime import datetime, timezone

import pytest

from conjunctions.cli import main


def test_text_output_geocentric(capsys):
    assert main(["2026-10-05", "2026-10-08", "-s", "1"]) == 0
    out = capsys.readouterr().out
    assert "Geocentric" in out
    assert "Moon · Jupiter" in out


def test_text_output_with_observer(capsys):
    main(["2026-10-27", "2026-10-29", "-s", "1", "--lat", "35.78", "--lon", "-78.64",
          "--tz", "America/New_York"])
    out = capsys.readouterr().out
    assert "Observer 35.780°N 78.640°W" in out
    assert "Moon · Pleiades (M45)" in out
    assert "best (closest while visible)" in out
    assert "EDT" in out
    assert "alt" in out and "az" in out


def test_verbose_shows_twilight_snapshots(capsys):
    main(["2026-10-27", "2026-10-29", "-s", "1", "--lat", "35.78", "--lon", "-78.64", "-v"])
    out = capsys.readouterr().out
    assert "1h after sunset" in out
    assert "1h before sunrise" in out


def test_json_output(capsys):
    main(["2026-10-05", "2026-10-08", "-s", "1", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["conjunctions"][0]["bodies"]


def test_min_bodies_flag(capsys):
    main(["2026-06-15", "2026-06-20", "-n", "3", "--compact"])
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    assert all(len(e["bodies"]) >= 3 for e in json.loads(out)["conjunctions"])


def test_default_dates(capsys):
    main(["-s", "0.5", "--json"])
    data = json.loads(capsys.readouterr().out)
    today = datetime.now(timezone.utc).date()
    assert data["start"] == f"{today.isoformat()}T00:00:00Z"
    assert data["end"].startswith(str(today.year + 1))


def test_timezone_from_location(capsys):
    main(["2026-10-27", "2026-10-29", "-s", "1", "--lat", "51.48", "--lon", "0.0"])
    out = capsys.readouterr().out
    assert "(Europe/London)" in out
    assert "BST" in out or "GMT" in out


def test_tz_override(capsys):
    main(["2026-10-27", "2026-10-29", "-s", "1", "--lat", "35.78", "--lon", "-78.64", "--tz", "UTC"])
    assert "UTC" in capsys.readouterr().out


def test_lat_without_lon_errors():
    with pytest.raises(SystemExit) as exc:
        main(["--lat", "35"])
    assert exc.value.code == 2


def test_planets_only_flag(capsys):
    main(["2026-10-01", "2026-11-15", "-p", "--json"])
    events = json.loads(capsys.readouterr().out)["conjunctions"]
    assert events
    planets = {"Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"}
    assert all(set(e["bodies"]) <= planets for e in events)


def test_paths(capsys, isolated_cache):
    main(["--paths"])
    data = json.loads(capsys.readouterr().out)
    assert data["cache_dir"] == str(isolated_cache)
