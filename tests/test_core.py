import json
from datetime import date, datetime, timedelta, timezone

import pytest

from conjunctions import find_conjunctions
from conjunctions.core import _default_range, _to_utc, timezone_for

RALEIGH = {"latitude": 35.78, "longitude": -78.64}


def _find(bodies, events):
    return [ev for ev in events if ev["bodies"] == bodies]


def _utc(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


@pytest.mark.parametrize("value, expected", [
    ("2026-10-01", datetime(2026, 10, 1, tzinfo=timezone.utc)),
    ("2026-10-01T12:30:00Z", datetime(2026, 10, 1, 12, 30, tzinfo=timezone.utc)),
    ("2026-10-01T08:00:00-04:00", datetime(2026, 10, 1, 12, tzinfo=timezone.utc)),
    (date(2026, 10, 1), datetime(2026, 10, 1, tzinfo=timezone.utc)),
    (datetime(2026, 10, 1, 6), datetime(2026, 10, 1, 6, tzinfo=timezone.utc)),
])
def test_to_utc(value, expected):
    assert _to_utc(value) == expected


@pytest.mark.parametrize("kwargs", [
    {"start": "2026-10-02", "end": "2026-10-01"},
    {"start": "2026-10-01", "end": "2026-10-02", "min_bodies": 1},
    {"start": "2026-10-01", "end": "2026-10-02", "latitude": 35.0},
])
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        find_conjunctions(**kwargs)


def test_moon_jupiter_geocentric():
    data = json.loads(find_conjunctions("2026-10-05", "2026-10-08", max_separation_deg=1))
    assert data["observer"] is None
    (ev,) = _find(["Moon", "Jupiter"], data["conjunctions"])
    assert abs(_utc(ev["utc"]) - datetime(2026, 10, 6, 10, 23, tzinfo=timezone.utc)) < timedelta(minutes=10)
    assert ev["separation_deg"] < 0.5
    assert "visibility" not in ev


def test_planet_pair_and_cluster():
    events = json.loads(find_conjunctions("2026-06-01", "2026-10-31", max_separation_deg=2))["conjunctions"]
    (vj,) = _find(["Venus", "Jupiter"], events)
    assert vj["utc"].startswith("2026-06-09")
    assert 1.4 < vj["separation_deg"] < 1.8
    (mb,) = _find(["Mars", "Beehive (M44)"], events)
    assert mb["utc"].startswith("2026-10-11")
    assert mb["separation_deg"] < 0.2


def test_results_sorted_and_within_limits():
    events = json.loads(find_conjunctions("2026-01-01", "2026-04-01", 3, 20))["conjunctions"]
    assert events
    assert [e["utc"] for e in events] == sorted(e["utc"] for e in events)
    assert all(e["separation_deg"] <= 3 and e["sun_elongation_deg"] >= 20 for e in events)


def test_min_bodies_groups():
    events = json.loads(find_conjunctions("2026-06-15", "2026-06-20", min_bodies=3))["conjunctions"]
    assert events and all(len(e["bodies"]) >= 3 for e in events)
    (group,) = _find(["Moon", "Venus", "Beehive (M44)"], events)
    assert group["separation_deg"] <= 5


def test_pairs_inside_group_are_merged():
    events = json.loads(find_conjunctions("2026-06-15", "2026-06-20"))["conjunctions"]
    assert _find(["Moon", "Venus", "Beehive (M44)"], events)
    assert not _find(["Moon", "Venus"], events)


def test_cache(isolated_cache):
    first = find_conjunctions("2026-10-05", "2026-10-08")
    files = list(isolated_cache.glob("*.json"))
    assert len(files) == 1
    files[0].write_text('{"cached": true}')
    assert find_conjunctions("2026-10-05", "2026-10-08") == '{"cached": true}'
    assert find_conjunctions("2026-10-05", "2026-10-08", use_cache=False) == first


def test_cache_key_includes_observer(isolated_cache):
    find_conjunctions("2026-10-05", "2026-10-08")
    find_conjunctions("2026-10-05", "2026-10-08", **RALEIGH)
    assert len(list(isolated_cache.glob("*.json"))) == 2


def test_topocentric_visibility():
    data = json.loads(find_conjunctions("2026-10-01", "2026-11-01", 3, **RALEIGH))
    assert data["observer"]["latitude"] == RALEIGH["latitude"]
    events = data["conjunctions"]
    assert events
    for ev in events:
        vis = ev["visibility"]
        assert set(vis) == {"closest", "closest_visible", "after_sunset", "before_sunrise", "best"}
        assert vis["closest"]["utc"] == ev["utc"]
        assert set(vis["closest"]["positions"]) == set(ev["bodies"])
        if vis["closest_visible"]:
            assert vis["best"] == "closest_visible"
            assert vis["closest_visible"]["all_above_horizon"]
            assert vis["closest_visible"]["sun_alt_deg"] <= -6
        if vis["best"]:
            assert vis[vis["best"]]["all_above_horizon"]
        for key in ("after_sunset", "before_sunrise"):
            if vis[key]:
                # Sun ~1 hour from the horizon: comfortably below it, not deep night.
                assert -25 < vis[key]["sun_alt_deg"] < -5

    (pleiades,) = _find(["Moon", "Pleiades (M45)"], events)
    assert pleiades["visibility"]["best"] == "closest_visible"


def test_min_altitude_filters_low_views():
    events = json.loads(find_conjunctions("2026-10-01", "2026-11-01", 3, min_altitude_deg=20,
                                          **RALEIGH))["conjunctions"]
    for ev in events:
        snap = ev["visibility"]["closest_visible"]
        if snap:
            assert all(p["alt_deg"] >= 20 for p in snap["positions"].values())


def test_default_range():
    start, end = _default_range(None, None)
    assert start == _to_utc(datetime.now(timezone.utc).date())
    assert (end.year, end.month, end.day) == (start.year + 1, start.month, start.day)
    assert _default_range("2028-02-29", None)[1] == datetime(2029, 2, 28, tzinfo=timezone.utc)
    assert _default_range("2026-01-01", "2026-02-01")[1] == datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("lat, lon, tz", [
    (35.78, -78.64, "America/New_York"),
    (51.48, 0.0, "Europe/London"),
    (-33.87, 151.21, "Australia/Sydney"),
])
def test_timezone_for(lat, lon, tz):
    assert timezone_for(lat, lon) == tz


def test_observer_timezone_in_output():
    data = json.loads(find_conjunctions("2026-10-05", "2026-10-08", **RALEIGH))
    assert data["observer"]["timezone"] == "America/New_York"


def test_bright_stars():
    events = json.loads(find_conjunctions("2026-11-01", "2026-11-15", 1))["conjunctions"]
    assert _find(["Moon", "Regulus"], events)
    assert _find(["Moon", "Antares"], events)


def test_planets_only():
    kwargs = dict(start="2026-10-01", end="2026-11-15", max_separation_deg=3)
    full = json.loads(find_conjunctions(**kwargs))["conjunctions"]
    planets = json.loads(find_conjunctions(**kwargs, planets_only=True))["conjunctions"]
    names = {"Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn"}
    assert planets and len(planets) < len(full)
    assert all(set(e["bodies"]) <= names for e in planets)
    # Each planet-only pair also appears in the full run, possibly merged into a larger group.
    for ev in planets:
        assert any(set(ev["bodies"]) <= set(f["bodies"]) for f in full)


def test_fixed_separation():
    from conjunctions.core import _fixed_separation
    assert 1 < _fixed_separation("Aldebaran", "Hyades") < 2.5
    assert _fixed_separation("Regulus", "Pleiades (M45)") > 50


def test_cluster_preferred_over_star():
    # 2034 is in a lunar series passing through the Hyades, right by Aldebaran.
    events = json.loads(find_conjunctions("2034-01-01", "2034-04-01"))["conjunctions"]
    hyades = _find(["Moon", "Hyades"], events)
    assert len(hyades) >= 3
    assert not [e for e in events if "Aldebaran" in e["bodies"] and "Moon" in e["bodies"]]


def test_star_kept_when_no_cluster_nearby():
    # Regulus is far from every cluster, so its conjunctions are never dropped.
    events = json.loads(find_conjunctions("2026-11-01", "2026-11-15", 1))["conjunctions"]
    assert _find(["Moon", "Regulus"], events)


def test_prefer_clusters_unit():
    from conjunctions.core import _prefer_clusters
    pairs = [
        {"tt": 100.0, "bodies": frozenset({"Moon", "Hyades"}), "sep": 1.0},
        {"tt": 100.2, "bodies": frozenset({"Moon", "Aldebaran"}), "sep": 0.5},
        {"tt": 103.0, "bodies": frozenset({"Moon", "Aldebaran"}), "sep": 0.5},   # too far in time
        {"tt": 100.1, "bodies": frozenset({"Mars", "Aldebaran"}), "sep": 0.5},   # different body
        {"tt": 100.1, "bodies": frozenset({"Moon", "Regulus"}), "sep": 0.5},     # star not near Hyades
    ]
    kept = _prefer_clusters(pairs, 5.0)
    assert pairs[1] not in kept
    assert [p for p in pairs if p is not pairs[1]] == kept
    # With a tight limit Aldebaran no longer counts as "near" the Hyades centre.
    assert _prefer_clusters(pairs, 1.0) == pairs
