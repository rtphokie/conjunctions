"""Find conjunctions (appulses) of the Moon, naked-eye planets, bright stars and star clusters.

A conjunction is a local minimum of the apparent angular separation between two
objects, kept when that minimum is within a threshold. Groups of three or more
objects are found by merging overlapping pair conjunctions; a group's separation
is the largest pairwise separation among its members.

Positions are geocentric unless an observer location is given, in which case they
are topocentric and each event gets altitude/azimuth snapshots for viewing.

Oppositions of the superior naked-eye planets (Mars, Jupiter, Saturn) are found
separately: the moment the planet's geocentric ecliptic longitude is 180 degrees
from the Sun's.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
from platformdirs import site_cache_dir, user_cache_dir
from skyfield import almanac
from skyfield.api import Loader, Star, wgs84
from skyfield.magnitudelib import planetary_magnitude
from skyfield.searchlib import find_minima
from timezonefinder import TimezoneFinder

APP_NAME = "conjunctions"
EPHEMERIS = "de440s.bsp"  # covers 1849-2150, ~32 MB
CACHE_VERSION = 5

# Name -> ephemeris target
MOVING_BODIES = {
    "Moon": "moon",
    "Mercury": "mercury",
    "Venus": "venus",
    "Mars": "mars barycenter",
    "Jupiter": "jupiter barycenter",
    "Saturn": "saturn barycenter",
}

# Name -> (RA hours, Dec degrees), ICRS/J2000 cluster centres
STAR_CLUSTERS = {
    "Pleiades (M45)": (3.7903, 24.1167),
    "Hyades": (4.4500, 15.8700),
    "Beehive (M44)": (8.6733, 19.6717),
    "M35": (6.1483, 24.3333),
}

# Name -> (RA hours, Dec degrees), ICRS/J2000; bright stars the Moon and planets can pass
BRIGHT_STARS = {
    "Aldebaran": (4.598678, 16.509306),
    "Pollux": (7.755264, 28.026194),
    "Castor": (7.576628, 31.888278),
    "Regulus": (10.139531, 11.967194),
    "Spica": (13.419883, -11.161333),
    "Antares": (16.490128, -26.432000),
}

FIXED_OBJECTS = {**STAR_CLUSTERS, **BRIGHT_STARS}
# Planets that can reach opposition (outside Earth's orbit)
OPPOSITION_BODIES = ("Mars", "Jupiter", "Saturn")
BODY_ORDER = list(MOVING_BODIES) + list(FIXED_OBJECTS)

# Coarse sampling step (days) for minimum search; the Moon moves ~13 deg/day.
MOON_STEP_DAYS = 0.1
PLANET_STEP_DAYS = 1.0
# Pair conjunctions sharing a body within this many days may merge into a group.
LINK_DAYS = 1.0
# Closest approach to Earth is searched within +/- this many days of opposition.
CLOSEST_APPROACH_WINDOW_DAYS = 30.0
# Visibility search: +/- this many days around closest approach, at this step.
VIS_WINDOW_DAYS = 1.0
VIS_STEP_DAYS = 2 / 1440
REFRACTION_TEMP_C = 10.0


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return os.access(path, os.W_OK)


def data_dir() -> Path:
    """Directory holding skyfield assets (ephemeris files).

    Order: $CONJUNCTIONS_DATA_DIR, /var/data, site-wide cache, per-user cache.
    A directory already containing the ephemeris wins even if read-only.
    """
    candidates = []
    if env := os.environ.get("CONJUNCTIONS_DATA_DIR"):
        candidates.append(Path(env))
    candidates += [
        Path("/var/data"),
        Path(site_cache_dir("skyfield")),
        Path(user_cache_dir("skyfield")),
    ]
    for d in candidates:
        if (d / EPHEMERIS).is_file():
            return d
    for d in candidates:
        if _writable_dir(d):
            return d
    raise RuntimeError("no writable directory available for skyfield data")


def cache_dir() -> Path:
    """Directory holding cached results: $CONJUNCTIONS_CACHE_DIR, site cache, or user cache."""
    candidates = []
    if env := os.environ.get("CONJUNCTIONS_CACHE_DIR"):
        candidates.append(Path(env))
    candidates += [Path(site_cache_dir(APP_NAME)), Path(user_cache_dir(APP_NAME))]
    for d in candidates:
        if _writable_dir(d):
            return d
    raise RuntimeError("no writable directory available for result cache")


@cache
def timezone_for(latitude: float, longitude: float) -> str:
    """IANA time zone name for a location (offline lookup); "UTC" if unknown."""
    return TimezoneFinder().timezone_at(lat=latitude, lng=longitude) or "UTC"


def _default_range(start, end) -> tuple[datetime, datetime]:
    """Start defaults to today (UTC midnight); end to one year after start."""
    start_dt = _to_utc(start) if start is not None else _to_utc(datetime.now(timezone.utc).date())
    if end is not None:
        return start_dt, _to_utc(end)
    try:
        return start_dt, start_dt.replace(year=start_dt.year + 1)
    except ValueError:  # Feb 29
        return start_dt, start_dt.replace(year=start_dt.year + 1, day=28)


def _to_utc(value: str | date | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(t) -> str:
    return t.utc_datetime().strftime("%Y-%m-%dT%H:%M:%SZ")


class _Sky:
    def __init__(self, latitude: float | None, longitude: float | None, elevation_m: float):
        load = Loader(str(data_dir()), verbose=False)
        self.ts = load.timescale()
        self.eph = eph = load(EPHEMERIS)
        self.earth, self.sun = eph["earth"], eph["sun"]
        self.objects = {name: eph[key] for name, key in MOVING_BODIES.items()}
        self.objects.update(
            {name: Star(ra_hours=ra, dec_degrees=dec) for name, (ra, dec) in FIXED_OBJECTS.items()}
        )
        self.topocentric = latitude is not None
        self.observer = (
            self.earth + wgs84.latlon(latitude, longitude, elevation_m) if self.topocentric else self.earth
        )

    def apparent(self, names, t):
        at = self.observer.at(t)
        return [at.observe(self.objects[n]).apparent() for n in names]

    def spread(self, names, t):
        """Largest pairwise separation (degrees) among ``names`` at ``t``."""
        pos = self.apparent(names, t)
        return np.max([a.separation_from(b).degrees for a, b in combinations(pos, 2)], axis=0)

    def sun_elongation(self, names, t):
        sun = self.observer.at(t).observe(self.sun).apparent()
        return min(p.separation_from(sun).degrees for p in self.apparent(names, t))

    def step_days(self, names):
        return MOON_STEP_DAYS if "Moon" in names else PLANET_STEP_DAYS

    def pair_minima(self, t0, t1, max_sep, planets_only=False):
        pairs = list(combinations(MOVING_BODIES, 2))
        if not planets_only:
            pairs += [(body, fixed) for body in MOVING_BODIES for fixed in FIXED_OBJECTS]
        events = []
        for names in pairs:
            f = lambda t, names=names: self.spread(names, t)
            f.step_days = self.step_days(names)
            for t, sep in zip(*find_minima(t0, t1, f)):
                if sep <= max_sep:
                    events.append({"tt": t.tt, "bodies": frozenset(names), "sep": float(sep)})
        return events

    def group_minimum(self, names, tt0, tt1):
        """Time (TT JD) and value of the smallest spread of ``names`` within [tt0, tt1]."""
        step = 0.02
        grid = self.ts.tt_jd(np.arange(tt0, tt1 + step, step))
        values = self.spread(names, grid)
        i = int(np.argmin(values))
        best_tt, best = grid.tt[i], float(values[i])
        f = lambda t: self.spread(names, t)
        f.step_days = step / 2
        times, seps = find_minima(self.ts.tt_jd(best_tt - 2 * step), self.ts.tt_jd(best_tt + 2 * step), f)
        if len(seps) and seps.min() < best:
            j = int(np.argmin(seps))
            best_tt, best = times[j].tt, float(seps[j])
        return best_tt, best

    def snapshot(self, names, t):
        at = self.observer.at(t)
        positions = {}
        for n in names:
            alt, az, _ = at.observe(self.objects[n]).apparent().altaz(temperature_C=REFRACTION_TEMP_C)
            positions[n] = {"alt_deg": round(alt.degrees, 2), "az_deg": round(az.degrees, 2)}
        sun_alt = at.observe(self.sun).apparent().altaz(temperature_C=REFRACTION_TEMP_C)[0].degrees
        snap = {"utc": _iso(t)}
        if len(names) > 1:
            snap["separation_deg"] = round(float(self.spread(names, t)), 4)
        return snap | {
            "sun_alt_deg": round(float(sun_alt), 2),
            "all_above_horizon": all(p["alt_deg"] > 0 for p in positions.values()),
            "positions": positions,
        }

    def closest_visible(self, names, t, min_alt, max_sun_alt):
        """Closest approach within +/- VIS_WINDOW_DAYS while all bodies are above
        ``min_alt`` and the Sun is below ``max_sun_alt``."""
        n = round(VIS_WINDOW_DAYS / VIS_STEP_DAYS)
        grid = self.ts.tt_jd(t.tt + np.arange(-n, n + 1) * VIS_STEP_DAYS)
        at = self.observer.at(grid)
        ok = at.observe(self.sun).apparent().altaz()[0].degrees <= max_sun_alt
        for p in self.apparent(names, grid):
            ok &= p.altaz()[0].degrees >= min_alt
        if not ok.any():
            return None
        seps = np.where(ok, self.spread(names, grid), np.inf)
        return self.snapshot(names, grid[int(np.argmin(seps))])

    def sun_event_offset(self, t, finder, offset_hours):
        """Snapshot time: the Sun event (rise/set) nearest ``t``, shifted by ``offset_hours``."""
        times, ok = finder(self.observer, self.sun, self.ts.tt_jd(t.tt - 1), self.ts.tt_jd(t.tt + 1))
        times = [ti for ti, y in zip(times, ok) if y]
        if not times:
            return None
        nearest = min(times, key=lambda ti: abs(ti.tt - t.tt))
        return self.ts.tt_jd(nearest.tt + offset_hours / 24)


    def oppositions(self, t0, t1):
        """Opposition events of OPPOSITION_BODIES between ``t0`` and ``t1``."""
        events = []
        for name in OPPOSITION_BODIES:
            body = self.objects[name]
            f = almanac.oppositions_conjunctions(self.eph, body)
            times, kinds = almanac.find_discrete(t0, t1, f)
            for t in times[kinds == 1]:
                geo = self.earth.at(t).observe(body)
                event = {
                    "utc": _iso(t),
                    "body": name,
                    "distance_au": round(float(geo.distance().au), 4),
                    "magnitude": round(float(planetary_magnitude(geo)), 2),
                    "sun_elongation_deg": round(float(self.sun_elongation([name], t)), 2),
                    "closest_approach": self.closest_approach(name, t),
                }
                if self.topocentric:
                    event["visibility"] = self.transit(name, t)
                events.append(event)
        return events

    def closest_approach(self, name, t):
        """Time and distance of the planet's minimum distance from Earth near ``t``."""
        body = self.objects[name]
        f = lambda t: self.earth.at(t).observe(body).distance().au
        f.step_days = 1.0
        w = CLOSEST_APPROACH_WINDOW_DAYS
        times, dists = find_minima(self.ts.tt_jd(t.tt - w), self.ts.tt_jd(t.tt + w), f)
        i = int(np.argmin(dists))
        return {"utc": _iso(times[i]), "distance_au": round(float(dists[i]), 4)}

    def transit(self, name, t):
        """Snapshot at the planet's meridian transit nearest ``t`` (highest in the sky)."""
        times = almanac.find_transits(self.observer, self.objects[name],
                                      self.ts.tt_jd(t.tt - 1), self.ts.tt_jd(t.tt + 1))
        nearest = min(times, key=lambda ti: abs(ti.tt - t.tt))
        return {"transit": self.snapshot([name], nearest)}


def _fixed_separation(a: str, b: str) -> float:
    """Angular separation (degrees) between two fixed objects from their catalogue positions."""
    (ra1, d1), (ra2, d2) = (np.radians([FIXED_OBJECTS[n][0] * 15, FIXED_OBJECTS[n][1]]) for n in (a, b))
    cos = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(ra1 - ra2)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _prefer_clusters(pairs: list[dict], max_sep: float) -> list[dict]:
    """Drop a body's conjunction with a bright star when the same body has a
    conjunction, within LINK_DAYS, with a star cluster lying within ``max_sep``
    of that star (e.g. Aldebaran in front of the Hyades)."""
    near = {
        star: {c for c in STAR_CLUSTERS if _fixed_separation(star, c) <= max_sep} for star in BRIGHT_STARS
    }
    cluster_events = [p for p in pairs if p["bodies"] & STAR_CLUSTERS.keys()]

    def shadowed(p):
        (star,) = p["bodies"] & BRIGHT_STARS.keys() or {None}
        if not star or not near[star]:
            return False
        moving = p["bodies"] - {star}
        return any(
            c["bodies"] & near[star] and c["bodies"] - STAR_CLUSTERS.keys() == moving
            and abs(c["tt"] - p["tt"]) <= LINK_DAYS
            for c in cluster_events
        )

    return [p for p in pairs if not shadowed(p)]


def _group(sky: _Sky, pairs: list[dict], min_bodies: int, max_sep: float) -> list[tuple]:
    """Merge pair conjunctions into maximal groups. Returns (tt, names, spread) tuples."""
    pairs = _prefer_clusters(pairs, max_sep)
    parent = list(range(len(pairs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(len(pairs)), 2):
        a, b = pairs[i], pairs[j]
        if a["bodies"] & b["bodies"] and abs(a["tt"] - b["tt"]) <= LINK_DAYS:
            parent[root(i)] = root(j)

    components: dict[int, list[dict]] = {}
    for i, p in enumerate(pairs):
        components.setdefault(root(i), []).append(p)

    groups = []
    for comp in components.values():
        bodies = sorted(set().union(*(p["bodies"] for p in comp)), key=BODY_ORDER.index)
        kept: list[frozenset] = []
        for k in range(len(bodies), max(min_bodies, 2) - 1, -1):
            for subset in combinations(bodies, k):
                s = frozenset(subset)
                if any(s <= g for g in kept):
                    continue
                inside = [p for p in comp if p["bodies"] <= s]
                if not inside:
                    continue
                if k == 2:
                    groups += [(p["tt"], subset, p["sep"]) for p in inside]
                    kept.append(s)
                    continue
                tts = [p["tt"] for p in inside]
                tt, spread = sky.group_minimum(subset, min(tts) - 0.5, max(tts) + 0.5)
                if spread <= max_sep:
                    groups.append((tt, subset, spread))
                    kept.append(s)
    return groups


def _compute(start, end, max_sep, min_elong, min_bodies, site, min_alt, max_sun_alt,
             planets_only) -> list[dict]:
    sky = _Sky(*site)
    t0, t1 = sky.ts.from_datetime(start), sky.ts.from_datetime(end)

    events = []
    for tt, names, spread in _group(sky, sky.pair_minima(t0, t1, max_sep, planets_only), min_bodies, max_sep):
        if not t0.tt <= tt <= t1.tt:
            continue
        t = sky.ts.tt_jd(tt)
        elong = sky.sun_elongation(names, t)
        if elong < min_elong:
            continue
        event = {
            "utc": _iso(t),
            "bodies": list(names),
            "separation_deg": round(spread, 4),
            "sun_elongation_deg": round(float(elong), 2),
        }
        if sky.topocentric:
            vis = {
                "closest": sky.snapshot(names, t),
                "closest_visible": sky.closest_visible(names, t, min_alt, max_sun_alt),
            }
            for label, finder, hours in (
                ("after_sunset", almanac.find_settings, 1),
                ("before_sunrise", almanac.find_risings, -1),
            ):
                ts_ = sky.sun_event_offset(t, finder, hours)
                vis[label] = sky.snapshot(names, ts_) if ts_ is not None else None
            if vis["closest_visible"]:
                best = "closest_visible"
            else:
                options = [k for k in ("after_sunset", "before_sunrise") if vis[k] and vis[k]["all_above_horizon"]]
                best = min(options, key=lambda k: vis[k]["separation_deg"]) if options else None
            vis["best"] = best
            event["visibility"] = vis
        events.append(event)
    events.sort(key=lambda ev: ev["utc"])
    return events


def _observer(latitude, longitude, elevation_m) -> dict | None:
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be given together")
    if latitude is None:
        return None
    return {"latitude": float(latitude), "longitude": float(longitude), "elevation_m": float(elevation_m),
            "timezone": timezone_for(latitude, longitude)}


def _cached(params: dict, key_extra: dict, field: str, compute, use_cache: bool) -> str:
    """Return cached JSON for ``params``, else ``{**params, field: compute()}`` written to the cache."""
    key_data = {**params, **key_extra, "version": CACHE_VERSION}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:32]
    cache_file = cache_dir() / f"{key}.json"

    if use_cache and cache_file.is_file():
        return cache_file.read_text()

    text = json.dumps({**params, field: compute()}, indent=2)
    fd, tmp = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, cache_file)
    return text


def find_conjunctions(
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    max_separation_deg: float = 5.0,
    min_sun_elongation_deg: float = 0.0,
    min_bodies: int = 2,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float = 0.0,
    min_altitude_deg: float = 0.0,
    max_sun_altitude_deg: float = -6.0,
    planets_only: bool = False,
    use_cache: bool = True,
) -> str:
    """Return JSON listing conjunctions between ``start`` and ``end``.

    ``start`` defaults to today (UTC) and ``end`` to one year after ``start``.
    Objects: the Moon and Mercury-Saturn, plus (unless ``planets_only``) the
    Pleiades, Hyades, Beehive and M35 clusters and the bright stars Aldebaran,
    Pollux, Castor, Regulus, Spica and Antares.
    Each event gives the UTC time of closest apparent separation and the bodies
    involved (at least ``min_bodies``, all within ``max_separation_deg`` of each
    other). Dates may be ISO strings, dates or datetimes (naive = UTC).

    With ``latitude``/``longitude`` the search is topocentric and each event gets
    the observer's time zone (``observer.timezone``) and a ``visibility`` block with alt/az snapshots at closest approach, at the
    closest moment when all bodies are above ``min_altitude_deg`` with the Sun
    below ``max_sun_altitude_deg`` (within +/- 1 day), and one hour after sunset
    and before sunrise; ``best`` names the recommended snapshot.

    Results are cached on disk keyed by the arguments.
    """
    if min_bodies < 2:
        raise ValueError("min_bodies must be at least 2")
    observer = _observer(latitude, longitude, elevation_m)
    start_dt, end_dt = _default_range(start, end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    params = {
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_separation_deg": float(max_separation_deg),
        "min_sun_elongation_deg": float(min_sun_elongation_deg),
        "min_bodies": int(min_bodies),
        "observer": observer,
        "min_altitude_deg": float(min_altitude_deg),
        "max_sun_altitude_deg": float(max_sun_altitude_deg),
        "planets_only": bool(planets_only),
        "ephemeris": EPHEMERIS,
    }
    site = (latitude, longitude, elevation_m)
    compute = lambda: _compute(start_dt, end_dt, max_separation_deg, min_sun_elongation_deg,
                               min_bodies, site, min_altitude_deg, max_sun_altitude_deg, planets_only)
    return _cached(params, {"bodies": MOVING_BODIES, "fixed": FIXED_OBJECTS}, "conjunctions", compute, use_cache)


def find_oppositions(
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    elevation_m: float = 0.0,
    use_cache: bool = True,
) -> str:
    """Return JSON listing oppositions of Mars, Jupiter and Saturn between ``start`` and ``end``.

    ``start`` defaults to today (UTC) and ``end`` to one year after ``start``.
    Each event gives the UTC time of opposition (geocentric ecliptic longitude
    180 degrees from the Sun), the planet's distance (AU), visual magnitude and
    elongation from the Sun, and its closest approach to Earth (which for Mars
    can be days from opposition).

    With ``latitude``/``longitude`` each event gets a ``visibility`` block with an
    alt/az snapshot at the meridian transit nearest opposition.

    Results are cached on disk keyed by the arguments.
    """
    observer = _observer(latitude, longitude, elevation_m)
    start_dt, end_dt = _default_range(start, end)
    if end_dt <= start_dt:
        raise ValueError("end must be after start")

    params = {
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "observer": observer,
        "ephemeris": EPHEMERIS,
    }

    def compute():
        sky = _Sky(latitude, longitude, elevation_m)
        events = sky.oppositions(sky.ts.from_datetime(start_dt), sky.ts.from_datetime(end_dt))
        return sorted(events, key=lambda ev: ev["utc"])

    return _cached(params, {"kind": "oppositions", "bodies": OPPOSITION_BODIES}, "oppositions", compute, use_cache)
