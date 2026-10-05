# conjunctions

Finds conjunctions (closest apparent angular separation, geocentric or from an observer location) between the Moon,
the naked-eye planets (Mercury, Venus, Mars, Jupiter, Saturn), bright star clusters
(Pleiades, Hyades, Beehive, M35) and bright stars near the ecliptic (Aldebaran, Pollux,
Castor, Regulus, Spica, Antares).

Star clusters take precedence over bright stars: when a body's conjunction with a star
coincides (within a day) with its conjunction with a cluster lying within the separation
limit of that star — e.g. Aldebaran in front of the Hyades — only the cluster is reported.

## Install

Requires Python 3.11+.

```sh
pip install conjunctions
```

or, from a checkout, with [uv](https://docs.astral.sh/uv/): `uv sync`.

## Library

```python
from conjunctions import find_conjunctions

print(find_conjunctions("2026-10-01", "2026-12-31", max_separation_deg=3))
print(find_conjunctions(planets_only=True))   # today -> one year from today, Moon/planets only
```

`start` defaults to today (UTC) and `end` to one year after `start`.

Returns a JSON string; each entry has `utc` (closest approach), `bodies`,
`separation_deg` (largest pairwise separation in the group) and `sun_elongation_deg`.
`min_bodies` (default 2) keeps only groups of at least that many objects; overlapping
pair conjunctions are merged into the largest group whose members all lie within
`max_separation_deg` of each other.

Pass `latitude`/`longitude` (and optionally `elevation_m`) for topocentric positions.
The output's `observer` includes the location's time zone (offline lookup via
`timezonefinder`), and each entry gets a `visibility` block with alt/az snapshots:

- `closest` – the moment of closest approach
- `closest_visible` – closest moment within ±1 day with every body above `min_altitude_deg`
  (default 0) and the Sun below `max_sun_altitude_deg` (default −6°, civil twilight)
- `after_sunset` / `before_sunrise` – 1 h after the nearest sunset / before the nearest sunrise
- `best` – which of the above to use (`closest_visible`, else whichever twilight snapshot
  has everything above the horizon; `null` if not visible)

## CLI

```
conjunctions 2026-01-01 2027-01-01 -s 3 -e 15   # within 3°, at least 15° from the Sun
conjunctions --lat 35.78 --lon -78.64            # next 12 months, local times for that location
conjunctions -p --lat 35.78 --lon -78.64         # Moon and planets only (no stars/clusters)
conjunctions 2026-10-01 2026-11-01 --lat 35.78 --lon -78.64 --tz UTC
conjunctions 2026-01-01 2027-01-01 -n 3 --lat 35.78 --lon -78.64 --min-alt 10 -v
conjunctions --paths                            # show data and cache directories
```

Times display in `--tz`, else the observer's time zone, else the system zone.
Prints a human-readable list (best viewing time with each body's alt/az and compass
direction); `-v` adds all snapshots, `--json` / `--compact` print the raw JSON.

## Files

- Ephemeris (`de440s.bsp`, downloaded on first use if missing): first of `$CONJUNCTIONS_DATA_DIR`,
  `/var/data`, the site cache dir, or the user cache dir that already has it (else first writable).
- Results cache: `$CONJUNCTIONS_CACHE_DIR`, else the site or user cache dir (via `platformdirs`).
  Use `--no-cache` (or `use_cache=False`) to recompute.

## Tests

```
uv run pytest
```

Tests use a temporary results cache and need `de440s.bsp` (downloaded on first run if missing).
