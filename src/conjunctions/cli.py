import argparse
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from zoneinfo import ZoneInfo

from .core import cache_dir, data_dir, find_conjunctions, find_oppositions, timezone_for

COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()

SNAPSHOT_LABELS = {
    "closest": "closest approach",
    "closest_visible": "closest while visible",
    "after_sunset": "1h after sunset",
    "before_sunrise": "1h before sunrise",
}


def _time(utc: str, tz) -> str:
    # tz=None converts to the system's local zone (DST-aware per date).
    dt = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M %Z")


def _snapshot_lines(label: str, snap: dict, tz, indent: str) -> list[str]:
    width = max(len(n) for n in snap["positions"])
    sep = f"  sep {snap['separation_deg']:.2f}°" if "separation_deg" in snap else ""
    lines = [f"{indent}{label}: {_time(snap['utc'], tz)}{sep}  sun alt {snap['sun_alt_deg']:+.0f}°"]
    for name, p in snap["positions"].items():
        direction = COMPASS[round(p["az_deg"] / 22.5) % 16]
        lines.append(
            f"{indent}    {name:<{width}}  alt {p['alt_deg']:+6.1f}°  az {p['az_deg']:5.1f}° {direction}"
        )
    return lines


def _observer_line(obs: dict | None) -> str:
    if not obs:
        return "Geocentric positions (give --lat/--lon for alt/az and viewing times)"
    lat, lon = obs["latitude"], obs["longitude"]
    return (f"Observer {abs(lat):.3f}°{'N' if lat >= 0 else 'S'} {abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"
            f" {obs['elevation_m']:.0f} m ({obs['timezone']})")


def format_oppositions(data: dict, tz) -> str:
    lines = [_observer_line(data["observer"]),
             f"{data['start']} to {data['end']}, {len(data['oppositions'])} opposition(s)"]
    for ev in data["oppositions"]:
        closest = ev["closest_approach"]
        lines.append("")
        lines.append(f"{_time(ev['utc'], tz)}  {ev['body']} at opposition  mag {ev['magnitude']:+.1f}"
                     f"  {ev['distance_au']:.3f} AU")
        lines.append(f"  closest to Earth: {_time(closest['utc'], tz)}  {closest['distance_au']:.3f} AU")
        vis = ev.get("visibility")
        if vis:
            lines += _snapshot_lines("highest (meridian transit)", vis["transit"], tz, "  ")
    return "\n".join(lines)


def format_text(data: dict, tz, verbose: bool = False) -> str:
    lines = [_observer_line(data["observer"])]
    lines.append(f"{data['start']} to {data['end']}, max separation {data['max_separation_deg']}°, "
                 f"{len(data['conjunctions'])} conjunction(s)")

    for ev in data["conjunctions"]:
        lines.append("")
        lines.append(f"{_time(ev['utc'], tz)}  {' · '.join(ev['bodies'])}"
                     f"  sep {ev['separation_deg']:.2f}°  elongation {ev['sun_elongation_deg']:.0f}°")
        vis = ev.get("visibility")
        if not vis:
            continue
        best = vis["best"]
        if best:
            lines += _snapshot_lines(f"best ({SNAPSHOT_LABELS[best]})", vis[best], tz, "  ")
        else:
            lines.append("  not visible from this location")
        if verbose:
            for key, label in SNAPSHOT_LABELS.items():
                if key != best and vis[key]:
                    lines += _snapshot_lines(label, vis[key], tz, "  ")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conjunctions",
        description="Find conjunctions of the Moon, naked-eye planets, bright stars and star clusters, "
                    "or oppositions of Mars, Jupiter and Saturn.",
    )
    parser.add_argument("start", nargs="?", help="start date/time, ISO 8601 (UTC if no offset); default today")
    parser.add_argument("end", nargs="?", help="end date/time, ISO 8601; default one year after start")
    parser.add_argument("-s", "--max-sep", type=float, default=5.0,
                        help="maximum separation in degrees (default: 5)")
    parser.add_argument("-e", "--min-elongation", type=float, default=0.0,
                        help="drop events closer than this many degrees to the Sun (default: 0)")
    parser.add_argument("-n", "--min-bodies", type=int, default=2,
                        help="minimum number of bodies in a conjunction (default: 2)")
    parser.add_argument("-p", "--planets-only", action="store_true",
                        help="only the Moon and naked-eye planets (skip stars and clusters)")
    parser.add_argument("-o", "--oppositions", action="store_true",
                        help="list oppositions of Mars, Jupiter and Saturn instead of conjunctions")
    parser.add_argument("--lat", type=float, help="observer latitude in degrees (north positive)")
    parser.add_argument("--lon", type=float, help="observer longitude in degrees (east positive)")
    parser.add_argument("--elevation", type=float, default=0.0, help="observer elevation in metres")
    parser.add_argument("--min-alt", type=float, default=0.0,
                        help="minimum altitude for bodies to count as visible (default: 0)")
    parser.add_argument("--max-sun-alt", type=float, default=-6.0,
                        help="maximum Sun altitude for 'closest while visible' (default: -6, civil twilight)")
    parser.add_argument("--tz", help="IANA time zone for displayed times "
                                     "(default: looked up from --lat/--lon, else the system zone)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show all viewing snapshots")
    parser.add_argument("--json", action="store_true", help="output JSON instead of text")
    parser.add_argument("--compact", action="store_true", help="single-line JSON output (implies --json)")
    parser.add_argument("--no-cache", action="store_true", help="recompute even if cached")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('conjunctions')}")
    parser.add_argument("--paths", action="store_true", help="print data and cache directories and exit")
    args = parser.parse_args(argv)

    if args.paths:
        print(json.dumps({"data_dir": str(data_dir()), "cache_dir": str(cache_dir())}, indent=2))
        return 0
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")
    tz_name = args.tz or (timezone_for(args.lat, args.lon) if args.lat is not None else None)
    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except Exception:
        parser.error(f"unknown time zone: {tz_name}")

    try:
        if args.oppositions:
            text = find_oppositions(args.start, args.end, args.lat, args.lon, args.elevation,
                                    use_cache=not args.no_cache)
        else:
            text = find_conjunctions(
                args.start, args.end, args.max_sep, args.min_elongation, args.min_bodies,
                args.lat, args.lon, args.elevation, args.min_alt, args.max_sun_alt,
                planets_only=args.planets_only,
                use_cache=not args.no_cache,
            )
    except ValueError as exc:
        parser.error(str(exc))

    if args.compact:
        text = json.dumps(json.loads(text), separators=(",", ":"))
    elif args.oppositions and not args.json:
        text = format_oppositions(json.loads(text), tz)
    elif not args.json:
        text = format_text(json.loads(text), tz, args.verbose)
    sys.stdout.write(text + "\n")
    return 0
