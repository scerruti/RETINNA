"""Look up a California fire and its official perimeter.

FRAP (CAL FIRE) has every year through the last full calendar year. Fires from the
current year come from NIFC WFIGS until FRAP's annual release.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from . import config

_TIMEOUT = 30


@dataclass
class Fire:
    id: str                 # "<source>:<objectid>"
    name: str
    year: int
    acres: float            # officially reported GIS acres
    start: date | None
    end: date | None        # containment; may be None for active fires
    source: str             # "FRAP" or "WFIGS"
    geometry: BaseGeometry | None = None  # EPSG:4326

    def summary(self) -> dict:
        return {
            "id": self.id, "name": self.name, "year": self.year, "acres": round(self.acres or 0, 1),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None, "source": self.source,
        }


def _ms_to_date(ms) -> date | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() if ms else None


def _sql_str(s: str) -> str:
    return s.upper().replace("'", "''")


def _query(url: str, where: str, out_fields: str, geometry: bool) -> list[dict]:
    params = {
        "where": where, "outFields": out_fields, "returnGeometry": str(geometry).lower(),
        "outSR": 4326, "f": "geojson",
    }
    r = requests.get(url, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"ArcGIS error from {url}: {body['error']}")
    return body.get("features", [])


def _from_frap(feat: dict) -> Fire:
    a = feat["properties"]
    return Fire(
        id=f"FRAP:{a['OBJECTID']}", name=(a["FIRE_NAME"] or "").title(), year=a["YEAR_"],
        acres=a["GIS_ACRES"] or 0.0, start=_ms_to_date(a["ALARM_DATE"]),
        end=_ms_to_date(a["CONT_DATE"]), source="FRAP",
        geometry=shape(feat["geometry"]) if feat.get("geometry") else None,
    )


def _from_wfigs(feat: dict) -> Fire:
    a = feat["properties"]
    start = _ms_to_date(a["attr_FireDiscoveryDateTime"])
    return Fire(
        id=f"WFIGS:{a['OBJECTID']}", name=(a["poly_IncidentName"] or "").title(),
        year=start.year if start else date.today().year, acres=a["poly_GISAcres"] or 0.0,
        start=start, end=_ms_to_date(a["attr_ContainmentDateTime"]), source="WFIGS",
        geometry=shape(feat["geometry"]) if feat.get("geometry") else None,
    )


_FRAP_FIELDS = "OBJECTID,YEAR_,FIRE_NAME,GIS_ACRES,ALARM_DATE,CONT_DATE"
_WFIGS_FIELDS = "OBJECTID,poly_IncidentName,poly_GISAcres,attr_FireDiscoveryDateTime,attr_ContainmentDateTime"


def search(name: str, year: int | None = None, min_acres: float = 1000, limit: int = 20) -> list[Fire]:
    """Find fires by name (substring, case-insensitive), largest first. No geometry."""
    fires: list[Fire] = []
    w = f"UPPER(FIRE_NAME) LIKE '%{_sql_str(name)}%' AND GIS_ACRES >= {float(min_acres)}"
    if year:
        w += f" AND YEAR_ = {int(year)}"
    fires += [_from_frap(f) for f in _query(config.FRAP_URL, w, _FRAP_FIELDS, False)]
    if year is None or year >= date.today().year:
        w = (f"UPPER(poly_IncidentName) LIKE '%{_sql_str(name)}%' AND attr_POOState = 'US-CA'"
             f" AND poly_GISAcres >= {float(min_acres)}")
        fires += [_from_wfigs(f) for f in _query(config.WFIGS_URL, w, _WFIGS_FIELDS, False)]
    fires.sort(key=lambda f: (-f.year, -f.acres))
    return fires[:limit]


def get(fire_id: str) -> Fire:
    """Fetch one fire with its perimeter geometry (EPSG:4326)."""
    source, _, oid = fire_id.partition(":")
    if source == "FRAP":
        feats, conv = _query(config.FRAP_URL, f"OBJECTID = {int(oid)}", _FRAP_FIELDS, True), _from_frap
    elif source == "WFIGS":
        feats, conv = _query(config.WFIGS_URL, f"OBJECTID = {int(oid)}", _WFIGS_FIELDS, True), _from_wfigs
    else:
        raise ValueError(f"Unknown fire id {fire_id!r}")
    if not feats:
        raise LookupError(f"No fire {fire_id}")
    fire = conv(feats[0])
    if fire.geometry is not None:
        fire.geometry = unary_union([fire.geometry]).buffer(0)  # repair invalid rings
    return fire
