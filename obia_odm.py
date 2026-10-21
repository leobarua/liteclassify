"""Drone photos -> orthomosaic + DSM through a WebODM install on this PC (no Qt).

Lite Classify does not bundle any photogrammetry: it drives the user's own WebODM over
its REST API, with one locked set of processing options, and downloads only the
orthophoto and the elevation DSM. Standard library only (urllib), so the installers
gain no dependency.

The flow (WebODMClient):
    login -> check a node is online -> create a partial task with the options ->
    upload the photos in batches -> commit -> poll status + console output ->
    download orthophoto.tif and dsm.tif -> check them -> remove the task.

The options are sent explicitly and read back before committing. resize_to is never
sent: WebODM queues a resize whenever that field is present.
"""
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_PORTS = (29800, 8000)          # native Windows WebODM, Docker WebODM
PROJECT_NAME = "Lite Classify"
IMAGE_EXTS = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng")

# Task status codes (WebODM / NodeODM).
QUEUED, RUNNING, FAILED, COMPLETED, CANCELED = 10, 20, 30, 40, 50
STATUS_NAMES = {None: "Waiting", QUEUED: "Queued", RUNNING: "Running", FAILED: "Failed",
                COMPLETED: "Completed", CANCELED: "Canceled"}

# ODM pipeline stages, as named in "Running <stage> stage" console lines.
STAGE_NAMES = {
    "dataset": "Loading photos", "split": "Splitting", "merge": "Merging",
    "opensfm": "Aligning photos", "openmvs": "Building dense cloud",
    "odm_filterpoints": "Filtering points", "odm_meshing": "Meshing",
    "mvs_texturing": "Texturing", "odm_georeferencing": "Georeferencing",
    "odm_dem": "Building DSM", "odm_orthophoto": "Building orthomosaic",
    "odm_report": "Writing report", "odm_postprocess": "Finishing",
}
_STAGE_RE = re.compile(r"Running (\w+) stage")

# Benchmarked against Metashape on the 217-image campus sample (experiments/webodm):
# full-resolution photos, dense cloud from half-resolution depth maps (Metashape "High"),
# a dense 2.5D mesh for crisp ortho edges, DSM at ~4x GSD, no DTM, no textured model.
# "Maximum" aligns at full resolution; it took 6 h longer there with the same geometry.
QUALITY_PRESETS = {
    "Standard": {"feature-quality": "high"},
    "Maximum": {"feature-quality": "ultra"},
}
BASE_OPTIONS = {
    "min-num-features": 40000,
    "pc-quality": "high",
    "mesh-size": 1000000,
    "mesh-octree-depth": 12,
    "dsm": True,
    "dtm": False,
    "auto-boundary": True,
    "skip-3dmodel": True,
    "fast-orthophoto": False,
}


class WebODMError(Exception):
    pass


class AuthError(WebODMError):
    pass


def build_options(quality, gsd_cm, boundary=None):
    """ The option list for a task: the locked base, the quality preset, the resolutions
    from the ground sampling distance (ortho at GSD, DSM at 4x), and an optional area of
    interest (GeoJSON string) that the outputs are cropped to. """
    opts = dict(BASE_OPTIONS)
    opts.update(QUALITY_PRESETS[quality])
    if gsd_cm:
        opts["orthophoto-resolution"] = round(float(gsd_cm), 2)
        opts["dem-resolution"] = round(4 * float(gsd_cm), 1)
    if boundary:
        opts["boundary"] = boundary
        opts["auto-boundary"] = False
    return [{"name": k, "value": v} for k, v in opts.items()]


# --- Photo folder summary (EXIF) ---

def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(folder, f)))


def _relative_altitude(path):
    """ DJI and most drones write the height above take-off in XMP. """
    with open(path, "rb") as f:
        head = f.read(256 * 1024)
    m = re.search(rb'RelativeAltitude="?\s*([+-]?[0-9.]+)', head)
    return float(m.group(1)) if m else None


def photo_summary(folder, sample=25):
    """ Count, size on disk, camera and an estimated GSD (cm/px) for a photo folder.
    GSD uses the sensor width implied by the 35 mm-equivalent focal length and the median
    height above take-off, so it is approximate (terrain, take-off point); the dialog
    lets the user change it. Reads EXIF from up to `sample` photos. """
    paths = list_images(folder)
    out = {"count": len(paths), "bytes": sum(os.path.getsize(p) for p in paths),
           "camera": None, "width": None, "height": None, "altitude_m": None, "gsd_cm": None}
    if not paths:
        return out
    try:
        from PIL import Image
    except ImportError:
        return out
    step = max(1, len(paths) // sample)
    alts, focal, f35 = [], None, None
    for p in paths[::step]:
        try:
            with Image.open(p) as im:
                out["width"], out["height"] = im.size
                ex = im.getexif()
                sub = ex.get_ifd(0x8769)
                out["camera"] = " ".join(str(ex.get(t, "")).strip("\x00 ") for t in (0x010F, 0x0110)).strip() or None
                if sub.get(0x920A):
                    focal = float(sub[0x920A])
                if sub.get(0xA405):
                    f35 = float(sub[0xA405])
            a = _relative_altitude(p)
            if a is not None and a > 1:
                alts.append(a)
        except Exception:
            continue
    if alts:
        out["altitude_m"] = sorted(alts)[len(alts) // 2]
    w, h = out["width"], out["height"]
    if focal and f35 and w and h and out["altitude_m"]:
        diag_mm = 43.2666 * focal / f35                 # 35 mm film diagonal scaled
        sensor_w = diag_mm * w / math.hypot(w, h)
        out["gsd_cm"] = 100 * out["altitude_m"] * sensor_w / (focal * w)
    return out


# --- Photo positions and subset selection ---

def _dms(v):
    d, m, s = (float(x) for x in v)
    return d + m / 60 + s / 3600


def photo_positions(paths):
    """ [{path, lon, lat, time}] for the photos with an EXIF GPS position, in capture order
    (DateTimeOriginal, then file name). """
    from PIL import Image
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                ex = im.getexif()
                gps = ex.get_ifd(0x8825)
                t = ex.get_ifd(0x8769).get(0x9003, "")
            if not (gps.get(2) and gps.get(4)):
                continue
            lat, lon = _dms(gps[2]), _dms(gps[4])
            if gps.get(1, "N") == "S":
                lat = -lat
            if gps.get(3, "E") == "W":
                lon = -lon
            out.append({"path": p, "lon": lon, "lat": lat, "time": str(t)})
        except Exception:
            continue
    out.sort(key=lambda r: (r["time"], os.path.basename(r["path"])))
    return out


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) % 60 + 1


def project(lons, lats, epsg, inverse=False):
    """ WGS84 lon/lat <-> the given projected CRS (pyproj). """
    from pyproj import Transformer
    t = Transformer.from_crs(epsg if inverse else 4326, 4326 if inverse else epsg, always_xy=True)
    return t.transform(lons, lats)


def select_photos(xy, aoi_xy, footprint_m):
    """ Mask of photos to process for an area of interest (polygon vertices, same CRS as
    xy): those within half a photo footprint of it, so the area keeps full overlap out to
    its edges. footprint_m: (width, height) of one photo on the ground. """
    from shapely import contains_xy
    from shapely.geometry import Polygon
    if len(aoi_xy) < 3:
        return [True] * len(xy)
    zone = Polygon(aoi_xy).buffer(max(footprint_m) / 2)
    xs = [p[0] for p in xy]
    ys = [p[1] for p in xy]
    return [bool(v) for v in contains_xy(zone, xs, ys)]


def boundary_geojson(aoi_xy, epsg):
    """ The area of interest as a WGS84 GeoJSON string for ODM's boundary option. """
    lons, lats = project([p[0] for p in aoi_xy], [p[1] for p in aoi_xy], epsg, inverse=True)
    ring = [[round(x, 8), round(y, 8)] for x, y in zip(lons, lats)]
    ring.append(ring[0])
    return json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]}}]})


# --- Remember-me password (Windows DPAPI, current user only) ---

def protect(text):
    """ Encrypts text for the current Windows user (DPAPI). Returns base64, or None off Windows. """
    if os.name != "nt":
        return None
    import base64
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    data = text.encode("utf-8")
    src = BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    dst = BLOB()
    if not ctypes.windll.crypt32.CryptProtectData(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        return None
    try:
        return base64.b64encode(ctypes.string_at(dst.pbData, dst.cbData)).decode("ascii")
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


def unprotect(blob):
    if os.name != "nt" or not blob:
        return None
    import base64
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]
    data = base64.b64decode(blob)
    src = BLOB(len(data), ctypes.create_string_buffer(data, len(data)))
    dst = BLOB()
    if not ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(dst)):
        return None
    try:
        return ctypes.string_at(dst.pbData, dst.cbData).decode("utf-8")
    finally:
        ctypes.windll.kernel32.LocalFree(dst.pbData)


# --- REST client ---

def find_local_webodm(ports=DEFAULT_PORTS, timeout=1.5):
    """ The first http://127.0.0.1:<port> that answers like WebODM, or None. """
    for port in ports:
        url = f"http://127.0.0.1:{port}"
        try:
            urllib.request.urlopen(url + "/api/", timeout=timeout)
            return url
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return url
        except Exception:
            pass
    return None


def _multipart(fields, files):
    """ multipart/form-data body. files: list of (field, filename, bytes). """
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    for field, name, data in files:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{name}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode())
        parts.append(data)
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class WebODMClient:
    """ Talks to one WebODM instance as one user. Logs in again by itself when the token
    expires (WebODM tokens last 6 h; a full run can take longer). """

    def __init__(self, url, username, password, timeout=60):
        self.url = url.rstrip("/")
        self.username, self.password = username, password
        self.timeout = timeout
        self.token = None

    def login(self):
        body = json.dumps({"username": self.username, "password": self.password}).encode()
        req = urllib.request.Request(self.url + "/api/token-auth/", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.token = json.load(r).get("token")
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise AuthError("WebODM did not accept this username and password.")
            raise WebODMError(f"WebODM login failed (HTTP {e.code}).")
        except urllib.error.URLError as e:
            raise WebODMError(f"Cannot reach WebODM at {self.url} ({e.reason}). Is it running?")
        if not self.token:
            raise AuthError("WebODM returned no login token.")

    def _request(self, method, path, data=None, content_type=None, raw=False, retry=True):
        if self.token is None:
            self.login()
        headers = {"Authorization": f"JWT {self.token}"}
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            r = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and retry:
                self.token = None
                return self._request(method, path, data, content_type, raw, retry=False)
            detail = e.read()[:500].decode("utf-8", "replace")
            raise WebODMError(f"WebODM {method} {path} failed (HTTP {e.code}): {detail}")
        except urllib.error.URLError as e:
            raise WebODMError(f"Cannot reach WebODM at {self.url} ({e.reason}). Is it running?")
        if raw:
            return r
        with r:
            body = r.read()
        return json.loads(body) if body else None

    def _get(self, path):
        return self._request("GET", path)

    def _post_json(self, path, obj=None):
        return self._request("POST", path, json.dumps(obj or {}).encode(), "application/json")

    # --- nodes and projects ---
    def online_nodes(self):
        nodes = self._get("/api/processingnodes/") or []
        return [n for n in nodes if n.get("online")]

    def ensure_project(self, name=PROJECT_NAME):
        page = "/api/projects/?ordering=-created_at"
        while page:
            res = self._get(page)
            items = res.get("results", []) if isinstance(res, dict) else (res or [])
            for p in items:
                if p.get("name") == name:
                    return p["id"]
            nxt = res.get("next") if isinstance(res, dict) else None
            page = nxt[len(self.url):] if nxt and nxt.startswith(self.url) else None
        return self._post_json("/api/projects/", {"name": name, "description": "Created by Lite Classify"})["id"]

    # --- tasks ---
    def _task_path(self, project, task, action=""):
        return f"/api/projects/{project}/tasks/{task}/" + (f"{action}/" if action else "")

    def create_task(self, project, name, options):
        """ A partial task (photos uploaded afterwards). No resize_to: see the module notes. """
        body, ctype = _multipart({"name": name, "partial": "true", "auto_processing_node": "true",
                                  "options": json.dumps(options)}, [])
        return self._request("POST", f"/api/projects/{project}/tasks/", body, ctype)

    def upload(self, project, task, paths, batch_bytes=200 * 1024 * 1024, progress=None, cancelled=None):
        """ Uploads photos in batches of about batch_bytes; progress(done, total) per batch. """
        total = len(paths)
        done, files, size = 0, [], 0

        def send():
            body, ctype = _multipart({}, files)
            for attempt in range(3):
                try:
                    return self._request("POST", self._task_path(project, task, "upload"), body, ctype)
                except WebODMError:
                    if attempt == 2:
                        raise
                    time.sleep(3)
        for p in paths:
            if cancelled and cancelled():
                return False
            with open(p, "rb") as f:
                data = f.read()
            files.append(("images", os.path.basename(p), data))
            size += len(data)
            if size >= batch_bytes:
                send()
                done += len(files)
                files, size = [], 0
                if progress:
                    progress(done, total)
        if files:
            send()
            done += len(files)
            if progress:
                progress(done, total)
        return True

    def task(self, project, task):
        return self._get(self._task_path(project, task))

    def commit(self, project, task):
        return self._post_json(self._task_path(project, task, "commit"))

    def cancel(self, project, task):
        return self._post_json(self._task_path(project, task, "cancel"))

    def remove(self, project, task):
        return self._post_json(self._task_path(project, task, "remove"))

    def output(self, project, task, line=0):
        """ Console lines from `line` on, and the total line count. """
        res = self._get(self._task_path(project, task, "output") + f"?line={int(line)}&f=json")
        return res.get("lines", []), res.get("count", line)

    def download(self, project, task, asset, dest, progress=None, cancelled=None):
        """ Streams one asset (orthophoto.tif, dsm.tif) to dest via a .part file. """
        r = self._request("GET", f"/api/projects/{project}/tasks/{task}/download/{asset}", raw=True)
        total = int(r.headers.get("Content-Length") or 0)
        tmp = dest + ".part"
        got = 0
        with r, open(tmp, "wb") as f:
            while True:
                if cancelled and cancelled():
                    f.close()
                    os.remove(tmp)
                    return False
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
        os.replace(tmp, dest)
        return True


def options_match(sent, task_options):
    """ Options WebODM stored differ from what was sent -> list of 'name: sent -> stored'. """
    def norm(v):
        if isinstance(v, str) and v[:1] in "{[":
            try:
                return json.loads(v)               # GeoJSON may come back parsed
            except ValueError:
                pass
        return str(v).lower() if not isinstance(v, (dict, list)) else v
    stored = {o["name"]: o["value"] for o in (task_options or [])}
    bad = []
    for o in sent:
        v = stored.get(o["name"], "<missing>")
        if norm(v) != norm(o["value"]):
            bad.append(f"{o['name']}: {str(o['value'])[:60]} -> {str(v)[:60]}")
    return bad


def stage_from_lines(lines, current=None):
    """ The latest ODM stage named in console lines (friendly name), else current. """
    for line in lines:
        m = _STAGE_RE.search(line)
        if m:
            current = STAGE_NAMES.get(m.group(1), m.group(1))
    return current


def check_outputs(ortho_path, dsm_path, gsd_cm=None, console_lines=()):
    """ Problems with the downloaded files, as a list of messages (empty = fine): the DSM
    must be one float band (not the coloured preview), and the ortho should be at the
    requested resolution (ODM caps it when the photos were resized). """
    import rasterio
    problems = []
    with rasterio.open(dsm_path) as d:
        if d.count != 1 or not d.dtypes[0].startswith("float"):
            problems.append(f"The DSM has {d.count} band(s) of {d.dtypes[0]}; expected 1 float band of elevations.")
    with rasterio.open(ortho_path) as o:
        px_cm = abs(o.transform.a) * 100 if not (o.crs and o.crs.is_geographic) else None
        if o.count < 3:
            problems.append(f"The orthomosaic has {o.count} band(s); expected RGB.")
    if gsd_cm and px_cm and px_cm > 1.15 * gsd_cm:
        problems.append(f"The orthomosaic is {px_cm:.2f} cm/px, coarser than the requested {gsd_cm:.2f}.")
    for line in console_lines:
        if "Maximum resolution set to" in line:
            problems.append("ODM capped the resolution: " + line.strip())
            break
    return problems


def free_disk_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None
