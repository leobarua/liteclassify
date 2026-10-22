"""Checks obia_odm.WebODMClient against a small fake WebODM (no network, no WebODM install).

    python experiments/webodm/test_odm_client.py

The fake implements the endpoints the client uses, with WebODM's behaviour where it
matters: JWT login, token expiry (401) mid-run, paginated projects, partial tasks,
batched uploads, commit, console output by line, and asset downloads.
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.getcwd())
import numpy as np
import rasterio
from rasterio.transform import from_origin

import obia_odm as odm

STATE = {"token": "t1", "tasks": {}, "uploads": 0, "logins": 0, "expire_after": 3, "calls": 0}
CONSOLE = ["[INFO]    Running dataset stage", "[INFO]    Running opensfm stage",
           "[INFO]    Running openmvs stage", "[INFO]    Running odm_dem stage"]


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        STATE["calls"] += 1
        if STATE["calls"] % STATE["expire_after"] == 0:          # token "expires" now and then
            STATE["token"] = f"t{STATE['calls']}"
        if self.headers.get("Authorization") != f"JWT {STATE['token']}":
            self._json(401, {"detail": "expired"})
            return False
        return True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n)

    def do_POST(self):
        p = self.path
        if p == "/api/token-auth/":
            d = json.loads(self._body())
            STATE["logins"] += 1
            if d.get("password") != "secret":
                return self._json(400, {"non_field_errors": ["bad"]})
            return self._json(200, {"token": STATE["token"]})
        body = self._body()
        if not self._authed():
            return
        if p == "/api/projects/":
            return self._json(201, {"id": 7, "name": "Lite Classify"})
        if p == "/api/projects/7/tasks/":
            text = body.decode("latin-1")
            assert "resize_to" not in text, "resize_to must never be sent"
            opts = json.loads(text.split('name="options"\r\n\r\n')[1].split("\r\n")[0])
            if not all(o.get("name") and o.get("value") for o in opts):   # WebODM's validate_task_options
                return self._json(400, {"options": ["Invalid options"]})
            STATE["tasks"]["abc"] = {"id": "abc", "status": None, "options": opts, "images": 0,
                                     "running_progress": 0.0, "polls": 0}
            return self._json(201, STATE["tasks"]["abc"])
        t = STATE["tasks"].get("abc")
        if p.endswith("/upload/"):
            n = body.count(b'name="images"')
            t["images"] += n
            STATE["uploads"] += 1
            return self._json(200, {"success": True})
        if p.endswith("/commit/"):
            t["status"] = odm.QUEUED
            return self._json(200, t)
        if p.endswith("/remove/"):
            t["pending_action"] = "remove"          # deleted later by WebODM's worker
            return self._json(200, {"success": True})
        if p.endswith("/cancel/"):
            t["status"] = odm.CANCELED
            return self._json(200, {"success": True})
        self._json(404, {})

    def do_GET(self):
        p = self.path
        if p == "/api/":
            return self._json(401, {})
        if not self._authed():
            return
        if p == "/api/processingnodes/":
            return self._json(200, [{"id": 1, "online": True}])
        if p.startswith("/api/projects/?"):
            return self._json(200, {"count": 1, "next": None, "results": [{"id": 3, "name": "Other"}]})
        t = STATE["tasks"].get("abc")
        if t and t.get("pending_action") == "remove":
            if t.setdefault("remove_polls", 0) >= 1:
                STATE["tasks"].pop("abc")
                t = None
            else:
                t["remove_polls"] += 1
        if p == "/api/projects/7/tasks/":
            return self._json(200, list(STATE["tasks"].values()))
        if t is None:
            return self._json(404, {"detail": "Not found."})
        if p == "/api/projects/7/tasks/abc/":
            t["polls"] += 1
            if t["status"] in (odm.QUEUED, odm.RUNNING):
                t["status"] = odm.RUNNING
                t["running_progress"] = min(1.0, 0.3 * t["polls"])
                if t["running_progress"] >= 1.0:
                    t["status"] = odm.COMPLETED
            return self._json(200, t)
        if p.startswith("/api/projects/7/tasks/abc/output/"):
            line = int(p.split("line=")[1].split("&")[0])
            upto = min(len(CONSOLE), t["polls"] + 1)
            return self._json(200, {"lines": CONSOLE[line:upto], "count": upto})
        if "/download/" in p:
            name = p.rsplit("/", 1)[1]
            data = open(ASSETS[name], "rb").read()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._json(404, {})


def make_assets(d):
    tr = from_origin(310000, 1566000, 0.0432, 0.0432)
    o, s = os.path.join(d, "o.tif"), os.path.join(d, "s.tif")
    with rasterio.open(o, "w", driver="GTiff", width=50, height=50, count=4, dtype="uint8",
                       crs="EPSG:32651", transform=tr) as f:
        f.write(np.full((4, 50, 50), 128, "uint8"))
    with rasterio.open(s, "w", driver="GTiff", width=13, height=13, count=1, dtype="float32",
                       crs="EPSG:32651", transform=from_origin(310000, 1566000, 0.17, 0.17)) as f:
        f.write(np.full((1, 13, 13), 50.0, "float32"))
    return {"orthophoto.tif": o, "dsm.tif": s}


if __name__ == "__main__":
    tmp = tempfile.mkdtemp()
    ASSETS = make_assets(tmp)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"

    assert odm.find_local_webodm(ports=(srv.server_port,)) == url
    bad = odm.WebODMClient(url, "leo", "wrong")
    try:
        bad.login()
        raise AssertionError("bad password accepted")
    except odm.AuthError:
        pass

    c = odm.WebODMClient(url, "leo", "secret")
    assert len(c.online_nodes()) == 1
    pid = c.ensure_project()
    assert pid == 7, pid
    opts = odm.build_options("Standard", 4.32, '{"type": "FeatureCollection", "features": []}')
    assert {"name": "auto-boundary", "value": "false"} in opts, opts
    t = c.create_task(pid, "test", opts)
    assert odm.options_match(opts, c.task(pid, t["id"])["options"]) == []
    photos = []
    for i in range(7):
        p = os.path.join(tmp, f"IMG_{i}.JPG")
        open(p, "wb").write(os.urandom(1000))
        photos.append(p)
    seen = []
    c.upload(pid, t["id"], photos, batch_bytes=2500, progress=lambda d, n: seen.append((d, n)))
    assert STATE["tasks"]["abc"]["images"] == 7 and seen[-1] == (7, 7) and STATE["uploads"] == 3, (STATE, seen)
    c.commit(pid, t["id"])
    line, stage = 0, None
    while True:
        info = c.task(pid, t["id"])
        lines, line = c.output(pid, t["id"], line)
        stage = odm.stage_from_lines(lines, stage)
        if info["status"] == odm.COMPLETED:
            break
    assert stage == "Building DSM", stage
    o, s = os.path.join(tmp, "out_o.tif"), os.path.join(tmp, "out_s.tif")
    assert c.download(pid, t["id"], "orthophoto.tif", o)
    assert c.download(pid, t["id"], "dsm.tif", s)
    assert odm.check_outputs(o, s, 4.32) == []
    assert odm.check_outputs(o, o, 4.32), "colour DSM not caught"
    assert odm.check_outputs(o, s, 4.32, ["Maximum resolution set to 7.39 cm/pixel"]), "cap not caught"
    assert [x["id"] for x in c.tasks(pid)] == ["abc"]
    assert c.remove_and_wait(pid, t["id"], timeout=5, poll=0.05)
    assert "abc" not in STATE["tasks"] and not c.exists(pid, t["id"]) and c.tasks(pid) == []
    assert STATE["logins"] > 2, "token expiry was never exercised"
    assert odm.options_match(opts, [{"name": "fast-orthophoto", "value": True}])
    print(f"OK ({STATE['logins']} logins incl. re-logins after expiry, {STATE['uploads']} upload batches)")
    srv.shutdown()
