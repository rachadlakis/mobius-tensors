"""IP cameras (e.g. PoE cameras on a switch wired to this machine): config, live capture, discovery.

Cameras are declared by name in the `cameras` section of a config file and used anywhere a
source is accepted (`--source gate`). Credentials stay out of the config: any `${VAR}` in a
camera entry is read from the environment or a `.env` file.

    uv run -m histor.engine.cameras list                 # configured cameras (passwords masked)
    uv run -m histor.engine.cameras discover             # ONVIF WS-Discovery on every local interface
    uv run -m histor.engine.cameras scan 192.168.1.0/24  # find hosts with RTSP (554) open
    uv run -m histor.engine.cameras test gate --save data/gate.jpg
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
import re
import select
import socket
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np

log = logging.getLogger(__name__)

STREAM_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")

# RTSP path per vendor for (channel, stream), stream being "main" or "sub".
# The sub stream (lower resolution) is usually all a detector needs and is far cheaper to decode.
VENDOR_PATHS: dict[str, Callable[[int, str], str]] = {
    "hikvision": lambda ch, s: f"/Streaming/Channels/{ch}0{1 if s == 'main' else 2}",
    "dahua": lambda ch, s: f"/cam/realmonitor?channel={ch}&subtype={0 if s == 'main' else 1}",
    "reolink": lambda ch, s: f"/h264Preview_{ch:02d}_{s}",
    "uniview": lambda ch, s: f"/unicast/c{ch}/s{0 if s == 'main' else 1}/live",
    "axis": lambda ch, s: f"/axis-media/media.amp?camera={ch}",
}

_ENV_VAR = re.compile(r"\$\{(\w+)\}")


def expand_env(value: str) -> str:
    """Replace `${VAR}` with the environment variable (loaded from .env if present)."""

    def lookup(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Environment variable {name} is not set (add it to .env)")
        return os.environ[name]

    return _ENV_VAR.sub(lookup, value)


def camera_url(camera: dict[str, Any]) -> str:
    """Build the stream URL for one entry of the `cameras` config section."""
    from dotenv import load_dotenv

    load_dotenv()
    cam = {k: expand_env(v) if isinstance(v, str) else v for k, v in camera.items()}
    if cam.get("url"):
        return cam["url"]
    if not cam.get("host"):
        raise ValueError("Camera needs either `url` or `host`")

    stream = cam.get("stream", "main")
    if stream not in ("main", "sub"):
        raise ValueError(f"Camera `stream` must be main or sub, got {stream!r}")
    vendor = str(cam.get("vendor", "generic")).lower()
    path = cam.get("path")
    if not path:
        if vendor not in VENDOR_PATHS:
            raise ValueError(
                f"Unknown vendor {vendor!r}: set `path` (or `url`), or use one of "
                f"{sorted(VENDOR_PATHS)}"
            )
        path = VENDOR_PATHS[vendor](int(cam.get("channel", 1)), stream)

    auth = ""
    if cam.get("username"):
        auth = quote(str(cam["username"]), safe="")
        if cam.get("password"):
            auth += ":" + quote(str(cam["password"]), safe="")
        auth += "@"
    port = int(cam.get("port", 554))
    return f"rtsp://{auth}{cam['host']}:{port}/{path.lstrip('/')}"


def main_stream_url(camera: dict[str, Any]) -> str:
    """Full-resolution stream of a camera, for recording and hi-res crops: `main_url` if set,
    else the vendor preset's main stream; a camera given only by `url` has just that stream."""
    if camera.get("main_url"):
        return camera_url({"url": camera["main_url"]})
    return camera_url({**camera, "stream": "main"})


def resolve_source(source: str | int, cameras: dict[str, Any] | None) -> str | int:
    """Map a camera name to its URL; any other source is returned unchanged."""
    if cameras and str(source) in cameras:
        return camera_url(cameras[str(source)])
    return source


FACE_VIEWS = ("good", "poor", "none")


def camera_about(cameras: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """What each camera looks at, from its `about:` entry (filled in at install time):
    description, aliases, location, covers, faces (good | poor | none), hours, notes, zones.
    The chat reads it to pick the right camera and to explain its answers."""
    out = {}
    for name, cam in (cameras or {}).items():
        about = dict((cam or {}).get("about") or {})
        if about.get("faces") is not None and str(about["faces"]).lower() not in FACE_VIEWS:
            raise ValueError(f"Camera {name}: about.faces must be one of {', '.join(FACE_VIEWS)}")
        out[name] = about
    return out


def camera_aliases(cameras: dict[str, Any] | None) -> dict[str, str]:
    """alias (lower case) -> camera name, from each camera's `about.aliases`."""
    out: dict[str, str] = {}
    for name, about in camera_about(cameras).items():
        for alias in about.get("aliases") or []:
            alias = str(alias).strip().lower()
            if out.get(alias, name) != name:
                raise ValueError(f"Alias {alias!r} is used by cameras {out[alias]} and {name}")
            out[alias] = name
    return out


def is_stream_url(source: str | int) -> bool:
    return isinstance(source, str) and source.lower().startswith(STREAM_SCHEMES)


def mask_url(url: str | int) -> str:
    """Hide the password of a stream URL so it can be printed or logged."""
    if not is_stream_url(url):
        return str(url)
    parts = urlsplit(str(url))
    if parts.password is None:
        return str(url)
    netloc = f"{parts.username}:***@{parts.netloc.rsplit('@', 1)[1]}"
    return urlunsplit(parts._replace(netloc=netloc))


# ---------------------------------------------------------------------------- live capture

_OPEN_LOCK = threading.Lock()  # OPENCV_FFMPEG_CAPTURE_OPTIONS is process-global


class LiveCapture:
    """Reads a network stream on a background thread, keeping only the newest frame.

    On CPU, detection is usually slower than the camera frame rate. Reading synchronously
    lets the decoder buffer fill up, so results lag further and further behind reality.
    Here stale frames are dropped instead. The stream is reopened automatically when it
    drops (camera reboot, cable unplugged, PoE switch power cycle).
    """

    def __init__(
        self,
        url: str,
        transport: str = "tcp",
        timeout: float = 5.0,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self.url = url
        self.transport = transport  # tcp avoids the smeared frames UDP gives on lossy links
        self.timeout = timeout
        self.max_reconnect_delay = max_reconnect_delay
        self.connected = False
        self.fps = 25.0  # reported by the camera once connected
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_read = 0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"capture {mask_url(url)}"
        )

    def __enter__(self) -> LiveCapture:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _open(self) -> cv2.VideoCapture:
        ms = int(self.timeout * 1000)
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms]
        if not self.url.lower().startswith("rtsp"):
            return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
        with _OPEN_LOCK:
            previous = os.environ.get(key)
            os.environ[key] = f"rtsp_transport;{self.transport}"
            try:
                return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            finally:
                if previous is None:
                    del os.environ[key]
                else:
                    os.environ[key] = previous

    def _run(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                cap.release()
                log.warning("Cannot open %s, retrying in %.0f s", mask_url(self.url), delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                continue
            log.info("Connected to %s", mask_url(self.url))
            self.connected, delay = True, 1.0
            self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                with self._cond:
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
            cap.release()
            self.connected = False
            if not self._stop.is_set():
                log.warning("Lost %s, reconnecting", mask_url(self.url))

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        """Newest frame not returned before; waits up to `timeout` s, then returns None."""
        with self._cond:
            fresh = self._cond.wait_for(
                lambda: self._seq != self._last_read or self._stop.is_set(), timeout
            )
            if not fresh or self._stop.is_set():
                return None
            self._last_read = self._seq
            return self._frame

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=self.timeout + 1)


# ---------------------------------------------------------------------------- discovery

WS_DISCOVERY_ADDR = ("239.255.255.250", 3702)
WS_DISCOVERY_PROBE = """<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
  xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
  xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
  xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
 <e:Header>
  <w:MessageID>uuid:{message_id}</w:MessageID>
  <w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
  <w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
 </e:Header>
 <e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body>
</e:Envelope>"""


def local_ipv4_addresses() -> list[str]:
    """IPv4 addresses of this machine's interfaces (Wi-Fi, Ethernet, ...), loopback excluded."""
    infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    return sorted({str(info[4][0]) for info in infos} - {"127.0.0.1"})


def parse_probe_match(data: bytes, sender: str) -> dict[str, Any] | None:
    """Extract host, name, hardware and service URLs from a WS-Discovery ProbeMatch."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    fields: dict[str, str] = {}
    for el in root.iter():
        fields.setdefault(el.tag.rsplit("}", 1)[-1], (el.text or "").strip())
    if "ProbeMatch" not in fields:
        return None

    scopes: dict[str, str] = {}
    for scope in fields.get("Scopes", "").split():
        prefix = "onvif://www.onvif.org/"
        if scope.startswith(prefix) and "/" in scope[len(prefix):]:
            key, value = scope[len(prefix):].split("/", 1)
            scopes.setdefault(key, unquote(value))
    xaddrs = fields.get("XAddrs", "").split()
    host = urlsplit(xaddrs[0]).hostname if xaddrs else None
    return {
        "host": host or sender,
        "name": scopes.get("name", ""),
        "hardware": scopes.get("hardware", ""),
        "xaddrs": xaddrs,
    }


def discover_onvif(
    timeout: float = 3.0, interfaces: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Multicast an ONVIF probe out of every interface and collect camera replies.

    Sending per interface matters on a laptop: the default multicast route usually goes
    out Wi-Fi, not the Ethernet port the PoE switch is plugged into.
    """
    probe = WS_DISCOVERY_PROBE.format(message_id=uuid.uuid4()).encode()
    socks: list[socket.socket] = []
    for ip in interfaces or local_ipv4_addresses():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
            sock.bind((ip, 0))
            sock.sendto(probe, WS_DISCOVERY_ADDR)
            socks.append(sock)
        except OSError as exc:
            log.warning("Cannot probe from %s: %s", ip, exc)
            sock.close()

    found: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + timeout
    try:
        while socks and (remaining := deadline - time.monotonic()) > 0:
            readable, _, _ = select.select(socks, [], [], remaining)
            for sock in readable:
                try:
                    data, (sender, _) = sock.recvfrom(65535)
                except OSError:
                    continue
                match = parse_probe_match(data, sender)
                if match:
                    found.setdefault(match["host"], match)
    finally:
        for sock in socks:
            sock.close()
    return sorted(found.values(), key=lambda m: _host_order(m["host"]))


def _host_order(host: str) -> tuple[int, int, str]:
    """Sort key: IPv4 addresses in numeric order, then IPv6 ones, then host names (an XAddr
    may give either)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return 7, 0, host
    return ip.version, int(ip), host


def scan_subnet(
    network: str, ports: Iterable[int] = (554,), timeout: float = 0.5
) -> list[tuple[str, int]]:
    """TCP-connect scan for open ports, for cameras with ONVIF discovery turned off."""
    net = ipaddress.ip_network(network, strict=False)
    if net.num_addresses > 1024:
        raise ValueError(f"{network} is too large to scan; use a /22 or smaller")
    targets = [(str(host), port) for host in net.hosts() for port in ports]

    def is_open(target: tuple[str, int]) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex(target) == 0

    with ThreadPoolExecutor(max_workers=128) as pool:
        return [t for t, ok in zip(targets, pool.map(is_open, targets)) if ok]


# ---------------------------------------------------------------------------- CLI


def load_cameras(config_path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config.get("cameras") or {}


def check_camera(
    source: str, cameras: dict[str, Any], seconds: float, save: Path | None
) -> None:
    url = str(resolve_source(source, cameras))
    print(f"Opening {mask_url(url)} ...")
    with LiveCapture(url) as cap:
        frame = cap.read(timeout=15)
        if frame is None:
            raise SystemExit(
                "No frame received. Check IP/port/path/credentials, that this machine has an "
                "address in the camera's subnet, and try the stream in VLC."
            )
        start, frames = time.monotonic(), 0
        while time.monotonic() - start < seconds:
            if cap.read(timeout=1) is not None:
                frames += 1
    h, w = frame.shape[:2]
    print(f"OK: {w}x{h}, ~{frames / seconds:.1f} FPS received")
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save), frame)
        print(f"Snapshot saved to {save}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="IP camera tools")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show configured cameras")
    p_disc = sub.add_parser("discover", help="Find ONVIF cameras on the local network")
    p_disc.add_argument("--timeout", type=float, default=3.0)
    p_disc.add_argument("--interface", action="append", help="Local IP to probe from (repeatable)")
    p_scan = sub.add_parser("scan", help="Find hosts with RTSP open, e.g. 192.168.1.0/24")
    p_scan.add_argument("network")
    p_scan.add_argument("--ports", type=int, nargs="+", default=[554])
    p_test = sub.add_parser("test", help="Connect to a camera (name or URL) and grab frames")
    p_test.add_argument("source")
    p_test.add_argument("--seconds", type=float, default=3.0)
    p_test.add_argument("--save", type=Path, help="Save a snapshot, e.g. data/cam.jpg")
    args = parser.parse_args()

    if args.command == "list":
        cameras = load_cameras(args.config)
        if not cameras:
            print(f"No cameras configured in {args.config}")
        for name, cam in cameras.items():
            try:
                print(f"{name:15} {mask_url(camera_url(cam))}")
            except ValueError as exc:
                print(f"{name:15} ERROR: {exc}")

    elif args.command == "discover":
        interfaces = args.interface or local_ipv4_addresses()
        print("Probing from:", ", ".join(interfaces))
        if any(ip.startswith("169.254.") for ip in interfaces):
            print(
                "  Note: a 169.254.x.x address means that adapter got no DHCP lease. Cameras "
                "on a plain PoE switch usually need a static IP on the laptop in their subnet."
            )
        cameras = discover_onvif(args.timeout, interfaces)
        for cam in cameras:
            label = " ".join(filter(None, (cam["hardware"], cam["name"])))
            print(f"{cam['host']:15} {label:30} {' '.join(cam['xaddrs'])}")
        if not cameras:
            print("No ONVIF cameras answered (ONVIF may be disabled; try `scan <subnet>`).")

    elif args.command == "scan":
        hits = scan_subnet(args.network, args.ports)
        for host, port in hits:
            print(f"{host}:{port} open")
        if not hits:
            print("Nothing found.")

    elif args.command == "test":
        check_camera(args.source, load_cameras(args.config), args.seconds, args.save)


if __name__ == "__main__":
    main()
