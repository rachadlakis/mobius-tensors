import pytest

from histor.engine.cameras import (_host_order, camera_url, load_cameras, mask_url,
                                   parse_probe_match, resolve_source, section_enabled)
from histor.engine.main import default_camera_name


def test_vendor_presets_build_rtsp_urls():
    assert (
        camera_url({"vendor": "hikvision", "host": "192.168.1.64", "stream": "sub"})
        == "rtsp://192.168.1.64:554/Streaming/Channels/102"
    )
    assert (
        camera_url({"vendor": "dahua", "host": "10.0.0.5", "channel": 3})
        == "rtsp://10.0.0.5:554/cam/realmonitor?channel=3&subtype=0"
    )
    assert camera_url({"vendor": "reolink", "host": "cam", "stream": "sub"}).endswith(
        "/h264Preview_01_sub"
    )


def test_credentials_are_url_encoded_and_read_from_env(monkeypatch):
    monkeypatch.setenv("CAM_PASS", "p@ss:w/rd")
    url = camera_url(
        {"host": "cam", "port": 8554, "path": "live", "username": "admin",
         "password": "${CAM_PASS}"}
    )
    assert url == "rtsp://admin:p%40ss%3Aw%2Frd@cam:8554/live"


def test_missing_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    with pytest.raises(ValueError, match="NOPE_NOT_SET"):
        camera_url({"url": "rtsp://admin:${NOPE_NOT_SET}@cam/live"})


def test_generic_camera_without_path_is_rejected():
    with pytest.raises(ValueError, match="path"):
        camera_url({"host": "cam"})


def test_resolve_source_maps_names_only():
    cameras = {"gate": {"url": "rtsp://gate/live"}}
    assert resolve_source("gate", cameras) == "rtsp://gate/live"
    assert resolve_source("video.mp4", cameras) == "video.mp4"
    assert resolve_source(0, None) == 0


def test_mask_url_hides_password():
    assert mask_url("rtsp://admin:secret@10.0.0.5:554/live") == "rtsp://admin:***@10.0.0.5:554/live"
    assert mask_url("rtsp://10.0.0.5/live") == "rtsp://10.0.0.5/live"
    assert mask_url("data/video.mp4") == "data/video.mp4"


PROBE_MATCH = b"""<?xml version="1.0" encoding="UTF-8"?>
<SOAP-ENV:Envelope xmlns:SOAP-ENV="http://www.w3.org/2003/05/soap-envelope"
  xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery">
 <SOAP-ENV:Body><d:ProbeMatches><d:ProbeMatch>
  <d:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/Gate%20Cam
   onvif://www.onvif.org/hardware/DS-2CD2143G2</d:Scopes>
  <d:XAddrs>http://192.168.1.64/onvif/device_service</d:XAddrs>
 </d:ProbeMatch></d:ProbeMatches></SOAP-ENV:Body>
</SOAP-ENV:Envelope>"""


def test_parse_probe_match():
    match = parse_probe_match(PROBE_MATCH, sender="192.168.1.64")
    assert match == {
        "host": "192.168.1.64",
        "name": "Gate Cam",
        "hardware": "DS-2CD2143G2",
        "xaddrs": ["http://192.168.1.64/onvif/device_service"],
    }


def test_parse_probe_match_ignores_other_packets():
    assert parse_probe_match(b"not xml", "1.2.3.4") is None
    assert parse_probe_match(b"<a><b/></a>", "1.2.3.4") is None


def test_a_bare_url_never_becomes_the_camera_name():
    url = "rtsp://admin:secret@10.0.0.5:554/Streaming/Channels/102"
    assert default_camera_name(url, None) == "10.0.0.5"  # no password in events or alerts
    assert default_camera_name(url, "gate") == "gate"
    assert default_camera_name("gate", None) == "gate"  # a configured camera
    assert default_camera_name("data/vtest.avi", None) == "data/vtest.avi"


def test_discovered_hosts_sort_with_ipv6_and_names():
    hosts = ["cam.local", "fe80::1", "192.168.1.10", "192.168.1.9"]
    assert sorted(hosts, key=_host_order) == ["192.168.1.9", "192.168.1.10", "fe80::1",
                                              "cam.local"]


def test_load_cameras_reads_utf8_on_any_platform(tmp_path):
    config = tmp_path / "site.yaml"
    config.write_text("cameras:\n  gate:\n    url: rtsp://10.0.0.5/live\n    about:\n"
                      "      description: البوابة الرئيسية · entrée\n", encoding="utf-8")
    assert load_cameras(config)["gate"]["about"]["description"] == "البوابة الرئيسية · entrée"


def test_section_enabled_for_a_camera():
    assert not section_enabled(None, "gate") and not section_enabled({}, "gate")
    assert not section_enabled({"enabled": False}, "gate")
    assert section_enabled({"enabled": True}, "gate")
    assert section_enabled({"enabled": True, "cameras": []}, "gate")  # empty: every camera
    assert section_enabled({"enabled": True, "cameras": ["gate"]}, "gate")
    assert not section_enabled({"enabled": True, "cameras": ["yard"]}, "gate")
