from __future__ import annotations

from types import SimpleNamespace

import pytest
from smoke_kit import FakeModels, FakeTelegram, FakeWebhook, record_site


@pytest.fixture(scope="session")
def fakes():
    servers = SimpleNamespace(models=FakeModels(), telegram=FakeTelegram(),
                              webhook=FakeWebhook())
    yield servers
    for server in vars(servers).values():
        server.close()


@pytest.fixture(scope="session")
def site(tmp_path_factory, fakes):
    """One site, recorded once for the whole run (smoke_kit.record_site)."""
    return record_site(tmp_path_factory.mktemp("site"), fakes)
