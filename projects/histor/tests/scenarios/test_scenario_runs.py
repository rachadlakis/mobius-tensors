"""Every bench spec whose clip is here, run and scored (colour, the suite's realtime setting):
its own expectations must hold. Slow: uv run pytest -m scenario. The baselines and the
night looks: python -m histor.scenarios (README, "Scenario bench")."""

from __future__ import annotations

import pytest

from histor.engine import scenarios

SUITES = scenarios.load_suites()
SPECS = scenarios.all_specs()


@pytest.mark.scenario
@pytest.mark.parametrize("spec", SPECS, ids=[f"{s['suite']}/{s['name']}" for s in SPECS])
def test_spec(spec, tmp_path):
    if not scenarios.clip_path(spec).exists():
        pytest.skip(f"clip not here: {scenarios.clip_path(spec)}")
    missing = scenarios.missing_models(scenarios.spec_config(spec, SUITES))
    if missing:
        pytest.skip(f"models not installed: {missing}")
    result = scenarios.run_spec(spec, workdir=tmp_path, suites=SUITES,
                                realtime=SUITES[spec["suite"]].get("realtime", False))
    s = scenarios.score(result, spec, scenarios.scored_types(spec, SUITES),
                        scenarios.known_false_alarms(spec, SUITES))
    if spec.get("xfail") and not s.passed(spec):
        pytest.xfail(spec["xfail"])
    assert s.passed(spec), (f"missed {s.missed}, still only {s.still}, false alarms "
                            f"{[(e['t'], e['type'], e['rule']) for e in s.false_alarms]}, "
                            f"events {[(e['t'], e['type'], e['rule']) for e in result.events]}")
