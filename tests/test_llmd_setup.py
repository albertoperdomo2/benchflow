from benchflow.setup import llmd


def test_wait_for_istiod_uses_deployment_availability(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_command(args, **_kwargs):
        calls.append(args)

    monkeypatch.setattr(llmd, "run_command", fake_run_command)

    llmd._wait_for_istiod("oc", timeout_seconds=120)

    assert calls == [
        [
            "oc",
            "wait",
            "--for=condition=available",
            "deployment/istiod",
            "-n",
            "istio-system",
            "--timeout=120s",
        ]
    ]
