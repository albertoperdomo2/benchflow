"""Check the current or historical AgentX contract inside its isolated venv."""

from importlib.metadata import version


def check() -> None:
    actual = version("aiperf")
    if actual == "0.12.0":
        from aiperf.config.flags import CLIConfig

        required = {"hf_weka_dataset", "system_idle_gap_cap_seconds"}
        missing = required - set(CLIConfig.model_fields)
        assert not missing, f"AIPerf contract mismatch: {sorted(missing)}"
    elif actual == "0.8.0":
        # The PyPI 0.8.0 wheel lacks these source-era AgentX additions.
        from aiperf.common.config.user_config import UserConfig
        from aiperf.common.scenario.registry import get_scenario

        inputs = UserConfig.model_fields["input"].annotation.model_fields
        loadgen = UserConfig.model_fields["loadgen"].annotation.model_fields
        assert "hf_weka_repo" in inputs, "Historical Weka loader is missing"
        assert "trace_idle_gap_cap_seconds" in loadgen, "Historical pacing is missing"
        scenario = get_scenario("inferencex-agentx-mvp")
        assert scenario.trace_idle_gap_cap_seconds == 60.0
        assert scenario.default_trajectory_start_min_ratio == 0.25
        assert scenario.default_trajectory_start_max_ratio == 0.75
        assert "weka_hf" in scenario.require_loader
    else:
        raise RuntimeError(f"Unsupported BenchFlow AIPerf version: {actual}")
    print(f"BenchFlow AIPerf contract verified: {actual}")


if __name__ == "__main__":
    check()
