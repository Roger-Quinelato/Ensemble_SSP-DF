from src.utils import tracking


class _DummyRunInfo:
    def __init__(self, run_id):
        self.run_id = run_id


class _DummyRun:
    def __init__(self, run_id):
        self.info = _DummyRunInfo(run_id)


class _DummyMlflow:
    def __init__(self):
        self.experiment_name = None
        self.started_run_name = None
        self.logged_params = None
        self.logged_metrics = None
        self.logged_step = None
        self.logged_artifact = None
        self.ended_status = None

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name

    def start_run(self, run_name=None):
        self.started_run_name = run_name
        return _DummyRun("run-123")

    def log_params(self, params):
        self.logged_params = params

    def log_metrics(self, metrics, step=None):
        self.logged_metrics = metrics
        self.logged_step = step

    def log_artifact(self, path):
        self.logged_artifact = path

    def end_run(self, status="FINISHED"):
        self.ended_status = status


def test_tracking_active_respects_flags(monkeypatch):
    monkeypatch.setattr(tracking, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(tracking, "_DISABLED", False)
    assert tracking.tracking_active() is True

    monkeypatch.setattr(tracking, "_DISABLED", True)
    assert tracking.tracking_active() is False


def test_init_experiment_and_logs(monkeypatch, tmp_path):
    dummy = _DummyMlflow()
    monkeypatch.setattr(tracking, "mlflow", dummy, raising=False)
    monkeypatch.setattr(tracking, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(tracking, "_DISABLED", False)

    run = tracking.init_experiment("exp_test", run_id="run_name")
    assert run.info.run_id == "run-123"
    assert dummy.experiment_name == "exp_test"
    assert dummy.started_run_name == "run_name"

    tracking.log_params({"x": 1, "flag": True})
    assert dummy.logged_params["x"] == "1"
    assert dummy.logged_params["flag"] == "True"

    tracking.log_metrics({"a": 1, "b": 2.5, "skip": True}, step=7)
    assert dummy.logged_metrics == {"a": 1.0, "b": 2.5}
    assert dummy.logged_step == 7

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("ok", encoding="utf-8")
    tracking.log_artifact(str(artifact))
    assert dummy.logged_artifact == str(artifact)

    tracking.end_run(status="FAILED")
    assert dummy.ended_status == "FAILED"


def test_tracking_handles_mlflow_exceptions(monkeypatch):
    class _BrokenMlflow(_DummyMlflow):
        def start_run(self, run_name=None):
            raise RuntimeError("boom_mlflow")

    monkeypatch.setattr(tracking, "mlflow", _BrokenMlflow(), raising=False)
    monkeypatch.setattr(tracking, "_MLFLOW_AVAILABLE", True)
    monkeypatch.setattr(tracking, "_DISABLED", False)

    # Nao deve propagar excecao.
    run = tracking.init_experiment("exp", run_id="x")
    assert run is None
