"""
Disable native telemetry before inference imports, including inherited child processes.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("previous", [None, "0", "1"])
def test_package_disables_telemetry_before_runtime_and_child_imports(
    tmp_path: Path, previous: str | None,
) -> None:
    """
    Verify early opt-out without importing inference libraries or loading user preferences.
    """
    environment = dict(os.environ)
    environment.pop("ORT_DISABLE_TELEMETRY", None)
    if previous is not None:
        environment["ORT_DISABLE_TELEMETRY"] = previous
    runtime = tmp_path / "onnxruntime.py"
    runtime.write_text(
        'import os\nassert os.environ["ORT_DISABLE_TELEMETRY"] == "1"\n', encoding="utf-8"
    )
    script = '''
import json
import os
import subprocess
import sys
import based_thermals
assert "onnxruntime" not in sys.modules
assert "numpy" not in sys.modules
assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
sys.path.insert(0, sys.argv[1])
import onnxruntime
child = subprocess.run(
    [sys.executable, "-c", "import onnxruntime; print('disabled')"],
    cwd=sys.argv[1], capture_output=True, text=True, check=True,
)
print(json.dumps({"child": child.stdout.strip(), "stderr": child.stderr}))
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env=environment,
        capture_output=True, text=True, check=True, timeout=20,
    )
    assert json.loads(result.stdout) == {"child": "disabled", "stderr": ""}
