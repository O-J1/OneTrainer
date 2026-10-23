import os
import re
import subprocess
import sys
import threading

from modules.util.enum.PathIOType import PathIOType
from modules.util.tqdm_util import tqdm
from modules.util.ui.validation import validate_path


def tensorboard_mode(config):
    # older presets allowed always-on with the Tensorboard checkbox off
    return "Always-on" if config.tensorboard_always_on else "Training" if config.tensorboard else "Off"


def tensorboard_args(config):
    """Validate before callers create directories or replace a running server."""
    workspace = config.workspace_dir
    error = validate_path(workspace, PathIOType.OUTPUT, output_is_dir=True)
    if workspace.startswith("cloud:"):
        error = "Tensorboard requires a local workspace for downloaded logs"
    if error:
        raise ValueError(error)
    if not isinstance(config.tensorboard_port, int) or not 1 <= config.tensorboard_port <= 65535:
        raise ValueError("Tensorboard port must be between 1 and 65535")
    args = ["--logdir", os.path.abspath(os.path.join(workspace, "tensorboard")),
            "--port", str(config.tensorboard_port), "--samples_per_plugin=images=100,scalars=10000"]
    if config.tensorboard_expose:
        args.append("--bind_all")
    return args


def stop_tensorboard(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)

# Match only known startup notices; unknown output must remain visible.
_QUIET_LINES = {
    "",
    "TensorFlow installation not found - running with reduced feature set.",
    "NOTE: Using experimental fast data loading logic. To disable, pass",
    '"--load_fast=false" and report issues on GitHub. More details:',
    "https://github.com/tensorflow/tensorboard/issues/4784",
    "Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all",
}
_PKG_RESOURCES_WARNING = re.compile(
    r".*[\\/]tensorboard[\\/]default\.py:\d+: UserWarning: pkg_resources is deprecated as an API\..*"
)


class TensorboardProcess(subprocess.Popen):
    """Drain TensorBoard output without hiding failures or blocking the UI."""

    def __init__(self, args):
        self._stopping = False
        self._debug = bool(os.environ.get("OT_DEBUG_WARNINGS"))
        super().__init__([sys.executable, "-m", "tensorboard.main", *args],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        self._reader = threading.Thread(target=self._read_output, name="tensorboard-output", daemon=True)
        self._reader.start()

    def _read_output(self):
        assert self.stdout is not None
        skip_import = False
        with self.stdout:
            for line in self.stdout:
                message = line.strip()
                if skip_import and message == "import pkg_resources":
                    skip_import = False
                    continue
                skip_import = not self._debug and bool(_PKG_RESOURCES_WARNING.fullmatch(message))
                if skip_import:
                    continue
                if self._debug or message not in _QUIET_LINES:
                    tqdm.write(line, end="", file=sys.stderr)
        returncode = self.wait()
        if returncode and not self._stopping:
            tqdm.write(f"TensorBoard exited with code {returncode}", file=sys.stderr)

    def terminate(self):
        if self.poll() is None:
            self._stopping = True
        super().terminate()

    def kill(self):
        if self.poll() is None:
            self._stopping = True
        super().kill()
