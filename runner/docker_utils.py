"""Shared isolation policy and bounded, exit-code-preserving subprocess capture."""
import subprocess
from tempfile import TemporaryFile
from threading import Thread
from time import monotonic

ACTORS = ("agent1", "agent2")
CONTAINER_MEMORY = "4g"
CONTAINER_CPUS = 2
CONTAINER_PIDS = 256
COMMAND_CAPTURE_BYTES = 64 * 1024
EVALUATOR_CAPTURE_BYTES = 1024 * 1024
MAX_PATCH_BYTES = 64 * 1024 * 1024
AGENT_OUTPUT_CHARS = 12000


def isolated_run_args(name, image, *, workdir=None, lifetime="infinity"):
    args = [
        "run", "-d", "--pull=never", "--name", name, "--network=none",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        f"--pids-limit={CONTAINER_PIDS}", f"--memory={CONTAINER_MEMORY}",
        f"--cpus={CONTAINER_CPUS}",
    ]
    if workdir:
        args += ["-w", workdir]
    return args + ["--entrypoint=sleep", image, lifetime]


class OutputLimitExceeded(RuntimeError):
    """A complete machine-readable result was required but exceeded its cap."""

    def __init__(self, returncode):
        super().__init__("Subprocess output exceeded the capture limit")
        self.returncode = returncode


def bounded_run(args, *, timeout, limit=COMMAND_CAPTURE_BYTES, data=None, check=False):
    """Drain both pipes concurrently, keeping at most limit bytes per stream.

    Continue draining after truncation so the child runs normally and its actual
    exit code is preserved. Use a file for optional stdin, avoiding pipe-write
    deadlocks. No shell is invoked on the host. The caller owns container cleanup.
    """
    if type(limit) is not int or limit < 1 or timeout <= 0:
        raise ValueError("Positive output limit and timeout required")

    end = monotonic() + timeout
    captured = [bytearray(), bytearray()]
    truncated = [False, False]
    errors = []

    def drain(stream, index):
        try:
            while chunk := stream.read(65536):
                available = limit - len(captured[index])
                captured[index].extend(chunk[:available])
                if len(chunk) > available:
                    truncated[index] = True
        except Exception as exc:
            errors.append(exc)
        finally:
            stream.close()

    with TemporaryFile() as stdin:
        if data is not None:
            stdin.write(data)
            stdin.seek(0)
        process = subprocess.Popen(
            args, stdin=stdin if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        readers = [
            Thread(target=drain, args=(stream, index), daemon=True)
            for index, stream in enumerate((process.stdout, process.stderr))
        ]
        try:
            for reader in readers:
                reader.start()
            process.wait(timeout=max(0, end - monotonic()))
            for reader in readers:
                reader.join(max(0, end - monotonic()))
            if any(reader.is_alive() for reader in readers):
                raise subprocess.TimeoutExpired(args, timeout)
            if errors:
                raise RuntimeError("Subprocess output capture failed") from errors[0]
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)
            for reader, stream in zip(readers, (process.stdout, process.stderr)):
                if reader.ident is None:
                    stream.close()
                else:
                    reader.join(0.1)
            raise

    result = subprocess.CompletedProcess(
        args, process.returncode, bytes(captured[0]), bytes(captured[1]),
    )
    result.output_truncated = any(truncated)
    if check:
        result.check_returncode()
    return result


def require_complete(result):
    if getattr(result, "output_truncated", False):
        raise OutputLimitExceeded(result.returncode)
    return result


def agent_feedback(result):
    output = result.stdout + result.stderr
    return {
        "output": output[:AGENT_OUTPUT_CHARS],
        "exit_code": result.returncode,
        "output_truncated": (
            getattr(result, "output_truncated", False)
            or len(output) > AGENT_OUTPUT_CHARS
        ),
    }
