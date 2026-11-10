"""Workspace isolation for file and process operations.

``Workspace`` confines every path to one root directory: absolute paths,
``..`` and symlinks that resolve outside the root are all rejected.

``LocalShellRunner`` runs commands with the workspace as cwd, a scrubbed
environment, a timeout that kills the whole process group, and POSIX resource
limits. It limits accidents; it is NOT a security boundary against hostile
code, which can still read files outside the root by absolute path. For that,
use ``DockerShellRunner`` (no network, only the workspace mounted) or another
OS-level sandbox.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


class SandboxViolation(Exception):
    pass


class Workspace:
    # Harness-owned state (trash, the shell's HOME and TMPDIR): read-only to
    # file tools and hidden from listings so the agent doesn't mistake it for
    # project files.
    HARNESS_DIR = ".harness"
    TRASH_DIR = HARNESS_DIR + "/trash"

    def __init__(self, root: str, max_file_bytes: int = 5_000_000, read_only_paths: Optional[List[str]] = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes
        # Workspace-relative files or directories the agent may read but not change.
        self.read_only_paths = read_only_paths if read_only_paths is not None else [".git", self.HARNESS_DIR]

    # --------------------------------------------------------- resolution

    def resolve(self, path: str) -> Path:
        """Map a model-supplied path to a real path inside the root."""
        if "\x00" in path:
            raise SandboxViolation("path contains a NUL byte")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # realpath follows symlinks, so a link pointing outside is caught too.
        real = Path(os.path.realpath(candidate))
        if real != self.root and self.root not in real.parents:
            raise SandboxViolation("%r is outside the workspace" % path)
        return real

    def relative(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel or "."

    def check_writable(self, path: Path) -> None:
        rel = self.relative(path)
        for ro in self.read_only_paths:
            if rel == ro or rel.startswith(ro.rstrip("/") + "/"):
                raise SandboxViolation("%r is read-only" % rel)

    # --------------------------------------------------------- operations

    def read_text(self, path: str, offset: int = 0, limit: Optional[int] = None) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise FileNotFoundError("no such file: %s" % path)
        if p.stat().st_size > self.max_file_bytes:
            raise SandboxViolation("%s is larger than %d bytes" % (path, self.max_file_bytes))
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        end = None if limit is None else offset + limit
        return "".join(lines[offset:end])

    def write_text(self, path: str, content: str) -> Path:
        p = self.resolve(path)
        self.check_writable(p)
        data = content.encode()
        if len(data) > self.max_file_bytes:
            raise SandboxViolation("content is larger than %d bytes" % self.max_file_bytes)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp-harness")
        tmp.write_bytes(data)
        os.replace(tmp, p)  # atomic: no half-written files
        return p

    def delete(self, path: str) -> str:
        """Soft delete: move into the workspace trash so it can be restored."""
        p = self.resolve(path)
        if p == self.root:
            raise SandboxViolation("refusing to delete the workspace root")
        self.check_writable(p)
        if not p.exists():
            raise FileNotFoundError("no such file: %s" % path)
        trash = self.root / self.TRASH_DIR / time.strftime("%Y%m%d-%H%M%S")
        dest = trash / self.relative(p)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        return self.relative(dest)

    def list_dir(self, path: str = ".", max_entries: int = 500) -> List[str]:
        p = self.resolve(path)
        if not p.is_dir():
            raise NotADirectoryError("not a directory: %s" % path)
        out = []
        for child in sorted(p.iterdir()):
            if child.name == self.HARNESS_DIR:
                continue
            out.append(self.relative(child) + ("/" if child.is_dir() else ""))
            if len(out) >= max_entries:
                out.append("...(truncated)")
                break
        return out


@dataclass
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    def format(self) -> str:
        parts = ["exit code: %s%s" % (self.exit_code, " (timed out)" if self.timed_out else "")]
        if self.stdout:
            parts.append("stdout:\n" + self.stdout)
        if self.stderr:
            parts.append("stderr:\n" + self.stderr)
        return "\n".join(parts)


class LocalShellRunner:
    SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TERM")

    def __init__(
        self,
        workspace: Workspace,
        timeout: float = 30.0,
        cpu_seconds: int = 30,
        max_file_bytes: int = 50_000_000,
        max_output_bytes: int = 200_000,
        env: Optional[Dict[str, str]] = None,
    ):
        self.workspace = workspace
        self.timeout = timeout
        self.cpu_seconds = cpu_seconds
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes
        # Secrets in the parent environment (API keys, tokens) are not passed down.
        base = {k: os.environ[k] for k in self.SAFE_ENV_KEYS if k in os.environ}
        # Caches that programs drop in $HOME (e.g. ~/Library on macOS) land in .harness/, not the project.
        self.home = workspace.root / workspace.HARNESS_DIR / "home"
        self.tmpdir = workspace.root / workspace.HARNESS_DIR / "tmp"
        base.update({"HOME": str(self.home), "TMPDIR": str(self.tmpdir)})
        base.update(env or {})
        self.env = base

    def _limits(self) -> None:  # runs in the child before exec
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.max_file_bytes, self.max_file_bytes))
        except (ImportError, ValueError, OSError):
            pass

    def run(self, command: str, timeout: Optional[float] = None) -> ShellResult:
        self.home.mkdir(parents=True, exist_ok=True)
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            ["/bin/sh", "-c", command],
            cwd=str(self.workspace.root),
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so we can kill children too
            preexec_fn=self._limits if os.name == "posix" else None,
        )
        try:
            out, err = proc.communicate(timeout=timeout or self.timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            timed_out = True
        cap = self.max_output_bytes
        return ShellResult(
            proc.returncode,
            out[:cap].decode(errors="replace"),
            err[:cap].decode(errors="replace"),
            timed_out,
        )


class DockerShellRunner(LocalShellRunner):
    """Runs each command in a throwaway container: no network, only the
    workspace mounted, memory/CPU/pid caps. Requires Docker."""

    def __init__(self, workspace: Workspace, image: str = "python:3.12-slim", network: str = "none", memory: str = "512m", **kw):
        super().__init__(workspace, **kw)
        self.image, self.network, self.memory = image, network, memory

    def run(self, command: str, timeout: Optional[float] = None) -> ShellResult:
        docker_cmd = (
            "docker run --rm -i --network %s --memory %s --cpus 1 --pids-limit 256 "
            "--read-only --tmpfs /tmp -v %s:/workspace -w /workspace %s sh -c %s"
            % (self.network, self.memory, _sh_quote(str(self.workspace.root)), self.image, _sh_quote(command))
        )
        return super().run(docker_cmd, timeout)


def _sh_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)
