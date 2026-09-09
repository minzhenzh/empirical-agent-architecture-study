"""Docker container lifecycle manager for benchmark environments.

Provides a DockerContainer class that manages container creation, command execution,
file operations, and cleanup. Used by the SWE-Bench Verified environment.
"""

import io
import tarfile
import threading
from typing import Optional, Tuple

import docker
from docker.models.containers import Container
from loguru import logger

# Maximum output size to prevent context window overflow
MAX_OUTPUT_SIZE = 50 * 1024  # 50KB


class DockerContainer:
    """Manages a Docker container lifecycle for benchmark evaluation.

    Supports command execution, file I/O, and git operations within the container.
    Thread-safe for concurrent access from multiple agent environments.
    """

    def __init__(
        self,
        image: str,
        workspace_dir: str = "/workspace",
        timeout: int = 600,
        memory_limit: str = "8g",
        network_mode: str = "none",
    ):
        self.image = image
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.network_mode = network_mode
        self._container: Optional[Container] = None
        self._client: Optional[docker.DockerClient] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Create and start the Docker container."""
        self._client = docker.from_env(timeout=300)  # 5min socket timeout
        logger.info(f"Starting container from image: {self.image}")
        self._container = self._client.containers.run(
            image=self.image,
            command="sleep infinity",
            detach=True,
            working_dir=self.workspace_dir,
            mem_limit=self.memory_limit,
            network_mode=self.network_mode,
            stdin_open=True,
            tty=True,
        )
        logger.info(f"Container started: {self._container.short_id}")

    def exec_command(
        self,
        cmd: str,
        timeout: int = 120,
        workdir: Optional[str] = None,
    ) -> Tuple[int, str, str]:
        """Execute a command inside the container.

        Args:
            cmd: Shell command to execute.
            timeout: Command timeout in seconds.
            workdir: Working directory for command execution.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        assert self._container is not None, "Container not started"
        work = workdir or self.workspace_dir
        # Wrap with timeout to prevent hanging on infinite loops
        escaped_cmd = cmd.replace("'", "'\"'\"'")
        wrapped_cmd = f"cd {work} && timeout {timeout} bash -c '{escaped_cmd}'"

        with self._lock:
            try:
                exec_result = self._container.exec_run(
                    cmd=["bash", "-c", wrapped_cmd],
                    demux=True,
                    workdir=work,
                )
            except Exception as e:
                logger.warning(f"Docker exec failed: {e}")
                return 1, "", f"[Docker exec error: {e}]"

        exit_code = exec_result.exit_code
        stdout_raw, stderr_raw = exec_result.output or (None, None)
        stdout = (stdout_raw or b"").decode("utf-8", errors="replace")
        stderr = (stderr_raw or b"").decode("utf-8", errors="replace")

        # timeout utility returns 124 when command is killed
        if exit_code == 124:
            stderr += f"\n[Command timed out after {timeout}s]"

        # Truncate output to prevent context window overflow
        stdout = _truncate_output(stdout, MAX_OUTPUT_SIZE)
        stderr = _truncate_output(stderr, MAX_OUTPUT_SIZE)

        return exit_code, stdout, stderr

    def read_file(self, path: str) -> str:
        """Read a file from inside the container.

        Args:
            path: Absolute path to the file inside the container.

        Returns:
            File contents as string.
        """
        exit_code, stdout, stderr = self.exec_command(f"cat {path}")
        if exit_code != 0:
            raise FileNotFoundError(f"Failed to read {path}: {stderr}")
        return stdout

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file inside the container using put_archive.

        Args:
            path: Path to the file inside the container (absolute or relative to workspace).
            content: File content to write.
        """
        assert self._container is not None, "Container not started"
        import os

        # Resolve relative paths against workspace_dir for put_archive API
        if not os.path.isabs(path):
            path = os.path.join(self.workspace_dir, path)

        dirname = os.path.dirname(path)
        filename = os.path.basename(path)

        # Create directory if needed
        self.exec_command(f"mkdir -p {dirname}")

        # Create a tar archive in memory with the file content
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=filename)
            tarinfo.size = len(data)
            tar.addfile(tarinfo, io.BytesIO(data))
        tar_stream.seek(0)

        with self._lock:
            self._container.put_archive(dirname, tar_stream)

    def get_diff(self, base_commit: Optional[str] = None) -> str:
        """Get git diff inside the container.

        Args:
            base_commit: Optional base commit to diff against.

        Returns:
            Git diff output as string.
        """
        if base_commit:
            cmd = f"git diff {base_commit}"
        else:
            cmd = "git diff"
        exit_code, stdout, stderr = self.exec_command(cmd)
        if exit_code != 0:
            logger.warning(f"git diff failed: {stderr}")
        return stdout

    def stop_and_remove(self) -> None:
        """Stop and remove the container."""
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            try:
                self._container.remove(force=True)
            except Exception as e:
                logger.warning(f"Error removing container: {e}")
            logger.info(f"Container removed: {self._container.short_id}")
            self._container = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "DockerContainer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_and_remove()


def _truncate_output(text: str, max_size: int) -> str:
    """Truncate output text if it exceeds max_size bytes."""
    if len(text.encode("utf-8")) > max_size:
        truncated = text.encode("utf-8")[:max_size].decode("utf-8", errors="ignore")
        return truncated + "\n... [OUTPUT TRUNCATED] ..."
    return text
