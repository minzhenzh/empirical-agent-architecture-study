"""SWE-Bench Docker environment with 4 tools for code editing and patch submission."""

import json
import re
import threading
from typing import Dict, List, Optional

from loguru import logger

from agent_scaling.datasets.swebench import SWEBenchInstance
from agent_scaling.env.base import AgentEnvironmentTools
from agent_scaling.env.docker_utils import DockerContainer
from agent_scaling.env.registry import register_env
from agent_scaling.env.tools import cls_tool


@register_env("swebench")
class SWEBenchEnvironment(AgentEnvironmentTools):
    """Docker-based environment for SWE-Bench Verified evaluation.

    Provides 4 tools: bash, read_file, edit_file, submit_patch.
    Multiple agent environments can share the same Docker container for
    multi-agent setups via class-level container caching.
    """

    # Class-level container cache for multi-agent sharing
    _container_cache: Dict[str, DockerContainer] = {}
    _container_locks: Dict[str, threading.Lock] = {}
    _cache_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_instance: Optional[SWEBenchInstance] = self.dataset_instance
        self.is_done = False
        self.success = False
        self.num_steps = 0
        self._container: Optional[DockerContainer] = None
        self._is_container_owner = False
        self._model_patch: str = ""

        if self.dataset_instance is not None:
            self._init_container()

    def _init_container(self) -> None:
        """Initialize or reuse a Docker container for this instance."""
        instance = self.dataset_instance
        assert instance is not None
        instance_id = instance.instance_id

        with SWEBenchEnvironment._cache_lock:
            if instance_id not in SWEBenchEnvironment._container_cache:
                # Create new container
                container = DockerContainer(
                    image=instance.docker_image,
                    workspace_dir="/testbed",
                    timeout=600,
                    memory_limit="8g",
                    network_mode="none",
                )
                container.start()

                # Setup: checkout base commit and clean
                exit_code, stdout, stderr = container.exec_command(
                    f"git checkout {instance.base_commit} && git clean -fd"
                )
                if exit_code != 0:
                    logger.warning(
                        f"Git checkout failed for {instance_id}: {stderr}"
                    )

                SWEBenchEnvironment._container_cache[instance_id] = container
                SWEBenchEnvironment._container_locks[instance_id] = threading.Lock()
                self._is_container_owner = True
                logger.info(f"Created container for {instance_id}")
            else:
                logger.info(f"Reusing container for {instance_id}")

            self._container = SWEBenchEnvironment._container_cache[instance_id]

    def get_instance_prompt_info(self) -> Dict[str, str]:
        """Get prompt info including repo structure."""
        info = {}
        if self.dataset_instance:
            info.update(self.dataset_instance.get_prompt_info())

            # Add repo structure overview
            if self._container:
                _, stdout, _ = self._container.exec_command(
                    "find . -type f -name '*.py' | head -50"
                )
                info["repo_structure"] = stdout

        info["tools_description"] = self.tools_description
        return info

    def env_done(self) -> bool:
        return self.is_done

    def _cleanup(self) -> None:
        """Clean up Docker container if this env is the owner."""
        if not self._is_container_owner or self.dataset_instance is None:
            return

        instance_id = self.dataset_instance.instance_id
        with SWEBenchEnvironment._cache_lock:
            container = SWEBenchEnvironment._container_cache.pop(instance_id, None)
            SWEBenchEnvironment._container_locks.pop(instance_id, None)

        if container is not None:
            container.stop_and_remove()
            logger.info(f"Cleaned up container for {instance_id}")

    @cls_tool
    def bash(self, command: str) -> str:
        """Execute a bash command in the repository container.

        The testbed conda environment is automatically activated so that
        all repository dependencies (pytest, etc.) are available.

        Args:
            command: The bash command to execute.

        Returns:
            Command output (stdout and stderr combined).
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        # Activate the testbed conda env so repo deps (pytest etc.) are available
        activated_cmd = f"source activate testbed 2>/dev/null; {command}"
        exit_code, stdout, stderr = self._container.exec_command(
            activated_cmd, timeout=300
        )

        output = stdout
        if stderr:
            output += f"\n[STDERR]\n{stderr}"
        if exit_code != 0:
            output += f"\n[Exit code: {exit_code}]"

        # Truncate long output to prevent context window pollution
        max_output = 8000
        if len(output) > max_output:
            # Keep first 2000 + last 5000 chars (most useful info is at the end)
            output = (
                output[:2000]
                + f"\n\n... [{len(output) - 7000} characters truncated] ...\n\n"
                + output[-5000:]
            )

        return output

    @cls_tool
    def read_file(
        self,
        file_path: str,
        start_line: int = 1,
        end_line: int = -1,
    ) -> str:
        """Read a file from the repository with optional line range.

        Args:
            file_path: Path to the file relative to the repository root.
            start_line: Starting line number (1-indexed, default: 1).
            end_line: Ending line number (inclusive, default: -1 for end of file).

        Returns:
            File contents with line numbers.
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        if end_line == -1:
            cmd = f"cat -n {file_path}"
        else:
            cmd = f"sed -n '{start_line},{end_line}p' {file_path} | cat -n"
            # Adjust line numbers to reflect actual file positions
            cmd = f"awk 'NR>={start_line} && NR<={end_line} {{printf \"%6d\\t%s\\n\", NR, $0}}' {file_path}"

        exit_code, stdout, stderr = self._container.exec_command(cmd)

        if exit_code != 0:
            return f"Error reading file: {stderr}"

        return stdout

    @cls_tool
    def edit_file(self, file_path: str, old_content: str, new_content: str) -> str:
        """Replace exact content in a file within the repository.

        Args:
            file_path: Path to the file relative to the repository root.
            old_content: The exact text to find and replace.
            new_content: The replacement text.

        Returns:
            Confirmation message or error.
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        # Read current file content
        try:
            current = self._container.read_file(file_path)
        except FileNotFoundError:
            return f"Error: File {file_path} not found."

        if old_content not in current:
            return (
                f"Error: The specified old_content was not found in {file_path}. "
                "Make sure to include the exact text including whitespace and indentation."
            )

        count = current.count(old_content)
        if count > 1:
            return (
                f"Error: old_content matches {count} locations in {file_path}. "
                "Please provide more surrounding context to make the match unique."
            )

        # Apply replacement
        new_file_content = current.replace(old_content, new_content, 1)
        self._container.write_file(file_path, new_file_content)

        # Lint check for Python files — reject syntactically invalid edits
        if file_path.endswith('.py'):
            lint_exit, lint_out, lint_err = self._container.exec_command(
                f"python3 -c \"compile(open('{file_path}').read(), '{file_path}', 'exec')\" 2>&1",
                timeout=10,
            )
            if lint_exit != 0:
                # Revert the edit
                self._container.write_file(file_path, current)
                error_msg = (lint_out + lint_err).strip().split('\n')[-1]
                return (
                    f"Edit reverted — syntax error in {file_path}: {error_msg}\n"
                    "Please fix the syntax and try again."
                )

        return f"Successfully edited {file_path}."

    def _is_django_repo(self) -> bool:
        """Check if the current instance is a Django repository."""
        if self.dataset_instance is None:
            return False
        return "django" in self.dataset_instance.repo.lower()

    def _convert_django_test_to_runtests_format(self, test_str: str) -> str:
        """Convert Django unittest format to runtests.py format.

        Django fail_to_pass strings look like:
            'test_method (module.path.ClassName)'
            'test_method (module.path.ClassName) (1)'  (with duplicate marker)
        We need to convert to:
            'module.path'  (the test module to pass to runtests.py)
        And the specific test:
            'module.path.ClassName.test_method'
        """
        test_str = test_str.strip()
        # Match pattern: test_name (module.path.ClassName)
        match = re.match(r'^(\w+)\s+\(([^)]+)\)', test_str)
        if match:
            test_name = match.group(1)
            module_class = match.group(2)
            return f"{module_class}.{test_name}"
        # Already in dotted format or other format, return as-is
        return test_str

    def _get_django_test_labels(self, tests: List[str]) -> List[str]:
        """Extract unique Django test module labels for runtests.py.

        From tests like:
            ['test_a (auth_tests.test_validators.MyClass)',
             'test_b (auth_tests.test_validators.MyClass)']
        Returns: ['auth_tests.test_validators']
        """
        labels = set()
        for test_str in tests:
            match = re.match(r'^\w+\s+\(([^)]+)\)', test_str.strip())
            if match:
                module_class = match.group(1)
                # Get the module path (everything up to the last dot = class name)
                parts = module_class.rsplit('.', 1)
                if len(parts) == 2:
                    labels.add(parts[0])
                else:
                    labels.add(module_class)
            else:
                labels.add(test_str)
        return list(labels)

    def _run_test(self, test: str, timeout: int = 120) -> tuple:
        """Run a single test, using the appropriate runner for the repo type.

        Returns (exit_code, stdout, stderr).
        """
        assert self._container is not None

        if self._is_django_repo():
            # Django: use ./tests/runtests.py with converted test format
            converted = self._convert_django_test_to_runtests_format(test)
            # Extract the test label (module path without class/method for runtests.py)
            match = re.match(r'^\w+\s+\(([^)]+)\)', test.strip())
            if match:
                module_class = match.group(1)
                # Get module path for runtests.py --parallel 1
                parts = module_class.rsplit('.', 1)
                test_label = parts[0] if len(parts) == 2 else module_class
            else:
                test_label = test

            cmd = (
                f"cd /testbed && source activate testbed 2>/dev/null; "
                f"python tests/runtests.py --parallel 1 {test_label} 2>&1"
            )
        else:
            # All other repos: use pytest (no --timeout flag, not always available)
            cmd = (
                f"cd /testbed && source activate testbed 2>/dev/null; "
                f"python -m pytest {test} -x 2>&1"
            )

        return self._container.exec_command(cmd, timeout=timeout)

    def _run_django_tests(self, tests: List[str], timeout: int = 120) -> tuple:
        """Run Django tests using runtests.py with all test labels at once.

        Returns (exit_code, stdout, stderr).
        """
        assert self._container is not None
        labels = self._get_django_test_labels(tests)
        labels_str = " ".join(labels)
        cmd = (
            f"cd /testbed && source activate testbed 2>/dev/null; "
            f"python tests/runtests.py --parallel 1 {labels_str} 2>&1"
        )
        return self._container.exec_command(cmd, timeout=timeout)

    @cls_tool
    def run_tests(self, test_names: str = "") -> str:
        """Run tests to check if your current changes fix the issue.

        This does NOT submit your patch — use it to verify your fix before submitting.
        If no test_names are provided, runs the fail-to-pass tests from the issue.

        Args:
            test_names: Optional space-separated test names to run. If empty,
                        runs the fail-to-pass tests from the issue.

        Returns:
            Test results showing which tests pass and fail.
        """
        assert self._container is not None, "Container not initialized"
        assert self.dataset_instance is not None

        self.num_steps += 1
        instance = self.dataset_instance

        # Determine which tests to run
        if test_names.strip():
            tests = test_names.strip().split()
        else:
            tests = instance.fail_to_pass_tests

        if not tests:
            return "No tests to run."

        results = []
        all_pass = True

        if self._is_django_repo():
            exit_code, stdout, stderr = self._run_django_tests(tests, timeout=180)
            output_combined = stdout + stderr
            if exit_code == 0 and ("OK" in output_combined or "ok" in output_combined.lower()):
                for test in tests:
                    results.append(f"  PASS: {test}")
            else:
                all_pass = False
                for test in tests:
                    results.append(f"  FAIL: {test}")
                # Include truncated output for debugging
                if len(output_combined) > 2000:
                    output_combined = output_combined[-2000:]
                results.append(f"\n[Test output (last 2000 chars)]:\n{output_combined}")
        else:
            for test in tests:
                exit_code, stdout, stderr = self._run_test(test)
                passed = exit_code == 0
                if not passed:
                    all_pass = False
                    # Include failure details (truncated)
                    fail_output = (stdout + stderr)[-1000:]
                    results.append(f"  FAIL: {test}\n    {fail_output}")
                else:
                    results.append(f"  PASS: {test}")

        status = "ALL PASSED" if all_pass else "SOME FAILED"
        return (
            f"## Test Results: {status}\n\n"
            + "\n".join(results)
            + "\n\nUse this feedback to fix any remaining issues before calling submit_patch."
        )

    @cls_tool
    def find_file(self, file_name: str, directory: str = ".") -> str:
        """Find files matching a name pattern in the repository.

        Args:
            file_name: File name or pattern to search for (supports wildcards like *.py).
            directory: Directory to search in (relative to /testbed, default: root).

        Returns:
            List of matching file paths.
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        # Use find with name matching, limit results
        cmd = f"find {directory} -type f -name '{file_name}' 2>/dev/null | head -50"
        exit_code, stdout, stderr = self._container.exec_command(cmd, timeout=30)

        if not stdout.strip():
            return f"No files matching '{file_name}' found in {directory}."

        lines = stdout.strip().split('\n')
        result = f"Found {len(lines)} file(s):\n" + "\n".join(lines)
        if len(lines) == 50:
            result += "\n(Results truncated to 50 files. Narrow your search.)"
        return result

    @cls_tool
    def search_dir(self, search_term: str, directory: str = ".") -> str:
        """Search for a term in all files within a directory.

        Args:
            search_term: The text or regex pattern to search for.
            directory: Directory to search in (relative to /testbed, default: root).

        Returns:
            Matching lines grouped by file, limited to prevent context overflow.
        """
        assert self._container is not None, "Container not initialized"
        self.num_steps += 1

        # Use grep with line numbers, limit output
        cmd = (
            f"grep -rn --include='*.py' '{search_term}' {directory} 2>/dev/null "
            f"| head -30"
        )
        exit_code, stdout, stderr = self._container.exec_command(cmd, timeout=30)

        if not stdout.strip():
            return f"No matches for '{search_term}' in {directory}."

        lines = stdout.strip().split('\n')
        result = f"Found matches ({len(lines)} shown):\n" + "\n".join(lines)
        if len(lines) == 30:
            result += "\n(Results truncated to 30 lines. Narrow your search.)"
        return result

    @cls_tool
    def submit_patch(self, reasoning: str) -> str:
        """Submit the current changes as a patch and run evaluation.

        Generates a git diff of all changes, applies the test patch,
        and runs the fail-to-pass test suite to validate the fix.

        Args:
            reasoning: Brief explanation of the changes made.

        Returns:
            Evaluation results showing whether the fix is correct.
        """
        assert self._container is not None, "Container not initialized"
        assert self.dataset_instance is not None

        self.num_steps += 1
        self.is_done = True

        instance = self.dataset_instance

        # 1. Capture the model patch (git diff)
        self._model_patch = self._container.get_diff(instance.base_commit)

        if not self._model_patch.strip():
            self.success = False
            return "No changes detected. You must modify files before submitting."

        # 2. Write and apply the test patch
        self._container.write_file("/tmp/test_patch.diff", instance.test_patch)
        exit_code, stdout, stderr = self._container.exec_command(
            "cd /testbed && git apply /tmp/test_patch.diff"
        )
        if exit_code != 0:
            logger.warning(f"Test patch apply failed: {stderr}")
            # Try with --3way fallback
            exit_code, stdout, stderr = self._container.exec_command(
                "cd /testbed && git apply --3way /tmp/test_patch.diff"
            )

        # 3. Run fail_to_pass tests
        fail_to_pass = instance.fail_to_pass_tests
        all_pass = True
        results = []

        if self._is_django_repo():
            # Django: run all fail_to_pass tests via runtests.py at once
            exit_code, stdout, stderr = self._run_django_tests(
                fail_to_pass, timeout=180
            )
            output_combined = stdout + stderr
            # Django runtests.py: exit_code=0 means all passed
            if exit_code == 0:
                for test in fail_to_pass:
                    results.append(f"  PASS: {test}")
            else:
                all_pass = False
                # Try to identify which tests failed from output
                for test in fail_to_pass:
                    converted = self._convert_django_test_to_runtests_format(test)
                    # Check multiple output formats Django uses
                    test_name_only = converted.rsplit('.', 1)[-1] if '.' in converted else converted
                    if any(marker in output_combined for marker in [
                        f"FAIL: {converted}",
                        f"ERROR: {converted}",
                        f"FAIL: {test_name_only}",
                        f"ERROR: {test_name_only}",
                    ]):
                        results.append(f"  FAIL: {test}")
                    elif f"ok" in output_combined.lower() and exit_code == 0:
                        results.append(f"  PASS: {test}")
                    else:
                        results.append(f"  FAIL: {test}")
                if not results:
                    results = [f"  FAIL: (all tests) exit_code={exit_code}"]
        else:
            # Non-Django: run each test individually via pytest
            for test in fail_to_pass:
                exit_code, stdout, stderr = self._run_test(test)
                passed = exit_code == 0
                if not passed:
                    all_pass = False
                results.append(
                    f"  {'PASS' if passed else 'FAIL'}: {test}"
                )

        # 4. Determine success based on fail-to-pass only
        # Note: pass-to-pass regression checking removed — it causes false
        # negatives when sampled tests are flaky or environment-sensitive.
        # The official SWE-bench harness evaluates fail-to-pass independently.
        self.success = all_pass

        status = "RESOLVED" if self.success else "FAILED"
        result_text = (
            f"## Evaluation Result: {status}\n\n"
            f"**Reasoning:** {reasoning}\n\n"
            f"**Fail-to-pass tests (must all pass):**\n"
            + "\n".join(results)
            + "\n\n"
        )

        return result_text
