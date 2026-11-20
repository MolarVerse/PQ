import os
from pathlib import Path
import subprocess
import tempfile


class LocalReleaseRepository:
    """A Git repository and its bare remote; no network or host configuration."""

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote.git"
        self.branch = self.root / "branch"
        self.git_environment = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Release fixture",
            "GIT_COMMITTER_NAME": "Release fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        }
        for arguments in (
            ("init", "--bare", str(self.remote)),
            ("init", str(self.branch)),
        ):
            subprocess.run(
                ["git", *arguments],
                env=self.git_environment,
                check=True,
                capture_output=True,
            )
        self.git("remote", "add", "origin", str(self.remote))
        for identity in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
            self.git("var", identity)
        self.tree = self.git("mktree", input="").stdout.strip()
        self.commit = self.git(
            "commit-tree", self.tree, input="fixture\n"
        ).stdout.strip()
        self.git("update-ref", "refs/heads/main", self.commit)
        self.git("symbolic-ref", "HEAD", "refs/heads/main")
        self.git("push", "origin", "main")

    def git(self, *arguments, input=None, remote=False):
        return subprocess.run(
            ["git", "-C", str(self.remote if remote else self.branch), *arguments],
            input=input,
            capture_output=True,
            text=True,
            check=True,
            env=self.git_environment,
        )

    def tag(self, version, commit=None, annotated=False):
        target = commit or self.commit
        if annotated:
            self.git("tag", "-a", version, target, "-m", "release fixture")
            self.git("push", "origin", f"refs/tags/{version}")
        else:
            self.git("push", "origin", f"{target}:refs/tags/{version}")
