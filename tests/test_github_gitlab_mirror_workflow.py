"""Contract tests for the GitHub-to-AI-Gov GitLab synchronization workflow."""

from __future__ import annotations

from pathlib import Path
import unittest


class GitHubGitLabMirrorWorkflowTests(unittest.TestCase):
    def test_workflow_uses_secret_backed_non_destructive_pushes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = root / ".github" / "workflows" / "mirror-to-aigov-gitlab.yml"

        text = workflow.read_text(encoding="utf-8")

        self.assertIn("push:", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("# GitHub에서 작업한 브랜치·태그 변경은 공공 GitLab에 자동 반영됩니다.", text)
        self.assertIn("# GitHub를 기준 저장소로 사용하며, 필요하면 Actions 화면에서 수동 실행할 수 있습니다.", text)
        self.assertIn("secrets.AIGOV_GITLAB_TOKEN", text)
        self.assertNotIn("AIGOV_GITLAB_TOKEN:", text)
        self.assertIn("https://oauth2:${{ secrets.AIGOV_GITLAB_TOKEN }}@", text)
        self.assertIn("gitlab.aigov.go.kr/cur0618/compasslm.git", text)
        self.assertIn("git push gitlab --all", text)
        self.assertIn("git push gitlab --tags", text)
        self.assertNotIn("git push --mirror", text)


if __name__ == "__main__":
    unittest.main()
