import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromptContractTests(unittest.TestCase):
    def test_answer_system_prompt_forbids_markdown(self):
        source = (ROOT / "src" / "compass_ai" / "prompts.py").read_text(encoding="utf-8")
        self.assertIn("마크다운", source)

    def test_runtime_instructions_forbid_markdown(self):
        source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
        self.assertIn("마크다운", source)

    def test_answer_task_prompt_forbids_markdown(self):
        source = (ROOT / "src" / "compass_ai" / "service.py").read_text(encoding="utf-8")
        self.assertIn("마크다운", source)


if __name__ == "__main__":
    unittest.main()
