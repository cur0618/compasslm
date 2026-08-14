import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResponseToneTests(unittest.TestCase):
    def test_prompts_and_fallbacks_use_polite_friendly_korean(self):
        prompts_source = (ROOT / "src" / "compass_ai" / "prompts.py").read_text(encoding="utf-8")
        main_source = (ROOT / "src" / "main.py").read_text(encoding="utf-8")
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        template_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")

        self.assertIn("친절하고 자연스러운 존댓말", prompts_source)
        self.assertIn("짧고 정확하지만 친절한 존댓말", prompts_source)
        self.assertIn("공손하고 친절한 존댓말", main_source)
        self.assertIn("죄송하지만", main_source)
        self.assertIn("실시간으로 확인", main_source)
        self.assertIn("안녕하세요. 편하게 질문해 주시면", template_source)
        self.assertIn("안녕하세요. 궁금하신 내용을 편하게 말씀해 주세요.", script_source)


if __name__ == "__main__":
    unittest.main()
