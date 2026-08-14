import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class FrontendVisualContractTests(unittest.TestCase):
    def test_style_defines_blue_gray_semantic_tokens(self):
        source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")
        self.assertIn("--color-primary:", source)
        self.assertIn("--color-user-history:", source)
        self.assertIn("--color-border:", source)

    def test_script_marks_latest_and_history_user_bubbles(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("is-latest-user", source)
        self.assertIn("is-history-user", source)
        self.assertIn("refreshUserBubbleStates", source)

    def test_default_user_bubble_style_stays_blue_until_marked_history_without_gradient(self):
        source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(r"\.message\.user \.bubble\s*\{[^}]*background-color:\s*var\(--accent-soft\)", re.DOTALL),
        )
        self.assertRegex(
            source,
            re.compile(r"\.message\.user\.is-history-user \.bubble\s*\{[^}]*var\(--color-user-history\)", re.DOTALL),
        )
        self.assertNotRegex(
            source,
            re.compile(r"\.message\.user(?:\.is-latest-user)? \.bubble\s*\{[^}]*linear-gradient", re.DOTALL),
        )

    def test_citation_chip_contract_exists(self):
        style_source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn(".citation-chip", style_source)
        self.assertIn("renderAssistantText", script_source)
        self.assertIn("citation-chip", script_source)
        self.assertIn("data-citation-label", script_source)
        self.assertIn("CITATION", script_source)
        self.assertIn(".citation-chip::after", style_source)
        self.assertIn(".citation-chip:focus-visible::after", style_source)

    def test_script_defines_document_pending_status_messages(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("문서를 확인 중입니다. 잠시만 기다려 주세요.", script_source)
        self.assertIn("이용자께서 업로드한 문서를 세세히 보는 중입니다.", script_source)
        self.assertIn("delayMs: 7000", script_source)
        self.assertIn("delayMs: 14000", script_source)
        self.assertIn("document_qa", script_source)
        self.assertNotIn("관련 문서를 찾고 있습니다. 조금만 더 기다려 주세요.", script_source)
        self.assertNotIn("delayMs: 1200", script_source)
        self.assertNotIn("delayMs: 3200", script_source)
        self.assertNotIn("delayMs: 5200", script_source)
        self.assertNotIn("delayMs: 6200", script_source)
        self.assertIn("setTimeout", script_source)

    def test_script_syncs_pending_detection_with_conversation_mode_header(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("X-Conversation-Mode", script_source)
        self.assertIn("data-conversation-mode", script_source)

    def test_auth_and_header_controls_do_not_wrap_korean_labels_vertically(self):
        source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")
        nowrap_selectors = [
            ".auth-mode-btn",
            ".auth-user-badge",
            ".logout-btn",
            ".admin-mode-btn",
            ".header-badge",
        ]
        for selector in nowrap_selectors:
            with self.subTest(selector=selector):
                self.assertRegex(
                    source,
                    re.compile(
                        rf"{re.escape(selector)}\s*\{{[^}}]*white-space:\s*nowrap",
                        re.DOTALL,
                    ),
                )
                self.assertRegex(
                    source,
                    re.compile(
                        rf"{re.escape(selector)}\s*\{{[^}}]*word-break:\s*keep-all",
                        re.DOTALL,
                    ),
                )

        self.assertRegex(
            source,
            re.compile(r"#chat-header-actions\s*\{[^}]*flex-wrap:\s*wrap", re.DOTALL),
        )
        self.assertRegex(
            source,
            re.compile(r"#chat-header h2\s*\{[^}]*min-width:\s*0", re.DOTALL),
        )


if __name__ == "__main__":
    unittest.main()
