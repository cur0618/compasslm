import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendSessionPersistenceTests(unittest.TestCase):
    def test_script_persists_chat_and_upload_state_locally(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertNotIn("function saveCurrentChat() {\n    return;\n}", source)
        self.assertIn("localStorage.getItem", source)
        self.assertIn("persistChatState", source)
        self.assertIn("resumePendingUploadJobsForKB", source)
        self.assertIn("conversationMode", source)
        self.assertIn("CHAT_STATE_STORAGE_KEY_PREFIX", source)
        self.assertIn("chatStorageKeyForUser", source)

    def test_script_scopes_persisted_chat_state_by_authenticated_user(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function chatStorageKeyForUser(user = currentUser) {", source)
        self.assertIn("currentUser ? chatStorageKeyForUser(currentUser) : null", source)
        self.assertIn("localStorage.getItem(storageKey)", source)
        self.assertIn("localStorage.setItem(storageKey, JSON.stringify(chatStateCache));", source)
        self.assertIn("function reloadChatStateForCurrentUser() {", source)
        self.assertIn("chatStateCache = loadPersistedChatStore();", source)
        self.assertIn("currentKB = normalizeKBName(chatStateCache.currentKB || 'default');", source)

    def test_script_reloads_chat_state_when_auth_user_changes(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn(
            "applyCurrentUser(user || null);\n"
            "    reloadChatStateForCurrentUser();\n"
            "    hideLogin();",
            source,
        )

    def test_script_uses_kb_record_identity_for_chat_buckets(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("let kbRecordByName = new Map();", source)
        self.assertIn("function normalizeKBRecord(raw = {}) {", source)
        self.assertIn("function replaceKBRecords(kbNames = [], records = []) {", source)
        self.assertIn("function chatBucketKeyForKB(kbName) {", source)
        self.assertIn("internal_kb_id", source)
        self.assertIn("const bucketKey = chatBucketKeyForKB(safeKB);", source)
        self.assertIn("chatStateCache.chats[bucketKey]", source)

    def test_script_removes_display_and_internal_chat_buckets_on_kb_delete(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function chatBucketKeysForKB(kbName) {", source)
        self.assertIn("chatBucketKeysForKB(safeKB).forEach(key =>", source)
        self.assertIn("stopUploadPollingForKB(deletedKB);", source)
        self.assertIn("kbRecordByName.delete(normalizeKBName(deletedKB));", source)

    def test_script_keeps_internal_chat_bucket_on_kb_rename(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("const oldBucketKey = chatBucketKeyForKB(oldKey);", source)
        self.assertIn("const newBucketKey = chatBucketKeyForKB(newKey);", source)
        self.assertIn("if (oldBucketKey !== newBucketKey) delete chatStateCache.chats[oldBucketKey];", source)

    def test_script_resets_chat_ui_on_logout_and_auth_loss(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function resetChatStateForLoggedOutUser() {", source)
        self.assertIn("chatStateCache = emptyChatState();", source)
        self.assertIn("chatContainer.innerHTML = '';", source)
        self.assertIn("resetChatStateForLoggedOutUser();", source)

    def test_script_resets_file_input_for_sequential_upload_selection(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("fileInput.value = '';", source)

    def test_file_picker_exposes_all_backend_supported_document_types(self):
        template_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('accept=".txt,.xlsx,.pdf,.hwpx"', template_source)

    def test_script_refreshes_upload_status_when_only_elapsed_time_changes(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("String(job && job.elapsed_seconds)", source)

    def test_script_treats_missing_upload_job_as_terminal_state(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("const TERMINAL_UPLOAD_STATUSES = new Set(['success', 'error', 'timeout', 'not_found']);", source)
        self.assertIn("if (res.status === 404) {", source)
        self.assertIn("status: 'not_found'", source)
        self.assertIn("파일 목록을 확인해 주세요.", source)

    def test_script_timeout_message_warns_that_upload_may_have_stopped(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("자동 확인이 시간 제한을 넘었습니다", source)
        self.assertIn("실제 업로드가 멈췄을 수 있습니다", source)

    def test_script_uses_stage_aware_upload_polling_intervals(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function getUploadPollingIntervalMs(job = null) {", source)
        self.assertIn("stage === 'run_pdf_ocr'", source)
        self.assertIn("stage === 'fallback_pdf_ocr'", source)
        self.assertIn("stage === 'merge_pdf_ocr'", source)
        self.assertIn("stage === 'store_chunks'", source)
        self.assertIn("intervalMs = getUploadPollingIntervalMs(job);", source)
        self.assertIn("getUploadPollingIntervalMs({ status: uploadStatus, progress_stage: uploadProgressStage })", source)

    def test_script_uses_long_polling_for_upload_jobs(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("since_version", source)
        self.assertIn("wait_seconds", source)
        self.assertIn("if (res.status === 204) {", source)
        self.assertIn("document.visibilityState", source)
        self.assertIn("jobVersion", source)

    def test_script_backs_off_after_repeated_idle_long_poll_responses(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("getUploadLongPollWaitSeconds(job = null, idlePollCount = 0)", source)
        self.assertIn("let idlePollCount = 0;", source)
        self.assertIn("idlePollCount += 1;", source)
        self.assertIn("idlePollCount = 0;", source)
        self.assertIn("getUploadLongPollWaitSeconds(latestJob, idlePollCount)", source)

    def test_script_shows_detailed_upload_progress_context(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function formatUploadProgressDetails(job) {", source)
        self.assertIn("current_page", source)
        self.assertIn("total_pages", source)
        self.assertIn("페이지", source)
        self.assertIn("function formatSimpleUploadBody(job) {", source)
        self.assertIn("PDF OCR 실행 중입니다.", source)
        self.assertIn("문서 정리가 끝났습니다.", source)

    def test_script_shows_exact_page_progress_when_page_counts_exist(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function canShowExactUploadPageProgress(job) {", source)
        self.assertIn("currentPage", source)
        self.assertIn("totalPages", source)
        self.assertIn("return `${normalizedCurrentPage}/${normalizedTotalPages}페이지`;", source)
        self.assertNotIn("details.push(`페이지:${Math.round(totalPages)}p`);", source)

    def test_script_hides_chunk_counts_from_upload_bubbles(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertNotIn("조각 ${chunks}", source)
        self.assertNotIn("통합 조각", source)
        self.assertNotIn("${detailLabel}${percentLabel}", source)

    def test_script_hides_raw_ocr_device_labels_and_updates_elapsed_locally(self):
        source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertNotIn("details.push(`OCR:${deviceLabel}`);", source)
        self.assertIn("setInterval(", source)
        self.assertIn("clearInterval(", source)

    def test_script_polls_background_ocr_indicator(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        template_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")
        style_source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")

        self.assertIn("ocr-status-trigger", template_source)
        self.assertIn("ocr-status-popover", template_source)
        self.assertIn("pollBackgroundOcrJobs", script_source)
        self.assertIn("ocr/jobs", script_source)
        self.assertIn("OCR 추출중", script_source)
        self.assertIn(".ocr-status-trigger", style_source)
        self.assertIn(".ocr-status-popover", style_source)

    def test_wiki_panel_exposes_ontology_facts_and_graph_tabs(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        style_source = (ROOT / "src" / "static" / "style.css").read_text(encoding="utf-8")

        self.assertIn("'Facts'", script_source)
        self.assertIn("'Graph'", script_source)
        self.assertIn("ontology/facts", script_source)
        self.assertIn("ontology/overview", script_source)
        self.assertIn("ontology/facts/${encodeURIComponent(factId)}/needs-review", script_source)
        self.assertIn("async function loadOntologyFactDetail(factId)", script_source)
        self.assertIn("function renderOntologyFactDetail(fact = {})", script_source)
        self.assertIn("ontology/facts/${encodeURIComponent(factId)}", script_source)
        self.assertIn("evidence_quote", script_source)
        self.assertIn("source_ref", script_source)
        self.assertIn("item.className = 'wiki-page-item fact-row';", script_source)
        self.assertIn("body.className = 'wiki-page-body';", script_source)
        self.assertIn(".wiki-page-item.fact-row", style_source)
        self.assertIn(".wiki-page-body", style_source)

    def test_wiki_facts_tab_manages_ontology_rebuild_jobs(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn("function renderOntologyRebuildJobs(jobs = [])", script_source)
        self.assertIn("ontology/rebuild/jobs?include_terminal=true", script_source)
        self.assertIn("ontology/rebuild?include_llm=1", script_source)
        self.assertIn("ontology/rebuild/jobs/${encodeURIComponent(jobId)}/cancel", script_source)
        self.assertIn("ontology/rebuild/jobs/${encodeURIComponent(jobId)}/retry", script_source)
        self.assertIn("function scheduleOntologyRebuildPolling(jobs = [])", script_source)
        self.assertIn("function stopOntologyRebuildPolling()", script_source)
        self.assertIn("status === 'processing' ? 'running'", script_source)
        self.assertIn("status === 'success' ? 'completed'", script_source)

    def test_fact_detail_renders_status_and_confidence_history(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn("function renderOntologyFactHistory(history = [])", script_source)
        self.assertIn("fact.history", script_source)
        self.assertIn("previous_confidence", script_source)
        self.assertIn("new_confidence", script_source)

    def test_login_overlay_is_visible_until_auth_succeeds(self):
        template_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('<div id="auth-overlay" class="auth-overlay">', template_source)
        self.assertNotIn('<div id="auth-overlay" class="auth-overlay" hidden>', template_source)

    def test_login_overlay_includes_register_form_controls(self):
        template_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="auth-mode-login"', template_source)
        self.assertIn('id="auth-mode-register"', template_source)
        self.assertIn('id="register-form"', template_source)
        self.assertIn('id="register-id"', template_source)
        self.assertIn('id="register-display-name"', template_source)
        self.assertIn('id="register-password-confirm"', template_source)
        self.assertIn("계정 만들기", template_source)

    def test_script_posts_registration_and_auto_enters_app(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function setAuthMode(mode) {", script_source)
        self.assertIn("registerForm.addEventListener('submit'", script_source)
        self.assertIn("apiUrl('auth/register')", script_source)
        self.assertIn("password_confirm", script_source)
        self.assertIn("비밀번호가 일치하지 않습니다.", script_source)
        self.assertIn("await enterAuthenticatedApp(data.user || null);", script_source)

    def test_script_treats_auth_me_network_failure_as_logged_out(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("async function fetchCurrentUser() {", script_source)
        self.assertIn("catch (err)", script_source)
        self.assertIn("return null;", script_source)

    def test_script_stops_background_polling_when_ocr_jobs_returns_401(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function stopBackgroundOcrPolling() {", script_source)
        self.assertIn("if (res.status === 401) {", script_source)
        self.assertIn("stopBackgroundOcrPolling();", script_source)
        self.assertIn("showLogin();", script_source)

    def test_script_resets_auth_state_when_kb_list_returns_401(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("async function loadKBs() {", script_source)
        self.assertIn(
            "if (res.status === 401) {\n"
            "        stopBackgroundOcrPolling();\n"
            "        stopAllUploadPolling();\n"
            "        applyCurrentUser(null);\n"
            "        resetChatStateForLoggedOutUser();\n"
            "        showLogin();",
            script_source,
        )

    def test_logout_clears_auth_scoped_polling_state(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")
        self.assertIn("function stopAllUploadPolling() {", script_source)
        self.assertIn("activeUploadPolls.forEach", script_source)
        self.assertIn("stopAllUploadPolling();", script_source)
        self.assertIn("stopBackgroundOcrPolling();", script_source)

    def test_frontend_exposes_wiki_panel_controls(self):
        html_source = (ROOT / "src" / "templates" / "index.html").read_text(encoding="utf-8")
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn('id="wiki-panel-btn"', html_source)
        self.assertIn('id="wiki-panel"', html_source)
        self.assertIn("async function loadWikiPanel()", script_source)
        self.assertIn("apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki`)", script_source)
        self.assertIn("apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/export`)", script_source)
        self.assertIn("apiUrl(`ops/wiki-lint?kb_name=${encodeURIComponent(currentKB)}`)", script_source)
        self.assertIn("wikiLintBtn.hidden = !(currentUser && currentUser.role === 'admin');", script_source)

    def test_frontend_exposes_answer_memory_save_controls(self):
        script_source = (ROOT / "src" / "static" / "script.js").read_text(encoding="utf-8")

        self.assertIn("response.headers.get('X-Query-Id')", script_source)
        self.assertIn("attachWikiAnswerActions", script_source)
        self.assertIn("Wiki 저장", script_source)
        self.assertIn("근거 부족 신고", script_source)
        self.assertIn("answers/${encodeURIComponent(queryId)}/save-to-wiki", script_source)
        self.assertIn("answers/${encodeURIComponent(queryId)}/report-citation-issue", script_source)
        self.assertIn("ontology_job_id", script_source)
        self.assertIn("Ontology 재점검 접수됨", script_source)
        self.assertIn("wiki/answers?limit=200", script_source)
        self.assertIn("wiki/quality", script_source)
        self.assertIn("quality_flags_json", script_source)
        self.assertIn("reused_count", script_source)
        self.assertIn("wiki-status-filter", script_source)
        self.assertIn("wiki-review-action", script_source)
        self.assertIn("wiki/answers/${encodeURIComponent(savedAnswerId)}/compile", script_source)
        self.assertIn("wiki/lint/${encodeURIComponent(findingId)}/resolve", script_source)
        self.assertIn("wiki/answers?limit=200&status=", script_source)
        self.assertIn("wiki/page-candidates", script_source)
        self.assertIn("wiki/build-pages", script_source)
        self.assertIn("wiki/pages/${encodeURIComponent(slug)}/publish", script_source)
        self.assertIn("wiki/pages/${encodeURIComponent(slug)}/archive", script_source)
        self.assertIn("wiki-tab-btn", script_source)
        self.assertIn("const WIKI_PAGE_WORKFLOW_UI_ENABLED = false;", script_source)
        self.assertIn("let wikiActiveTab = 'Answers';", script_source)
        self.assertIn("function wikiTabNames()", script_source)
        self.assertIn("Candidates", script_source)
        self.assertIn("Pages", script_source)
        self.assertIn("Facts", script_source)
        self.assertIn("Graph", script_source)


if __name__ == "__main__":
    unittest.main()
