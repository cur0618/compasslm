const CHAT_STATE_STORAGE_KEY_PREFIX = 'compasslm.chat_state.v1';
const MAX_PERSISTED_MESSAGES_PER_KB = 200;
const TERMINAL_UPLOAD_STATUSES = new Set(['success', 'error', 'timeout', 'not_found']);
const WIKI_PAGE_WORKFLOW_UI_ENABLED = false;
const activeUploadPolls = new Map();

let currentKB = 'default';
const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const uploadTrigger = document.getElementById('upload-trigger');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const uploadDocRoleSelect = document.getElementById('upload-doc-role');
const kbListElement = document.getElementById('kb-list');
const createKbBtn = document.getElementById('create-kb-btn');
const headerTitle = document.getElementById('current-kb-name');
const adminModeBtn = document.getElementById('admin-mode-btn');
const ocrStatusWidget = document.getElementById('ocr-status-widget');
const ocrStatusTrigger = document.getElementById('ocr-status-trigger');
const ocrStatusPopover = document.getElementById('ocr-status-popover');
const authOverlay = document.getElementById('auth-overlay');
const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
const authModeLoginBtn = document.getElementById('auth-mode-login');
const authModeRegisterBtn = document.getElementById('auth-mode-register');
const loginIdInput = document.getElementById('login-id');
const loginPasswordInput = document.getElementById('login-password');
const registerIdInput = document.getElementById('register-id');
const registerDisplayNameInput = document.getElementById('register-display-name');
const registerPasswordInput = document.getElementById('register-password');
const registerPasswordConfirmInput = document.getElementById('register-password-confirm');
const loginError = document.getElementById('login-error');
const authUserBadge = document.getElementById('auth-user-badge');
const logoutBtn = document.getElementById('logout-btn');
const wikiPanelBtn = document.getElementById('wiki-panel-btn');
const wikiPanel = document.getElementById('wiki-panel');
const wikiPanelClose = document.getElementById('wiki-panel-close');
const wikiPanelKb = document.getElementById('wiki-panel-kb');
const wikiPanelStatus = document.getElementById('wiki-panel-status');
const wikiPageList = document.getElementById('wiki-page-list');
const wikiRefreshBtn = document.getElementById('wiki-refresh-btn');
const wikiExportBtn = document.getElementById('wiki-export-btn');
const wikiLintBtn = document.getElementById('wiki-lint-btn');
const wikiExportOutput = document.getElementById('wiki-export-output');
let adminModeEnabled = false;
let ocrPollTimerId = 0;
let ocrStatusPopoverOpen = false;
let currentUser = null;
let chatStateCache = emptyChatState();
let kbRecordByName = new Map();
let wikiStatusFilter = '';
let wikiActiveTab = 'Answers';
let ontologyRebuildPollTimerId = 0;

const BASE_PATH = (() => {
    const p = window.location.pathname || '/';
    return p.endsWith('/') ? p : `${p}/`;
})();

function apiUrl(path) {
    const rel = (path || '').replace(/^\/+/, '');
    return new URL(rel, `${window.location.origin}${BASE_PATH}`).toString();
}

function showLogin(message = '') {
    if (authOverlay) authOverlay.hidden = false;
    if (loginError) loginError.textContent = message;
    setTimeout(() => {
        if (registerForm && !registerForm.hidden) {
            if (registerIdInput) registerIdInput.focus();
            return;
        }
        if (loginIdInput) loginIdInput.focus();
    }, 0);
}

function hideLogin() {
    if (authOverlay) authOverlay.hidden = true;
    if (loginError) loginError.textContent = '';
    if (loginPasswordInput) loginPasswordInput.value = '';
    if (registerPasswordInput) registerPasswordInput.value = '';
    if (registerPasswordConfirmInput) registerPasswordConfirmInput.value = '';
}

function setAuthMode(mode) {
    const nextMode = mode === 'register' ? 'register' : 'login';
    const isRegister = nextMode === 'register';
    if (loginForm) loginForm.hidden = isRegister;
    if (registerForm) registerForm.hidden = !isRegister;
    if (authModeLoginBtn) {
        authModeLoginBtn.classList.toggle('active', !isRegister);
        authModeLoginBtn.setAttribute('aria-selected', isRegister ? 'false' : 'true');
    }
    if (authModeRegisterBtn) {
        authModeRegisterBtn.classList.toggle('active', isRegister);
        authModeRegisterBtn.setAttribute('aria-selected', isRegister ? 'true' : 'false');
    }
    if (loginError) loginError.textContent = '';
    setTimeout(() => {
        if (isRegister && registerIdInput) {
            registerIdInput.focus();
        } else if (loginIdInput) {
            loginIdInput.focus();
        }
    }, 0);
}

async function fetchCurrentUser() {
    try {
        const res = await fetch(apiUrl('auth/me'));
        if (!res.ok) return null;
        const data = await res.json();
        return data && data.user ? data.user : null;
    } catch (err) {
        return null;
    }
}

function applyCurrentUser(user) {
    currentUser = user || null;
    if (authUserBadge) {
        authUserBadge.textContent = currentUser ? (currentUser.display_name || currentUser.login_id || '') : '';
    }
    if (adminModeBtn) {
        adminModeBtn.hidden = !(currentUser && currentUser.role === 'admin');
    }
    if (wikiLintBtn) {
        wikiLintBtn.hidden = !(currentUser && currentUser.role === 'admin');
    }
}

function setWikiPanelStatus(message = '', isError = false) {
    if (!wikiPanelStatus) return;
    wikiPanelStatus.textContent = message;
    wikiPanelStatus.classList.toggle('error', Boolean(isError));
}

function wikiTabNames() {
    return WIKI_PAGE_WORKFLOW_UI_ENABLED
        ? ['Pages', 'Answers', 'Review', 'Candidates', 'Facts', 'Graph']
        : ['Answers', 'Review', 'Facts', 'Graph'];
}

function renderWikiTabBar() {
    const tabBar = document.createElement('div');
    tabBar.className = 'wiki-tab-bar';
    const wikiTabs = wikiTabNames();
    if (!wikiTabs.includes(wikiActiveTab)) {
        wikiActiveTab = 'Answers';
    }
    wikiTabs.forEach(tabName => {
        const tabBtn = document.createElement('button');
        tabBtn.type = 'button';
        tabBtn.className = `wiki-tab-btn${wikiActiveTab === tabName ? ' active' : ''}`;
        tabBtn.textContent = tabName;
        tabBtn.addEventListener('click', () => {
            wikiActiveTab = tabName;
            if (tabName !== 'Facts') stopOntologyRebuildPolling();
            loadWikiPanel();
        });
        tabBar.appendChild(tabBtn);
    });
    return tabBar;
}

function renderWikiPages(pages = [], answers = []) {
    if (!wikiPageList) return;
    wikiPageList.innerHTML = '';
    wikiPageList.appendChild(renderWikiTabBar());
    const filterBar = document.createElement('div');
    filterBar.className = 'wiki-status-filter';
    ['', 'published', 'needs_review', 'reported', 'archived'].forEach(status => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `wiki-status-filter-btn${wikiStatusFilter === status ? ' active' : ''}`;
        btn.textContent = status || 'all';
        btn.addEventListener('click', () => {
            wikiStatusFilter = status;
            loadWikiPanel();
        });
        filterBar.appendChild(btn);
    });
    wikiPageList.appendChild(filterBar);
    const safePages = WIKI_PAGE_WORKFLOW_UI_ENABLED && wikiActiveTab === 'Pages' && Array.isArray(pages) ? pages : [];
    let safeAnswers = Array.isArray(answers) ? answers : [];
    if (wikiActiveTab === 'Review') {
        safeAnswers = safeAnswers.filter(answer => ['needs_review', 'reported'].includes(String(answer.status || '')));
    }
    if (safePages.length === 0 && safeAnswers.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = '아직 wiki page나 answer memory가 없습니다.';
        wikiPageList.appendChild(empty);
        return;
    }
    safeAnswers.forEach(answer => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'wiki-page-item';
        const title = String(answer.question_text || answer.answer_summary || 'Saved answer');
        const status = String(answer.status || 'draft');
        let qualityFlags = [];
        try {
            const parsedFlags = JSON.parse(String(answer.quality_flags_json || '[]'));
            qualityFlags = Array.isArray(parsedFlags) ? parsedFlags : [];
        } catch (_) {
            qualityFlags = [];
        }
        const reusedCount = Number(answer.reused_count || 0);
        const updatedAt = Number(answer.updated_at || answer.created_at || 0);
        const qualityMeta = qualityFlags.length ? ` · flags: ${qualityFlags.join(', ')}` : '';
        const reuseMeta = reusedCount ? ` · reused ${reusedCount}` : '';
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(title)}</span>` +
            `<span class="wiki-page-meta">answer memory · ${escapeHtml(status)}${escapeHtml(qualityMeta)}${escapeHtml(reuseMeta)}${updatedAt ? ` · ${escapeHtml(new Date(updatedAt * 1000).toLocaleDateString())}` : ''}</span>`
        );
        if (currentUser && currentUser.role === 'admin') {
            const savedAnswerId = String(answer.saved_answer_id || '');
            const actions = document.createElement('div');
            actions.className = 'wiki-review-action';
            [
                ['published', '승격'],
                ['needs_review', '보류'],
                ['archived', '보관']
            ].forEach(([nextStatus, label]) => {
                const actionBtn = document.createElement('button');
                actionBtn.type = 'button';
                actionBtn.className = 'wiki-review-action-btn';
                actionBtn.textContent = label;
                actionBtn.addEventListener('click', async (event) => {
                    event.stopPropagation();
                    await submitWikiReviewStatus(savedAnswerId, nextStatus);
                    await loadWikiPanel();
                });
                actions.appendChild(actionBtn);
            });
            if (WIKI_PAGE_WORKFLOW_UI_ENABLED) {
                const compileBtn = document.createElement('button');
                compileBtn.type = 'button';
                compileBtn.className = 'wiki-review-action-btn';
                compileBtn.textContent = '구조화';
                compileBtn.addEventListener('click', async (event) => {
                    event.stopPropagation();
                    await compileWikiAnswer(savedAnswerId);
                    await loadWikiPanel();
                });
                actions.appendChild(compileBtn);
            }
            item.appendChild(actions);
        }
        wikiPageList.appendChild(item);
    });
    safePages.forEach(page => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'wiki-page-item';
        const title = String(page.title || page.slug || 'Wiki page');
        const pageType = String(page.page_type || 'note');
        const updatedAt = Number(page.updated_at || 0);
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(title)}</span>` +
            `<span class="wiki-page-meta">${escapeHtml(pageType)}${updatedAt ? ` · ${escapeHtml(new Date(updatedAt * 1000).toLocaleDateString())}` : ''}</span>`
        );
        item.addEventListener('click', async () => {
            await loadWikiPage(String(page.slug || ''));
        });
        if (currentUser && currentUser.role === 'admin') {
            const slug = String(page.slug || '');
            const actions = document.createElement('div');
            actions.className = 'wiki-review-action';
            [
                ['publish', '승격'],
                ['archive', '보관']
            ].forEach(([action, label]) => {
                const actionBtn = document.createElement('button');
                actionBtn.type = 'button';
                actionBtn.className = 'wiki-review-action-btn';
                actionBtn.textContent = label;
                actionBtn.addEventListener('click', async (event) => {
                    event.stopPropagation();
                    await submitWikiPageStatus(slug, action);
                    await loadWikiPanel();
                });
                actions.appendChild(actionBtn);
            });
            item.appendChild(actions);
        }
        wikiPageList.appendChild(item);
    });
}

function renderWikiCandidates(candidates = []) {
    if (!wikiPageList) return;
    wikiPageList.innerHTML = '';
    wikiPageList.appendChild(renderWikiTabBar());
    const safeCandidates = Array.isArray(candidates) ? candidates : [];
    if (currentUser && currentUser.role === 'admin') {
        const buildBtn = document.createElement('button');
        buildBtn.type = 'button';
        buildBtn.className = 'wiki-review-action-btn';
        buildBtn.textContent = '페이지 생성';
        buildBtn.addEventListener('click', async () => {
            await buildWikiPages();
            await loadWikiPanel();
        });
        wikiPageList.appendChild(buildBtn);
    }
    if (safeCandidates.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = 'page candidate가 없습니다.';
        wikiPageList.appendChild(empty);
        return;
    }
    safeCandidates.forEach(candidate => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = 'wiki-page-item';
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(String(candidate.title || candidate.slug || 'Candidate'))}</span>` +
            `<span class="wiki-page-meta">${escapeHtml(String(candidate.page_type || 'page'))} · ${escapeHtml(String(candidate.status || 'draft'))} · sources ${Number(candidate.source_count || 0)} · claims ${Number(candidate.claim_count || 0)}</span>`
        );
        wikiPageList.appendChild(item);
    });
}

function normalizeOntologyRebuildStatus(job = {}) {
    const status = String(job.status || '').trim().toLowerCase();
    if (job.cancel_requested && !['success', 'error', 'cancelled'].includes(status)) {
        return 'cancel_requested';
    }
    return status === 'processing' ? 'running' : status === 'success' ? 'completed' : status;
}

function stopOntologyRebuildPolling() {
    if (!ontologyRebuildPollTimerId) return;
    window.clearTimeout(ontologyRebuildPollTimerId);
    ontologyRebuildPollTimerId = 0;
}

function scheduleOntologyRebuildPolling(jobs = []) {
    stopOntologyRebuildPolling();
    const hasActiveJob = (Array.isArray(jobs) ? jobs : []).some(job => {
        const status = normalizeOntologyRebuildStatus(job);
        return ['queued', 'running', 'cancel_requested'].includes(status);
    });
    if (!hasActiveJob || !wikiPanel || wikiPanel.hidden || wikiActiveTab !== 'Facts') return;
    ontologyRebuildPollTimerId = window.setTimeout(() => {
        ontologyRebuildPollTimerId = 0;
        loadWikiPanel();
    }, document.visibilityState === 'hidden' ? 10000 : 2500);
}

function formatOntologyRebuildJobMeta(job = {}) {
    const status = normalizeOntologyRebuildStatus(job);
    const processed = Math.max(0, Number(job.chunks_processed || 0));
    const total = Math.max(0, Number(job.chunks_total || 0));
    const percent = Math.max(0, Math.min(100, Number(job.progress_percent || 0)));
    const errors = Math.max(0, Number(job.ontology_extraction_errors || 0));
    const elapsed = formatDurationSeconds(job.processing_elapsed_seconds || job.elapsed_seconds);
    const parts = [status || 'unknown', `${Math.round(percent)}%`];
    if (total > 0) parts.push(`${processed}/${total} chunks`);
    if (errors > 0) parts.push(`LLM 오류 ${errors}`);
    if (elapsed) parts.push(elapsed);
    return parts.join(' · ');
}

function renderOntologyRebuildJobs(jobs = []) {
    if (!wikiPageList || !(currentUser && currentUser.role === 'admin')) return;
    const section = document.createElement('section');
    section.className = 'ontology-rebuild-section';

    const header = document.createElement('div');
    header.className = 'ontology-rebuild-header';
    const title = document.createElement('div');
    title.className = 'wiki-page-title';
    title.textContent = 'Ontology rebuild';
    const startBtn = document.createElement('button');
    startBtn.type = 'button';
    startBtn.className = 'wiki-review-action-btn';
    startBtn.textContent = 'LLM rebuild 시작';
    startBtn.addEventListener('click', async () => {
        startBtn.disabled = true;
        try {
            await startOntologyRebuild();
            await loadWikiPanel();
        } catch (err) {
            setWikiPanelStatus(err.message || 'ontology rebuild를 시작하지 못했습니다.', true);
        } finally {
            startBtn.disabled = false;
        }
    });
    header.appendChild(title);
    header.appendChild(startBtn);
    section.appendChild(header);

    const safeJobs = Array.isArray(jobs) ? jobs : [];
    if (safeJobs.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = '최근 rebuild 작업이 없습니다.';
        section.appendChild(empty);
    }
    safeJobs.slice(0, 8).forEach(job => {
        const row = document.createElement('div');
        row.className = 'ontology-rebuild-job';
        const jobId = String(job.job_id || '');
        const status = normalizeOntologyRebuildStatus(job);
        row.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(String(job.message || 'Ontology rebuild'))}</span>` +
            `<span class="wiki-page-meta">${escapeHtml(formatOntologyRebuildJobMeta(job))}</span>`
        );
        const actions = document.createElement('div');
        actions.className = 'wiki-review-action';
        if (['queued', 'running', 'cancel_requested'].includes(status)) {
            const cancelBtn = document.createElement('button');
            cancelBtn.type = 'button';
            cancelBtn.className = 'wiki-review-action-btn';
            cancelBtn.textContent = status === 'cancel_requested' ? '취소 요청됨' : '취소';
            cancelBtn.disabled = status === 'cancel_requested';
            cancelBtn.addEventListener('click', async () => {
                await cancelOntologyRebuildJob(jobId);
                await loadWikiPanel();
            });
            actions.appendChild(cancelBtn);
        }
        if (['completed', 'error', 'cancelled'].includes(status)) {
            const retryBtn = document.createElement('button');
            retryBtn.type = 'button';
            retryBtn.className = 'wiki-review-action-btn';
            retryBtn.textContent = '재시도';
            retryBtn.addEventListener('click', async () => {
                await retryOntologyRebuildJob(jobId);
                await loadWikiPanel();
            });
            actions.appendChild(retryBtn);
        }
        if (actions.childElementCount > 0) row.appendChild(actions);
        section.appendChild(row);
    });
    wikiPageList.appendChild(section);
}

function renderOntologyFacts(facts = [], rebuildJobs = []) {
    if (!wikiPageList) return;
    wikiPageList.innerHTML = '';
    wikiPageList.appendChild(renderWikiTabBar());
    renderOntologyRebuildJobs(rebuildJobs);
    const safeFacts = Array.isArray(facts) ? facts : [];
    if (safeFacts.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = 'ontology fact가 없습니다.';
        wikiPageList.appendChild(empty);
        return;
    }
    safeFacts.forEach(fact => {
        const item = document.createElement('div');
        item.className = 'wiki-page-item fact-row';
        item.setAttribute('role', 'button');
        item.tabIndex = 0;
        const factId = String(fact.fact_id || '');
        const title = `${String(fact.subject || '')} --${String(fact.predicate || '')}--> ${String(fact.object_entity || fact.object_value || '')}`;
        const body = document.createElement('div');
        body.className = 'wiki-page-body';
        body.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(title)}</span>` +
            `<span class="wiki-page-meta">fact · ${escapeHtml(String(fact.status || 'active'))} · confidence ${Number(fact.confidence || 0).toFixed(2)}</span>`
        );
        item.appendChild(body);
        const openDetail = async () => {
            await loadOntologyFactDetail(factId);
        };
        item.addEventListener('click', openDetail);
        item.addEventListener('keydown', async (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            await openDetail();
        });
        if (currentUser && currentUser.role === 'admin') {
            const actions = document.createElement('div');
            actions.className = 'wiki-review-action';
            [
                ['publish', '승격'],
                ['needs-review', '보류'],
                ['archive', '보관']
            ].forEach(([action, label]) => {
                const actionBtn = document.createElement('button');
                actionBtn.type = 'button';
                actionBtn.className = 'wiki-review-action-btn';
                actionBtn.textContent = label;
                actionBtn.addEventListener('click', async (event) => {
                    event.stopPropagation();
                    await submitOntologyFactStatus(factId, action);
                    await loadWikiPanel();
                });
                actions.appendChild(actionBtn);
            });
            item.appendChild(actions);
        }
        wikiPageList.appendChild(item);
    });
}

function renderOntologyFactDetail(fact = {}) {
    if (!wikiPageList) return;
    wikiPageList.innerHTML = '';
    wikiPageList.appendChild(renderWikiTabBar());
    const backBtn = document.createElement('button');
    backBtn.type = 'button';
    backBtn.className = 'wiki-review-action-btn';
    backBtn.textContent = '목록';
    backBtn.addEventListener('click', async () => {
        await loadWikiPanel();
    });
    wikiPageList.appendChild(backBtn);

    const title = `${String(fact.subject || '')} --${String(fact.predicate || '')}--> ${String(fact.object_entity || fact.object_value || '')}`;
    const header = document.createElement('div');
    header.className = 'wiki-page-item';
    header.innerHTML = (
        `<span class="wiki-page-title">${escapeHtml(title)}</span>` +
        `<span class="wiki-page-meta">fact · ${escapeHtml(String(fact.status || 'active'))} · confidence ${Number(fact.confidence || 0).toFixed(2)}</span>`
    );
    wikiPageList.appendChild(header);

    const sources = Array.isArray(fact.sources) ? fact.sources : [];
    if (sources.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = '연결된 source/evidence가 없습니다.';
        wikiPageList.appendChild(empty);
        return;
    }
    sources.forEach(source => {
        const item = document.createElement('div');
        item.className = 'wiki-page-item';
        const sourceLabel = String(source.source_ref || source.source_path || `chunk ${source.chunk_id || ''}`);
        const evidence = String(source.evidence_quote || '').trim();
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(sourceLabel)}</span>` +
            `<span class="wiki-page-meta">chunk ${Number(source.chunk_id || 0)} · page ${Number(source.page_no || 0)}</span>` +
            `<span class="wiki-page-meta">${escapeHtml(evidence || 'evidence_quote 없음')}</span>`
        );
        wikiPageList.appendChild(item);
    });
    renderOntologyFactHistory(fact.history);
}

function renderOntologyFactHistory(history = []) {
    if (!wikiPageList) return;
    const safeHistory = Array.isArray(history) ? history : [];
    const heading = document.createElement('div');
    heading.className = 'wiki-page-item';
    heading.innerHTML = (
        `<span class="wiki-page-title">변경 이력</span>` +
        `<span class="wiki-page-meta">${safeHistory.length} events</span>`
    );
    wikiPageList.appendChild(heading);
    if (safeHistory.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'wiki-empty';
        empty.textContent = 'confidence/status 변경 이력이 없습니다.';
        wikiPageList.appendChild(empty);
        return;
    }
    safeHistory.slice(0, 20).forEach(row => {
        const item = document.createElement('div');
        item.className = 'wiki-page-item';
        const previousStatus = String(row.previous_status || '');
        const newStatus = String(row.new_status || '');
        const previousConfidence = Number(row.previous_confidence || 0).toFixed(2);
        const newConfidence = Number(row.new_confidence || 0).toFixed(2);
        const source = String(row.source || row.signal || 'history');
        const createdAt = Number(row.created_at || 0);
        const dateLabel = createdAt ? new Date(createdAt * 1000).toLocaleString() : '';
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(previousStatus)} → ${escapeHtml(newStatus)}</span>` +
            `<span class="wiki-page-meta">confidence ${escapeHtml(previousConfidence)} → ${escapeHtml(newConfidence)} · ${escapeHtml(source)}${dateLabel ? ` · ${escapeHtml(dateLabel)}` : ''}</span>`
        );
        wikiPageList.appendChild(item);
    });
}

function renderOntologyGraphOverview(overview = {}) {
    if (!wikiPageList) return;
    wikiPageList.innerHTML = '';
    wikiPageList.appendChild(renderWikiTabBar());
    const rows = [
        ['Entities', Number(overview.entity_count || 0)],
        ['Facts', Number(overview.fact_count || 0)],
        ['Relations', Number(overview.relation_count || 0)]
    ];
    rows.forEach(([label, value]) => {
        const item = document.createElement('div');
        item.className = 'wiki-page-item';
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(label)}</span>` +
            `<span class="wiki-page-meta">${Number(value)}</span>`
        );
        wikiPageList.appendChild(item);
    });
    const topRelations = Array.isArray(overview.top_relations) ? overview.top_relations : [];
    topRelations.slice(0, 12).forEach(rel => {
        const item = document.createElement('div');
        item.className = 'wiki-page-item';
        item.innerHTML = (
            `<span class="wiki-page-title">${escapeHtml(String(rel.predicate || 'relation'))}</span>` +
            `<span class="wiki-page-meta">count ${Number(rel.count || 0)}</span>`
        );
        wikiPageList.appendChild(item);
    });
}

async function loadWikiPage(slug) {
    if (!slug) return;
    setWikiPanelStatus('불러오는 중...');
    if (wikiExportOutput) wikiExportOutput.hidden = true;
    try {
        const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/pages/${encodeURIComponent(slug)}`));
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'wiki page를 불러오지 못했습니다.');
        const page = data.page || {};
        renderWikiPages([page]);
        setWikiPanelStatus(String(page.slug || slug));
    } catch (err) {
        setWikiPanelStatus(err.message || 'wiki page를 불러오지 못했습니다.', true);
    }
}

async function loadOntologyFactDetail(factId) {
    if (!factId) return;
    setWikiPanelStatus('fact 근거를 불러오는 중...');
    try {
        const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/facts/${encodeURIComponent(factId)}`));
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'ontology fact 상세를 불러오지 못했습니다.');
        const fact = data.fact || {};
        renderOntologyFactDetail(fact);
        setWikiPanelStatus(`fact ${factId}`);
    } catch (err) {
        setWikiPanelStatus(err.message || 'ontology fact 상세를 불러오지 못했습니다.', true);
    }
}

async function startOntologyRebuild() {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/rebuild?include_llm=1`), {
        method: 'POST',
        credentials: 'same-origin'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.queued) {
        throw new Error(data.message || 'ontology rebuild를 시작하지 못했습니다.');
    }
    return data.job;
}

async function cancelOntologyRebuildJob(jobId) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/rebuild/jobs/${encodeURIComponent(jobId)}/cancel`), {
        method: 'POST',
        credentials: 'same-origin'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'ontology rebuild 취소 요청에 실패했습니다.');
    }
    return data.job;
}

async function retryOntologyRebuildJob(jobId) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/rebuild/jobs/${encodeURIComponent(jobId)}/retry`), {
        method: 'POST',
        credentials: 'same-origin'
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.queued) {
        throw new Error(data.message || 'ontology rebuild 재시도에 실패했습니다.');
    }
    return data.job;
}

function formatWikiSpaceSubtitle(space = {}, summary = {}) {
    const displayName = String(space.display_name || currentKB || '지침서 공간');
    const guideCount = Number(summary.guide_file_count || 0);
    const casebookCount = Number(summary.casebook_file_count || 0);
    return `${displayName} · 지침서 ${guideCount} · 사례집 ${casebookCount}`;
}

async function loadWikiPanel() {
    if (!wikiPanel) return;
    if (!WIKI_PAGE_WORKFLOW_UI_ENABLED && ['Pages', 'Candidates'].includes(wikiActiveTab)) {
        wikiActiveTab = 'Answers';
    }
    if (wikiPanelKb) wikiPanelKb.textContent = currentKB;
    if (wikiExportOutput) wikiExportOutput.hidden = true;
    setWikiPanelStatus('불러오는 중...');
    try {
        if (wikiActiveTab === 'Facts') {
            const factsRequest = fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/facts?limit=200&status=${encodeURIComponent(wikiStatusFilter)}`));
            const jobsRequest = currentUser && currentUser.role === 'admin'
                ? fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/rebuild/jobs?include_terminal=true`))
                : Promise.resolve(null);
            const [res, jobsRes] = await Promise.all([factsRequest, jobsRequest]);
            const data = await res.json();
            if (!res.ok) throw new Error(data.message || 'ontology fact 목록을 불러오지 못했습니다.');
            const jobsData = jobsRes ? await jobsRes.json().catch(() => ({})) : {};
            if (jobsRes && !jobsRes.ok) throw new Error(jobsData.message || 'ontology rebuild 작업을 불러오지 못했습니다.');
            const facts = Array.isArray(data.facts) ? data.facts : [];
            const jobs = Array.isArray(jobsData.jobs) ? jobsData.jobs : [];
            renderOntologyFacts(facts, jobs);
            scheduleOntologyRebuildPolling(jobs);
            setWikiPanelStatus(`${facts.length} facts${jobs.length ? ` · ${jobs.length} rebuild jobs` : ''}`);
            return;
        }
        stopOntologyRebuildPolling();
        if (wikiActiveTab === 'Graph') {
            const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/ontology/overview`));
            const data = await res.json();
            if (!res.ok) throw new Error(data.message || 'ontology graph를 불러오지 못했습니다.');
            const overview = data.overview || {};
            renderOntologyGraphOverview(overview);
            setWikiPanelStatus(`${Number(overview.entity_count || 0)} entities · ${Number(overview.fact_count || 0)} facts`);
            return;
        }
        if (wikiActiveTab === 'Candidates') {
            const [candidateRes, overviewRes] = await Promise.all([
                fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/page-candidates`)),
                fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/overview`))
            ]);
            const candidateData = await candidateRes.json();
            const overviewData = await overviewRes.json().catch(() => ({}));
            if (!candidateRes.ok) throw new Error(candidateData.message || 'wiki page candidate를 불러오지 못했습니다.');
            const overview = overviewData.overview || {};
            if (wikiPanelKb) wikiPanelKb.textContent = formatWikiSpaceSubtitle(overview.space || {}, overview.space_summary || {});
            const candidates = Array.isArray(candidateData.candidates) ? candidateData.candidates : [];
            renderWikiCandidates(candidates);
            setWikiPanelStatus(`${candidates.length} candidates`);
            return;
        }
        const [pageRes, answerRes, qualitySummary] = await Promise.all([
            fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki`)),
            fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/answers?limit=200&status=${encodeURIComponent(wikiStatusFilter)}`)),
            loadWikiQualitySummary().catch(() => null)
        ]);
        const data = await pageRes.json();
        const answerData = await answerRes.json();
        if (!pageRes.ok) throw new Error(data.message || 'wiki 목록을 불러오지 못했습니다.');
        if (!answerRes.ok) throw new Error(answerData.message || 'answer memory 목록을 불러오지 못했습니다.');
        const pages = Array.isArray(data.pages) ? data.pages : [];
        const answers = Array.isArray(answerData.answers) ? answerData.answers : [];
        if (wikiPanelKb) wikiPanelKb.textContent = formatWikiSpaceSubtitle(data.space || {}, data.space_summary || {});
        renderWikiPages(pages, answers);
        const statusCounts = qualitySummary && typeof qualitySummary.status_counts === 'object'
            ? qualitySummary.status_counts
            : {};
        const reviewCount = Number(statusCounts.needs_review || 0) + Number(statusCounts.reported || 0);
        const guideCount = Number((data.space_summary || {}).guide_file_count || 0);
        const casebookCount = Number((data.space_summary || {}).casebook_file_count || 0);
        setWikiPanelStatus(`${pages.length} pages · ${answers.length} answers · guide ${guideCount} · casebook ${casebookCount}${reviewCount ? ` · ${reviewCount} review` : ''}`);
    } catch (err) {
        stopOntologyRebuildPolling();
        renderWikiPages([]);
        setWikiPanelStatus(err.message || 'wiki 목록을 불러오지 못했습니다.', true);
    }
}

async function buildWikiPages() {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/build-pages`), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'wiki page 생성에 실패했습니다.');
    }
    return data.pages || [];
}

async function submitWikiPageStatus(slug, action) {
    const endpoint = action === 'archive'
        ? `wiki/pages/${encodeURIComponent(slug)}/archive`
        : `wiki/pages/${encodeURIComponent(slug)}/publish`;
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/${endpoint}`), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'wiki page 상태 변경에 실패했습니다.');
    }
    return data.page;
}

async function submitOntologyFactStatus(factId, action) {
    const endpoint = action === 'archive'
        ? `ontology/facts/${encodeURIComponent(factId)}/archive`
        : action === 'needs-review'
            ? `ontology/facts/${encodeURIComponent(factId)}/needs-review`
            : `ontology/facts/${encodeURIComponent(factId)}/publish`;
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/${endpoint}`), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'ontology fact 상태 변경에 실패했습니다.');
    }
    return data.fact;
}

async function submitWikiReviewStatus(savedAnswerId, status) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/answers/${encodeURIComponent(savedAnswerId)}`), {
        method: 'PATCH',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'wiki answer 상태 변경에 실패했습니다.');
    }
    return data.answer;
}

async function compileWikiAnswer(savedAnswerId) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/answers/${encodeURIComponent(savedAnswerId)}/compile`), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'wiki answer 구조화에 실패했습니다.');
    }
    return data.compiled;
}

async function resolveWikiLintFinding(findingId) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/lint/${encodeURIComponent(findingId)}/resolve`), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.status !== 'success') {
        throw new Error(data.message || 'wiki lint finding resolve에 실패했습니다.');
    }
    return data.finding;
}

async function loadWikiQualitySummary() {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/quality`));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.message || 'wiki quality summary를 불러오지 못했습니다.');
    return data.quality || {};
}

async function exportWikiMarkdown() {
    if (!wikiExportOutput) return;
    setWikiPanelStatus('내보내는 중...');
    try {
        const res = await fetch(apiUrl(`kbs/${encodeURIComponent(currentKB)}/wiki/export`));
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'wiki export를 불러오지 못했습니다.');
        const files = data.files && typeof data.files === 'object' ? data.files : {};
        wikiExportOutput.value = Object.entries(files)
            .map(([path, content]) => `# ${path}\n\n${content}`)
            .join('\n\n---\n\n');
        wikiExportOutput.hidden = false;
        setWikiPanelStatus(`${Object.keys(files).length} files`);
    } catch (err) {
        setWikiPanelStatus(err.message || 'wiki export를 불러오지 못했습니다.', true);
    }
}

async function runWikiLint() {
    setWikiPanelStatus('점검 중...');
    if (wikiExportOutput) wikiExportOutput.hidden = true;
    try {
        const res = await fetch(apiUrl(`ops/wiki-lint?kb_name=${encodeURIComponent(currentKB)}`));
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'wiki 점검에 실패했습니다.');
        const findings = Array.isArray(data.findings) ? data.findings : [];
        renderWikiPages(findings.map((finding, index) => ({
            title: `${finding.severity || 'info'} · ${finding.finding_type || 'finding'}`,
            page_type: finding.message || '',
            slug: `lint-${index + 1}`,
            updated_at: Math.floor(Date.now() / 1000)
        })));
        setWikiPanelStatus(`${findings.length} findings`);
    } catch (err) {
        setWikiPanelStatus(err.message || 'wiki 점검에 실패했습니다.', true);
    }
}

function setWikiPanelOpen(open) {
    if (!wikiPanel) return;
    wikiPanel.hidden = !open;
    if (wikiPanelBtn) wikiPanelBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
        loadWikiPanel();
    } else {
        stopOntologyRebuildPolling();
    }
}

function emptyChatState() {
    return { currentKB: 'default', chats: {} };
}

function chatStorageKeyForUser(user = currentUser) {
    if (!user) return null;
    const userKey = String(user.user_id || user.login_id || '').trim();
    if (!userKey) return null;
    return `${CHAT_STATE_STORAGE_KEY_PREFIX}.user.${encodeURIComponent(userKey)}`;
}

function currentChatStorageKey() {
    return currentUser ? chatStorageKeyForUser(currentUser) : null;
}

function resetChatStateForLoggedOutUser() {
    chatStateCache = emptyChatState();
    currentKB = 'default';
    if (headerTitle) headerTitle.textContent = currentKB;
    if (chatContainer) chatContainer.innerHTML = '';
    renderBackgroundOcrStatus([]);
}

async function requireLoggedIn() {
    const user = await fetchCurrentUser();
    if (!user) {
        applyCurrentUser(null);
        resetChatStateForLoggedOutUser();
        showLogin();
        return false;
    }
    applyCurrentUser(user);
    reloadChatStateForCurrentUser();
    hideLogin();
    return true;
}

async function enterAuthenticatedApp(user) {
    applyCurrentUser(user || null);
    reloadChatStateForCurrentUser();
    hideLogin();
    await loadKBs();
    await restoreChat(currentKB);
    resumeAllPendingUploadJobs();
    startBackgroundOcrPolling();
}

function loadInitialKB() {
    return normalizeKBName(chatStateCache.currentKB || 'default');
}

function loadPersistedChatStore() {
    const storageKey = currentChatStorageKey();
    if (!storageKey) {
        return emptyChatState();
    }
    try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) {
            return { currentKB: currentKB || 'default', chats: {} };
        }
        const parsed = JSON.parse(raw);
        return {
            currentKB: typeof parsed?.currentKB === 'string' && parsed.currentKB.trim() ? parsed.currentKB.trim() : (currentKB || 'default'),
            chats: parsed && typeof parsed.chats === 'object' && parsed.chats ? parsed.chats : {}
        };
    } catch (err) {
        return { currentKB: currentKB || 'default', chats: {} };
    }
}

function reloadChatStateForCurrentUser() {
    chatStateCache = loadPersistedChatStore();
    currentKB = normalizeKBName(chatStateCache.currentKB || 'default');
    chatStateCache.currentKB = currentKB;
    if (headerTitle) headerTitle.textContent = currentKB;
}

function normalizeKBName(kbName) {
    const value = String(kbName || '').trim();
    return value || 'default';
}

function normalizeKBRecord(raw = {}) {
    const name = normalizeKBName(raw.display_name || raw.name || raw.kb_name || '');
    return {
        name,
        display_name: name,
        kb_id: String(raw.kb_id || ''),
        internal_kb_id: normalizeKBName(raw.internal_kb_id || name)
    };
}

function rememberKBRecord(raw = {}) {
    const record = normalizeKBRecord(raw);
    kbRecordByName.set(record.name, record);
    return record;
}

function replaceKBRecords(kbNames = [], records = []) {
    kbRecordByName = new Map();
    const normalizedRecords = Array.isArray(records) ? records.map(item => normalizeKBRecord(item)) : [];
    normalizedRecords.forEach(record => {
        kbRecordByName.set(record.name, record);
    });
    (Array.isArray(kbNames) ? kbNames : []).forEach(name => {
        const safeName = normalizeKBName(name);
        if (!kbRecordByName.has(safeName)) {
            kbRecordByName.set(safeName, normalizeKBRecord({ name: safeName, internal_kb_id: safeName }));
        }
    });
}

function kbRecordForName(kbName) {
    const safeKB = normalizeKBName(kbName);
    return kbRecordByName.get(safeKB) || normalizeKBRecord({ name: safeKB, internal_kb_id: safeKB });
}

function chatBucketKeyForKB(kbName) {
    return normalizeKBName(kbRecordForName(kbName).internal_kb_id || kbName);
}

function chatBucketKeysForKB(kbName) {
    const safeKB = normalizeKBName(kbName);
    return [...new Set([chatBucketKeyForKB(safeKB), safeKB])];
}

function migratePersistedChatBucketsForRecords() {
    if (!chatStateCache.chats || typeof chatStateCache.chats !== 'object') return;
    kbRecordByName.forEach(record => {
        const displayKey = normalizeKBName(record.name);
        const bucketKey = normalizeKBName(record.internal_kb_id || displayKey);
        if (displayKey === bucketKey) return;
        if (Array.isArray(chatStateCache.chats[displayKey]) && !Array.isArray(chatStateCache.chats[bucketKey])) {
            chatStateCache.chats[bucketKey] = chatStateCache.chats[displayKey].map(item => normalizePersistedMessage({ ...item, kbName: displayKey }));
            delete chatStateCache.chats[displayKey];
        }
    });
    persistChatState(currentKB);
}

function createMessageId() {
    return `msg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function normalizePersistedMessage(raw = {}) {
    const role = raw.role === 'user' ? 'user' : 'assistant';
    const kind = raw.kind === 'upload' ? 'upload' : 'chat';
    const message = {
        id: String(raw.id || createMessageId()),
        role,
        text: String(raw.text || ''),
        kind,
        kbName: normalizeKBName(raw.kbName || currentKB),
        createdAt: String(raw.createdAt || new Date().toISOString()),
        uploadJobId: '',
        uploadJobVersion: 0,
        uploadStatus: '',
        uploadProgressStage: '',
        fileName: '',
        roleLabel: '',
        conversationMode: String(raw.conversationMode || ''),
        queryId: String(raw.queryId || ''),
        wikiSaved: Boolean(raw.wikiSaved),
        wikiReported: Boolean(raw.wikiReported)
    };
    if (kind === 'upload') {
        message.uploadJobId = String(raw.uploadJobId || '');
        message.uploadJobVersion = Math.max(0, Number(raw.uploadJobVersion) || 0);
        message.uploadStatus = String(raw.uploadStatus || '');
        message.uploadProgressStage = String(raw.uploadProgressStage || '');
        message.fileName = String(raw.fileName || '');
        message.roleLabel = String(raw.roleLabel || '');
    }
    return message;
}

function loadPersistedMessages(kbName) {
    const safeKB = normalizeKBName(kbName);
    const bucketKey = chatBucketKeyForKB(safeKB);
    const rows = chatStateCache.chats && Array.isArray(chatStateCache.chats[bucketKey]) ? chatStateCache.chats[bucketKey] : [];
    return rows.map(item => normalizePersistedMessage({ ...item, kbName: safeKB }));
}

function persistChatState(kbName = currentKB) {
    const storageKey = currentChatStorageKey();
    if (!storageKey) return;
    const safeKB = normalizeKBName(kbName);
    const bucketKey = chatBucketKeyForKB(safeKB);
    if (!chatStateCache.chats || typeof chatStateCache.chats !== 'object') {
        chatStateCache.chats = {};
    }
    if (!Array.isArray(chatStateCache.chats[bucketKey])) {
        chatStateCache.chats[bucketKey] = [];
    }
    chatStateCache.currentKB = normalizeKBName(currentKB || safeKB);
    try {
        localStorage.setItem(storageKey, JSON.stringify(chatStateCache));
    } catch (err) {
        // Ignore storage quota errors and continue with in-memory state.
    }
}

function setPersistedMessagesForKB(kbName, messages) {
    const safeKB = normalizeKBName(kbName);
    const bucketKey = chatBucketKeyForKB(safeKB);
    const normalized = (Array.isArray(messages) ? messages : [])
        .map(item => normalizePersistedMessage({ ...item, kbName: safeKB }))
        .slice(-MAX_PERSISTED_MESSAGES_PER_KB);
    chatStateCache.chats[bucketKey] = normalized;
    persistChatState(safeKB);
    return normalized;
}

function appendPersistedMessage(kbName, message) {
    const safeKB = normalizeKBName(kbName);
    const rows = loadPersistedMessages(safeKB);
    const normalized = normalizePersistedMessage({ ...message, kbName: safeKB });
    rows.push(normalized);
    setPersistedMessagesForKB(safeKB, rows);
    return normalized;
}

function updatePersistedMessage(kbName, messageId, patch = {}) {
    const safeKB = normalizeKBName(kbName);
    const rows = loadPersistedMessages(safeKB);
    const index = rows.findIndex(item => item.id === messageId);
    if (index < 0) return null;
    rows[index] = normalizePersistedMessage({
        ...rows[index],
        ...patch,
        id: messageId,
        kbName: safeKB
    });
    setPersistedMessagesForKB(safeKB, rows);
    return rows[index];
}

function persistActiveKB(kbName) {
    currentKB = normalizeKBName(kbName);
    chatStateCache.currentKB = currentKB;
    persistChatState(currentKB);
}

function renamePersistedChatState(oldKbName, newKbName) {
    const oldKey = normalizeKBName(oldKbName);
    const newKey = normalizeKBName(newKbName);
    if (oldKey === newKey) return;
    const existing = loadPersistedMessages(oldKey);
    const oldBucketKey = chatBucketKeyForKB(oldKey);
    const newBucketKey = chatBucketKeyForKB(newKey);
    if (existing.length > 0) {
        chatStateCache.chats[newBucketKey] = existing.map(item => normalizePersistedMessage({ ...item, kbName: newKey }));
    }
    if (oldBucketKey !== newBucketKey) delete chatStateCache.chats[oldBucketKey];
    delete chatStateCache.chats[oldKey];
    if (newBucketKey !== newKey) delete chatStateCache.chats[newKey];
    if (chatStateCache.currentKB === oldKey) {
        chatStateCache.currentKB = newKey;
    }
    persistChatState(newKey);
}

function removePersistedChatState(kbName) {
    const safeKB = normalizeKBName(kbName);
    chatBucketKeysForKB(safeKB).forEach(key => {
        delete chatStateCache.chats[key];
    });
    if (chatStateCache.currentKB === safeKB) {
        chatStateCache.currentKB = 'default';
    }
    persistChatState(chatStateCache.currentKB || 'default');
}

function isTerminalUploadStatus(status) {
    return TERMINAL_UPLOAD_STATUSES.has(String(status || '').trim().toLowerCase());
}

function renderUploadJobText({ kbName, messageId, fileName, roleLabel, job }) {
    if (!messageId || normalizeKBName(kbName) !== normalizeKBName(currentKB)) return;
    const msgDiv = findMessageDivById(messageId);
    if (!msgDiv) return;
    const textDiv = getMessageTextElement(msgDiv);
    if (!textDiv) return;
    setBubbleText(textDiv, formatUploadStatusMessage(fileName, roleLabel, job), 'assistant');
}

function stopUploadElapsedTimer(jobId) {
    const pollState = activeUploadPolls.get(jobId);
    if (!pollState || !pollState.elapsedTimerId) return;
    clearInterval(pollState.elapsedTimerId);
    pollState.elapsedTimerId = 0;
}

function rememberLatestUploadJob(jobId, job) {
    const pollState = activeUploadPolls.get(jobId);
    if (!pollState) return;
    pollState.latestJob = job ? { ...job } : null;
    pollState.elapsedBaseAtMs = Date.now();
    pollState.renderedElapsedSeconds = Math.max(0, Number(job && job.elapsed_seconds) || 0);
}

function ensureUploadElapsedTimer(jobId) {
    const pollState = activeUploadPolls.get(jobId);
    if (!pollState || pollState.elapsedTimerId) return;
    pollState.elapsedTimerId = setInterval(() => {
        const currentState = activeUploadPolls.get(jobId);
        if (!currentState) {
            clearInterval(pollState.elapsedTimerId);
            return;
        }
        const latestJob = currentState.latestJob;
        if (!latestJob || isTerminalUploadStatus(latestJob.status)) {
            stopUploadElapsedTimer(jobId);
            return;
        }
        const baseElapsedSeconds = Math.max(0, Number(latestJob.elapsed_seconds) || 0);
        const baseAtMs = Math.max(0, Number(currentState.elapsedBaseAtMs) || 0);
        if (baseAtMs <= 0) return;
        const localElapsedSeconds = Math.max(
            baseElapsedSeconds,
            baseElapsedSeconds + Math.floor((Date.now() - baseAtMs) / 1000)
        );
        if (localElapsedSeconds === currentState.renderedElapsedSeconds) return;
        currentState.renderedElapsedSeconds = localElapsedSeconds;
        renderUploadJobText({
            kbName: currentState.kbName,
            messageId: currentState.messageId,
            fileName: currentState.fileName,
            roleLabel: currentState.roleLabel,
            job: { ...latestJob, elapsed_seconds: localElapsedSeconds }
        });
    }, 1000);
}

function applyMessageMetadata(msgDiv, message) {
    if (!msgDiv || !message) return;
    msgDiv.dataset.messageId = message.id || '';
    msgDiv.dataset.kbName = normalizeKBName(message.kbName || currentKB);
    msgDiv.dataset.messageKind = message.kind || 'chat';
    msgDiv.dataset.uploadJobId = message.uploadJobId || '';
    msgDiv.dataset.uploadJobVersion = String(message.uploadJobVersion || 0);
    msgDiv.dataset.uploadStatus = message.uploadStatus || '';
    msgDiv.dataset.uploadProgressStage = message.uploadProgressStage || '';
    msgDiv.dataset.fileName = message.fileName || '';
    msgDiv.dataset.roleLabel = message.roleLabel || '';
    msgDiv.dataset.conversationMode = message.conversationMode || '';
    msgDiv.dataset.queryId = message.queryId || '';
    msgDiv.dataset.wikiSaved = message.wikiSaved ? '1' : '';
    msgDiv.dataset.wikiReported = message.wikiReported ? '1' : '';
    msgDiv.setAttribute('data-conversation-mode', message.conversationMode || '');
}

function findMessageDivById(messageId) {
    if (!messageId) return null;
    return chatContainer.querySelector(`.message[data-message-id="${messageId}"]`);
}

function refreshUserBubbleStates(kbName = currentKB) {
    const safeKB = normalizeKBName(kbName);
    const userMessages = Array.from(chatContainer.querySelectorAll(`.message.user[data-kb-name="${safeKB}"]`));
    userMessages.forEach((node, index) => {
        node.classList.remove('is-latest-user', 'is-history-user');
        if (index === userMessages.length - 1) {
            node.classList.add('is-latest-user');
        } else {
            node.classList.add('is-history-user');
        }
    });
}

function syncRenderedMessage(message) {
    if (!message || normalizeKBName(message.kbName) !== normalizeKBName(currentKB)) return;
    const msgDiv = findMessageDivById(message.id);
    if (!msgDiv) return;
    applyMessageMetadata(msgDiv, message);
    const textDiv = getMessageTextElement(msgDiv);
    if (textDiv) {
        const role = msgDiv.classList.contains('user') ? 'user' : 'assistant';
        setBubbleText(textDiv, message.text || '', role);
    }
    refreshUserBubbleStates(currentKB);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function renderStoredMessages(messages) {
    chatContainer.innerHTML = '';
    messages.forEach(message => {
        renderMessage(message);
    });
    refreshUserBubbleStates(currentKB);
}

// TEMP: 관리자 QA 평가/로그 기능. 배포 시 이 블록을 주석 처리해서 비활성화할 수 있음.
function updateAdminModeButton() {
    if (!adminModeBtn) return;
    adminModeBtn.textContent = `관리자 모드: ${adminModeEnabled ? 'ON' : 'OFF'}`;
    adminModeBtn.classList.toggle('is-on', adminModeEnabled);
    adminModeBtn.setAttribute('aria-pressed', String(adminModeEnabled));
}

if (adminModeBtn) {
    adminModeBtn.addEventListener('click', () => {
        adminModeEnabled = !adminModeEnabled;
        updateAdminModeButton();
    });
}
updateAdminModeButton();

async function submitAnswerFeedback({ kbName, question, answer, isCorrect, expectedAnswer = '' }) {
    const response = await fetch(apiUrl('feedback'), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            kb_name: kbName,
            question,
            answer,
            is_correct: isCorrect,
            expected_answer: expectedAnswer
        })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.status !== 'success') {
        throw new Error(payload.error || payload.message || '평가 로그 저장에 실패했습니다.');
    }
    return payload;
}

async function submitWikiAnswerAction({ kbName, queryId, action }) {
    const endpoint = action === 'report'
        ? `kbs/${encodeURIComponent(kbName)}/answers/${encodeURIComponent(queryId)}/report-citation-issue`
        : `kbs/${encodeURIComponent(kbName)}/answers/${encodeURIComponent(queryId)}/save-to-wiki`;
    const response = await fetch(apiUrl(endpoint), {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.status !== 'success') {
        throw new Error(payload.error || payload.message || 'Wiki memory 저장에 실패했습니다.');
    }
    return payload;
}

function attachWikiAnswerActions(msgDiv, { kbName, queryId }) {
    if (!msgDiv || !queryId) return;
    const content = msgDiv.querySelector('.message-content');
    if (!content) return;
    const bubble = content.querySelector('.bubble');
    if (!bubble || bubble.querySelector('.wiki-answer-actions')) return;

    const panel = document.createElement('div');
    panel.className = 'wiki-answer-actions';
    panel.innerHTML = `
        <button type="button" class="wiki-answer-btn save">Wiki 저장</button>
        <button type="button" class="wiki-answer-btn report">근거 부족 신고</button>
        <span class="wiki-answer-status"></span>
    `;
    bubble.appendChild(panel);

    const saveBtn = panel.querySelector('.wiki-answer-btn.save');
    const reportBtn = panel.querySelector('.wiki-answer-btn.report');
    const statusEl = panel.querySelector('.wiki-answer-status');
    let busy = false;

    const setBusy = (value) => {
        busy = Boolean(value);
        if (saveBtn) saveBtn.disabled = busy || msgDiv.dataset.wikiSaved === '1';
        if (reportBtn) reportBtn.disabled = busy || msgDiv.dataset.wikiReported === '1';
    };

    const applyAction = async (action) => {
        if (busy) return;
        setBusy(true);
        statusEl.classList.remove('error');
        statusEl.textContent = action === 'report' ? '신고 저장 중...' : 'Wiki memory 저장 중...';
        try {
            const actionResult = await submitWikiAnswerAction({ kbName, queryId, action });
            const patch = action === 'report' ? { wikiReported: true } : { wikiSaved: true };
            updatePersistedMessage(kbName, msgDiv.dataset.messageId, patch);
            applyMessageMetadata(msgDiv, normalizePersistedMessage({
                id: msgDiv.dataset.messageId,
                role: 'assistant',
                text: getMessageTextElement(msgDiv) ? getMessageTextElement(msgDiv).textContent : '',
                kind: msgDiv.dataset.messageKind || 'chat',
                kbName,
                queryId,
                wikiSaved: action === 'report' ? msgDiv.dataset.wikiSaved === '1' : true,
                wikiReported: action === 'report' ? true : msgDiv.dataset.wikiReported === '1',
                conversationMode: msgDiv.dataset.conversationMode || ''
            }));
            const ontologyJobId = String((actionResult && actionResult.ontology_job_id) || '');
            statusEl.textContent = action === 'report'
                ? (ontologyJobId ? '근거 검토 저장됨 · Ontology 재점검 접수됨' : '근거 검토 대상으로 저장됨')
                : 'Wiki memory에 저장됨';
            if (wikiPanel && !wikiPanel.hidden) {
                loadWikiPanel();
            }
        } catch (err) {
            statusEl.classList.add('error');
            statusEl.textContent = err.message || '저장 실패';
        } finally {
            setBusy(false);
        }
    };

    saveBtn.addEventListener('click', () => applyAction('save'));
    reportBtn.addEventListener('click', () => applyAction('report'));
    setBusy(false);
}

function attachAdminFeedback(msgDiv, { kbName, question, answer }) {
    if (!msgDiv) return;
    const content = msgDiv.querySelector('.message-content');
    if (!content) return;
    const bubble = content.querySelector('.bubble');
    if (!bubble || !question || !answer) return;

    const panel = document.createElement('div');
    panel.className = 'admin-feedback';
    panel.innerHTML = `
        <div class="admin-feedback-title">관리자 평가</div>
        <div class="admin-feedback-actions">
            <button type="button" class="admin-feedback-btn ok">O</button>
            <button type="button" class="admin-feedback-btn bad">X</button>
        </div>
        <div class="admin-feedback-correction" hidden>
            <textarea placeholder="X 선택 시 올바른 정답을 입력하세요."></textarea>
            <button type="button" class="admin-feedback-btn save">정답 저장</button>
        </div>
        <div class="admin-feedback-status"></div>
    `;
    bubble.appendChild(panel);

    const okBtn = panel.querySelector('.admin-feedback-btn.ok');
    const badBtn = panel.querySelector('.admin-feedback-btn.bad');
    const correctionBox = panel.querySelector('.admin-feedback-correction');
    const correctionInput = panel.querySelector('.admin-feedback-correction textarea');
    const saveBtn = panel.querySelector('.admin-feedback-btn.save');
    const statusEl = panel.querySelector('.admin-feedback-status');

    let submitted = false;
    let submitting = false;

    const setButtonsDisabled = (disabled) => {
        [okBtn, badBtn, saveBtn].forEach(btn => {
            if (btn) btn.disabled = disabled;
        });
    };

    const submitFeedback = async (isCorrect, expectedAnswer = '') => {
        if (submitted || submitting) return;
        submitting = true;
        setButtonsDisabled(true);
        statusEl.classList.remove('error');
        statusEl.textContent = '평가 로그 저장 중...';

        try {
            await submitAnswerFeedback({
                kbName,
                question,
                answer,
                isCorrect,
                expectedAnswer
            });
            submitted = true;
            panel.classList.add('submitted');
            statusEl.textContent = '저장 완료';
        } catch (err) {
            setButtonsDisabled(false);
            statusEl.classList.add('error');
            statusEl.textContent = err.message || '저장 실패';
        } finally {
            submitting = false;
        }
    };

    okBtn.addEventListener('click', () => {
        submitFeedback(true, '');
    });

    badBtn.addEventListener('click', () => {
        correctionBox.hidden = false;
        correctionInput.focus();
        statusEl.classList.remove('error');
        statusEl.textContent = '올바른 정답을 입력하고 저장하세요.';
    });

    saveBtn.addEventListener('click', () => {
        const expectedAnswer = correctionInput.value.trim();
        if (!expectedAnswer) {
            statusEl.classList.add('error');
            statusEl.textContent = '정답을 입력해 주세요.';
            return;
        }
        submitFeedback(false, expectedAnswer);
    });
}

// Context Menu Setup
const ctxMenu = document.createElement('div');
ctxMenu.id = 'context-menu';
ctxMenu.innerHTML = `
    <div class="ctx-item" id="ctx-rename">이름 변경</div>
    <div class="ctx-item delete" id="ctx-delete">삭제</div>
`;
document.body.appendChild(ctxMenu);

let menuTargetKB = null;

// Hide menu on click outside
document.addEventListener('click', () => {
    ctxMenu.style.display = 'none';
});

// Auto resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    if(this.value.trim()) sendBtn.classList.add('active');
    else sendBtn.classList.remove('active');
});

// Chat History Management
async function restoreChat(kbName) {
    const safeKB = normalizeKBName(kbName);
    const cachedMessages = loadPersistedMessages(safeKB);
    if (cachedMessages.length > 0) {
        renderStoredMessages(cachedMessages);
        resumePendingUploadJobsForKB(safeKB);
        return;
    }
    chatContainer.innerHTML = '';
    try {
        const res = await fetch(apiUrl(`chat/history?kb_name=${encodeURIComponent(safeKB)}`), {
            credentials: 'same-origin'
        });
        if (res.ok) {
            const data = await res.json();
            const messages = Array.isArray(data.messages) ? data.messages : [];
            if (messages.length > 0) {
                const persisted = messages.map(m => normalizePersistedMessage({
                    role: m.role === 'user' ? 'user' : 'assistant',
                    text: m.text || '',
                    kbName: safeKB
                }));
                setPersistedMessagesForKB(safeKB, persisted);
                renderStoredMessages(persisted);
                resumePendingUploadJobsForKB(safeKB);
                return;
            }
        }
    } catch (err) {
        // Fallback to default greeting below.
    }
    const greeting = normalizePersistedMessage({
        role: 'assistant',
        text: `🤖 안녕하세요. 궁금하신 내용을 편하게 말씀해 주세요. '${safeKB}' 공간입니다.`,
        kbName: safeKB
    });
    setPersistedMessagesForKB(safeKB, [greeting]);
    renderStoredMessages([greeting]);
}

function saveCurrentChat() {
    persistChatState(currentKB);
}

function getLastAssistantConversationMode(kbName = currentKB) {
    const safeKB = normalizeKBName(kbName);
    const rows = loadPersistedMessages(safeKB);
    for (let i = rows.length - 1; i >= 0; i -= 1) {
        const row = rows[i];
        if (row && row.role === 'assistant' && row.kind === 'chat' && row.conversationMode) {
            return String(row.conversationMode);
        }
    }
    return '';
}

function buildPendingAssistantStatusSteps(userText, kbName = currentKB, conversationMode = '') {
    const raw = String(userText || '').trim();
    if (!raw) return [];
    const steps = [
        { delayMs: 7000, text: '문서를 확인 중입니다. 잠시만 기다려 주세요.' }
    ];
    if (String(conversationMode || '').trim().toLowerCase() === 'document_qa') {
        steps.push({ delayMs: 14000, text: '이용자께서 업로드한 문서를 세세히 보는 중입니다.' });
    }
    return steps;
}

function startPendingAssistantStatus(msgDiv, userText, kbName, conversationMode = '', startedAtMs = Date.now()) {
    if (!msgDiv) return [];
    const steps = buildPendingAssistantStatusSteps(userText, kbName, conversationMode);
    const elapsedMs = Math.max(0, Date.now() - Number(startedAtMs || 0));
    return steps
        .map(step => {
            const remainingMs = Math.max(0, Number(step.delayMs) - elapsedMs);
            if (remainingMs <= 0) return null;
            return window.setTimeout(() => {
                setMessageText(msgDiv, step.text, { kbName });
            }, remainingMs);
        })
        .filter(Boolean);
}

function stopPendingAssistantStatus(timerIds = []) {
    timerIds.forEach(id => window.clearTimeout(id));
}

// Send Message Logic
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;
    const requestKB = normalizeKBName(currentKB);

    appendMessage('user', text, false, { kbName: requestKB });
    userInput.value = '';
    userInput.style.height = 'auto';

    const assistantMsgDiv = appendMessage('assistant', '', true, { kbName: requestKB });
    const textDiv = getMessageTextElement(assistantMsgDiv);
    textDiv.classList.add('typing');
    const pendingStatusStartedAt = Date.now();
    let pendingStatusTimers = startPendingAssistantStatus(
        assistantMsgDiv,
        text,
        requestKB,
        getLastAssistantConversationMode(requestKB),
        pendingStatusStartedAt
    );

    try {
        const response = await fetch(apiUrl('chat'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, kb_name: requestKB })
        });
        if (!response.ok) {
            const errPayload = await response.json().catch(() => ({}));
            throw new Error(errPayload.error || errPayload.message || `HTTP ${response.status}`);
        }
        if (!response.body) {
            throw new Error('응답 스트림이 비어 있습니다.');
        }

        const responseConversationMode = String(response.headers.get('X-Conversation-Mode') || '').trim();
        const responseQueryId = String(response.headers.get('X-Query-Id') || '').trim();
        stopPendingAssistantStatus(pendingStatusTimers);
        pendingStatusTimers = startPendingAssistantStatus(
            assistantMsgDiv,
            text,
            requestKB,
            responseConversationMode,
            pendingStatusStartedAt
        );
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let streamedText = '';
        let receivedFirstChunk = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            if (!receivedFirstChunk) {
                receivedFirstChunk = true;
                stopPendingAssistantStatus(pendingStatusTimers);
            }
            streamedText += chunk;
            setMessageText(assistantMsgDiv, streamedText, { kbName: requestKB, conversationMode: responseConversationMode, queryId: responseQueryId });
        }
        stopPendingAssistantStatus(pendingStatusTimers);
        textDiv.classList.remove('typing');
        const finalAnswer = streamedText.trim();
        setMessageText(assistantMsgDiv, finalAnswer, { kbName: requestKB, conversationMode: responseConversationMode, queryId: responseQueryId });
        if (finalAnswer && responseQueryId) {
            attachWikiAnswerActions(assistantMsgDiv, {
                kbName: requestKB,
                queryId: responseQueryId
            });
        }
        if (adminModeEnabled && finalAnswer) {
            attachAdminFeedback(assistantMsgDiv, {
                kbName: requestKB,
                question: text,
                answer: finalAnswer
            });
        }
        persistChatState(requestKB);
    } catch (error) {
        stopPendingAssistantStatus(pendingStatusTimers);
        setMessageText(assistantMsgDiv, "오류: " + error.message, { kbName: requestKB });
        textDiv.classList.remove('typing');
    }
}

function renderMessage(message, returnDiv = false) {
    const msgDiv = document.createElement('div');
    const role = message.role === 'user' ? 'user' : 'assistant';
    msgDiv.className = `message ${role}`;
    applyMessageMetadata(msgDiv, message);
    const icon = role === 'user' ? '👤' : '🤖';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const iconDiv = document.createElement('div');
    iconDiv.className = 'avatar-icon';
    iconDiv.textContent = icon;

    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'bubble';

    const textDiv = document.createElement('div');
    textDiv.className = 'bubble-text';
    setBubbleText(textDiv, message.text || '', role);

    bubbleDiv.appendChild(textDiv);
    contentDiv.appendChild(iconDiv);
    contentDiv.appendChild(bubbleDiv);
    msgDiv.appendChild(contentDiv);
    chatContainer.appendChild(msgDiv);
    if (role === 'assistant' && message.kind === 'chat' && message.queryId && message.text) {
        attachWikiAnswerActions(msgDiv, {
            kbName: message.kbName || currentKB,
            queryId: message.queryId
        });
    }
    refreshUserBubbleStates(message.kbName || currentKB);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    if (returnDiv) return msgDiv;
}

function appendMessage(role, text, returnDiv = false, options = {}) {
    const safeKB = normalizeKBName(options.kbName || currentKB);
    const message = normalizePersistedMessage({
        ...options,
        role,
        text,
        kbName: safeKB
    });
    if (options.persist !== false) {
        appendPersistedMessage(safeKB, message);
    }
    return renderMessage(message, returnDiv);
}

function getMessageTextElement(msgDiv) {
    return msgDiv ? msgDiv.querySelector('.bubble-text') : null;
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderCitationChip(number, label) {
    const safeNumber = escapeHtml(String(number || '').trim());
    const tooltipLabel = escapeHtml(`[${String(label || '').trim()}]`);
    return (
        `<span class="citation-chip" tabindex="0" role="note" ` +
        `data-citation-number="${safeNumber}" data-citation-label="${tooltipLabel}" ` +
        `aria-label="${tooltipLabel}">[${safeNumber}]</span>`
    );
}

function renderAssistantInlineText(text) {
    const raw = String(text || '');
    const citationPattern = /\[\[CITATION:(\d+)\|([^\]\n]{1,240})\]\]/g;
    let html = '';
    let lastIndex = 0;
    let match;

    while ((match = citationPattern.exec(raw)) !== null) {
        html += escapeHtml(raw.slice(lastIndex, match.index));
        html += renderCitationChip(match[1], match[2]);
        lastIndex = match.index + match[0].length;
    }

    html += escapeHtml(raw.slice(lastIndex));
    return html;
}

function splitAssistantParagraphs(text) {
    return String(text || '')
        .replace(/\r\n/g, '\n')
        .split(/\n{2,}/)
        .map(block => block.trim())
        .filter(Boolean);
}

function isEvidenceParagraph(block) {
    return /^(?:문서\s*)?근거\s*[:：]/.test(String(block || '').trim());
}

function isCompactAnswerLine(line) {
    const normalized = String(line || '').trim();
    if (!normalized || isEvidenceParagraph(normalized)) return false;
    if (/^\d+[\.\)]\s+/.test(normalized)) return true;
    if (/^[가-힣A-Za-z0-9\s]{1,18}\s*[:：]\s+\S/.test(normalized)) return true;
    return false;
}

function renderAssistantEvidence(block) {
    const body = String(block || '').replace(/^(?:문서\s*)?근거\s*[:：]\s*/, '').trim();
    return (
        '<div class="answer-evidence" aria-label="답변 근거">' +
        '<span class="answer-evidence-label">근거</span>' +
        `<span class="answer-evidence-body">${renderAssistantInlineText(body)}</span>` +
        '</div>'
    );
}

function renderAssistantParagraph(block, index) {
    const lines = String(block || '').split('\n').map(line => line.trim()).filter(Boolean);
    if (lines.length > 1 && lines.every(isCompactAnswerLine)) {
        const rows = lines.map(line => {
            const numbered = line.match(/^(\d+[\.\)])\s+(.+)$/);
            const labeled = line.match(/^([가-힣A-Za-z0-9\s]{1,18})\s*[:：]\s+(.+)$/);
            if (numbered) {
                return (
                    '<div class="answer-row">' +
                    `<span class="answer-row-key">${escapeHtml(numbered[1])}</span>` +
                    `<span class="answer-row-value">${renderAssistantInlineText(numbered[2])}</span>` +
                    '</div>'
                );
            }
            if (labeled) {
                return (
                    '<div class="answer-row">' +
                    `<span class="answer-row-key">${escapeHtml(labeled[1].trim())}</span>` +
                    `<span class="answer-row-value">${renderAssistantInlineText(labeled[2])}</span>` +
                    '</div>'
                );
            }
            return `<div class="answer-row"><span class="answer-row-value">${renderAssistantInlineText(line)}</span></div>`;
        }).join('');
        return `<div class="answer-structured-list">${rows}</div>`;
    }
    const body = renderAssistantInlineText(lines.join(' '));
    const emphasisClass = index === 0 ? ' answer-paragraph-lead' : '';
    return `<p class="answer-paragraph${emphasisClass}">${body}</p>`;
}

function renderAssistantText(text) {
    const blocks = splitAssistantParagraphs(text);
    if (!blocks.length) return '';
    return blocks.map((block, index) => (
        isEvidenceParagraph(block)
            ? renderAssistantEvidence(block)
            : renderAssistantParagraph(block, index)
    )).join('');
}

function setBubbleText(textDiv, text, role) {
    if (!textDiv) return;
    if (role === 'assistant') {
        textDiv.innerHTML = renderAssistantText(text);
        return;
    }
    textDiv.textContent = text || '';
}

function setMessageText(msgDiv, text, options = {}) {
    const safeKB = normalizeKBName(options.kbName || (msgDiv && msgDiv.dataset.kbName) || currentKB);
    const messageId = String(options.messageId || (msgDiv && msgDiv.dataset.messageId) || '');
    const patch = {
        text: text || '',
        kind: options.kind || (msgDiv && msgDiv.dataset.messageKind) || 'chat',
        kbName: safeKB,
        uploadJobId: options.uploadJobId !== undefined ? String(options.uploadJobId || '') : (msgDiv && msgDiv.dataset.uploadJobId) || '',
        uploadStatus: options.uploadStatus !== undefined ? String(options.uploadStatus || '') : (msgDiv && msgDiv.dataset.uploadStatus) || '',
        uploadProgressStage: options.uploadProgressStage !== undefined ? String(options.uploadProgressStage || '') : (msgDiv && msgDiv.dataset.uploadProgressStage) || '',
        fileName: options.fileName !== undefined ? String(options.fileName || '') : (msgDiv && msgDiv.dataset.fileName) || '',
        roleLabel: options.roleLabel !== undefined ? String(options.roleLabel || '') : (msgDiv && msgDiv.dataset.roleLabel) || '',
        conversationMode: options.conversationMode !== undefined ? String(options.conversationMode || '') : (msgDiv && msgDiv.dataset.conversationMode) || '',
        queryId: options.queryId !== undefined ? String(options.queryId || '') : (msgDiv && msgDiv.dataset.queryId) || '',
        wikiSaved: options.wikiSaved !== undefined ? Boolean(options.wikiSaved) : Boolean(msgDiv && msgDiv.dataset.wikiSaved === '1'),
        wikiReported: options.wikiReported !== undefined ? Boolean(options.wikiReported) : Boolean(msgDiv && msgDiv.dataset.wikiReported === '1')
    };
    const updated = messageId ? updatePersistedMessage(safeKB, messageId, patch) : null;
    const targetDiv = msgDiv || (safeKB === normalizeKBName(currentKB) ? findMessageDivById(messageId) : null);
    if (targetDiv) {
        const textDiv = getMessageTextElement(targetDiv);
        if (textDiv) {
            const role = targetDiv.classList.contains('user') ? 'user' : 'assistant';
            setBubbleText(textDiv, text || '', role);
        }
        applyMessageMetadata(targetDiv, updated || normalizePersistedMessage({ ...patch, id: messageId, role: targetDiv.classList.contains('user') ? 'user' : 'assistant' }));
        refreshUserBubbleStates(safeKB);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

if (loginForm) {
    loginForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (loginError) loginError.textContent = '';
        const loginId = loginIdInput ? loginIdInput.value.trim() : '';
        const password = loginPasswordInput ? loginPasswordInput.value : '';
        try {
            const res = await fetch(apiUrl('auth/login'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ login_id: loginId, password })
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.message || '로그인에 실패했습니다.');
            }
            await enterAuthenticatedApp(data.user || null);
        } catch (err) {
            if (loginError) loginError.textContent = err.message || '로그인에 실패했습니다.';
        }
    });
}

if (registerForm) {
    registerForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (loginError) loginError.textContent = '';
        const loginId = registerIdInput ? registerIdInput.value.trim() : '';
        const displayName = registerDisplayNameInput ? registerDisplayNameInput.value.trim() : '';
        const password = registerPasswordInput ? registerPasswordInput.value : '';
        const passwordConfirm = registerPasswordConfirmInput ? registerPasswordConfirmInput.value : '';
        if (password !== passwordConfirm) {
            if (loginError) loginError.textContent = '비밀번호가 일치하지 않습니다.';
            return;
        }
        try {
            const res = await fetch(apiUrl('auth/register'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    login_id: loginId,
                    display_name: displayName,
                    password,
                    password_confirm: passwordConfirm
                })
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.message || '계정을 만들 수 없습니다.');
            }
            await enterAuthenticatedApp(data.user || null);
        } catch (err) {
            if (loginError) loginError.textContent = err.message || '계정을 만들 수 없습니다.';
        }
    });
}

if (authModeLoginBtn) {
    authModeLoginBtn.addEventListener('click', () => setAuthMode('login'));
}

if (authModeRegisterBtn) {
    authModeRegisterBtn.addEventListener('click', () => setAuthMode('register'));
}

if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
        await fetch(apiUrl('auth/logout'), { method: 'POST' }).catch(() => null);
        setWikiPanelOpen(false);
        stopBackgroundOcrPolling();
        stopAllUploadPolling();
        applyCurrentUser(null);
        resetChatStateForLoggedOutUser();
        showLogin();
    });
}

// KB Management UI
async function loadKBs() {
    const res = await fetch(apiUrl('kbs'));
    if (res.status === 401) {
        stopBackgroundOcrPolling();
        stopAllUploadPolling();
        applyCurrentUser(null);
        resetChatStateForLoggedOutUser();
        showLogin();
        return [];
    }
    const data = await res.json();
    const kbList = Array.isArray(data.kbs) ? data.kbs : [];
    const kbRecords = Array.isArray(data.kb_records) ? data.kb_records : [];
    replaceKBRecords(kbList, kbRecords);
    migratePersistedChatBucketsForRecords();
    if (kbList.length > 0 && !kbList.includes(currentKB)) {
        persistActiveKB(kbList.includes('default') ? 'default' : kbList[0]);
        if (headerTitle) headerTitle.textContent = currentKB;
    }
    renderKBList(kbList);
    return kbList;
}

function renderKBList(kbs) {
    kbListElement.innerHTML = '';
    kbs.forEach(kb => {
        const wrapper = document.createElement('div');
        wrapper.className = 'kb-wrapper';

        const item = document.createElement('div');
        item.className = `sidebar-item ${kb === currentKB ? 'active-kb' : ''}`;

        const toggleSpan = document.createElement('span');
        toggleSpan.textContent = '▶';
        toggleSpan.style.cursor = 'pointer';
        toggleSpan.style.width = '20px';

        const nameSpan = document.createElement('span');
        nameSpan.textContent = kb;
        nameSpan.style.flex = '1';
        nameSpan.onclick = () => selectKB(kb);

        const menuBtn = document.createElement('span');
        menuBtn.className = 'kb-menu-btn';
        menuBtn.textContent = '⋮';
        menuBtn.onclick = (e) => {
            e.stopPropagation();
            showMenu(kb, e.pageX, e.pageY);
        };

        item.appendChild(toggleSpan);
        item.appendChild(nameSpan);
        item.appendChild(menuBtn);

        const fileList = document.createElement('div');
        fileList.className = 'kb-file-list';
        fileList.style.display = 'none';

        toggleSpan.onclick = (e) => {
            e.stopPropagation();
            if (fileList.style.display === 'none') {
                fileList.style.display = 'block';
                toggleSpan.textContent = '▼';
                loadFiles(kb, fileList);
            } else {
                fileList.style.display = 'none';
                toggleSpan.textContent = '▶';
            }
        };

        wrapper.appendChild(item);
        wrapper.appendChild(fileList);
        kbListElement.appendChild(wrapper);
    });
}

function selectKB(name) {
    if (currentKB === name) return;
    saveCurrentChat();
    persistActiveKB(name);
    if (headerTitle) headerTitle.textContent = currentKB;
    restoreChat(currentKB);
    loadKBs(); // Refresh active state
    renderBackgroundOcrStatus([]);
    pollBackgroundOcrJobs();
    if (wikiPanel && !wikiPanel.hidden) {
        loadWikiPanel();
    }
}

if (wikiPanelBtn) {
    wikiPanelBtn.addEventListener('click', () => {
        setWikiPanelOpen(!(wikiPanel && !wikiPanel.hidden));
    });
}

if (wikiPanelClose) {
    wikiPanelClose.addEventListener('click', () => setWikiPanelOpen(false));
}

if (wikiRefreshBtn) {
    wikiRefreshBtn.addEventListener('click', loadWikiPanel);
}

if (wikiExportBtn) {
    wikiExportBtn.addEventListener('click', exportWikiMarkdown);
}

if (wikiLintBtn) {
    wikiLintBtn.addEventListener('click', runWikiLint);
}

async function loadFiles(kbName, container) {
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(kbName)}/files`));
    const data = await res.json();
    container.innerHTML = '';
    if (data.files.length === 0) {
        container.innerHTML = '<div class="kb-file-item">파일 없음</div>';
        return;
    }
    data.files.forEach(fileEntry => {
        const meta = (typeof fileEntry === 'string')
            ? {
                file_id: fileEntry,
                display_name: fileEntry,
                stored_name: fileEntry,
                delete_key: fileEntry
            }
            : {
                file_id: fileEntry.file_id || fileEntry.delete_key || fileEntry.stored_name || fileEntry.display_name,
                display_name: fileEntry.display_name || fileEntry.original_filename || fileEntry.stored_name || '문서',
                stored_name: fileEntry.stored_name || '',
                delete_key: fileEntry.delete_key || fileEntry.file_id || fileEntry.stored_name || fileEntry.display_name
            };
        const fItem = document.createElement('div');
        fItem.className = 'kb-file-item';

        const fileNameEl = document.createElement('span');
        fileNameEl.className = 'kb-file-name';
        fileNameEl.textContent = `📄 ${meta.display_name}`;
        fileNameEl.title = meta.stored_name && meta.stored_name !== meta.display_name
            ? `${meta.display_name}\n저장명: ${meta.stored_name}`
            : meta.display_name;

        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'file-del-btn';
        deleteBtn.textContent = '🗑️';
        deleteBtn.title = '파일 삭제';
        deleteBtn.onclick = (event) => {
            event.stopPropagation();
            deleteFile(kbName, meta);
        };

        fItem.appendChild(fileNameEl);
        fItem.appendChild(deleteBtn);
        container.appendChild(fItem);
    });
}

async function deleteFile(kbName, fileMeta) {
    const displayName = (fileMeta && fileMeta.display_name) || String(fileMeta || '');
    const deleteKey = (fileMeta && fileMeta.delete_key) || (fileMeta && fileMeta.file_id) || String(fileMeta || '');
    if(!confirm(`'${displayName}' 파일을 삭제하시겠습니까?`)) return;
    await fetch(apiUrl(`kbs/${encodeURIComponent(kbName)}/files/${encodeURIComponent(deleteKey)}`), { method: 'DELETE' });
    loadKBs(); // Refresh lists
}

function showMenu(kb, x, y) {
    menuTargetKB = kb;
    ctxMenu.style.display = 'block';
    ctxMenu.style.left = x + 'px';
    ctxMenu.style.top = y + 'px';
}

document.getElementById('ctx-delete').onclick = async () => {
    if (menuTargetKB === 'default') return alert("기본 공간은 삭제할 수 없습니다.");
    if (!confirm(`'${menuTargetKB}' 공간을 삭제하시겠습니까? 모든 자료가 사라집니다.`)) return;
    const deletedKB = menuTargetKB;
    await fetch(apiUrl(`kbs/${encodeURIComponent(deletedKB)}`), { method: 'DELETE' });
    stopUploadPollingForKB(deletedKB);
    if (wikiPanel && !wikiPanel.hidden && currentKB === deletedKB) setWikiPanelOpen(false);
    removePersistedChatState(deletedKB);
    kbRecordByName.delete(normalizeKBName(deletedKB));
    if (currentKB === deletedKB) selectKB('default');
    loadKBs();
};

document.getElementById('ctx-rename').onclick = async () => {
    const newName = prompt(`'${menuTargetKB}'의 새 이름을 입력하세요:`, menuTargetKB);
    if (!newName || newName === menuTargetKB) return;
    const res = await fetch(apiUrl(`kbs/${encodeURIComponent(menuTargetKB)}`), {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({new_name: newName})
    });
    if (res.ok) {
        const data = await res.json();
        const oldName = menuTargetKB;
        const record = data && data.kb_record ? rememberKBRecord(data.kb_record) : null;
        renamePersistedChatState(oldName, newName);
        if (record) {
            kbRecordByName.delete(normalizeKBName(oldName));
        }
        if (currentKB === menuTargetKB) {
            persistActiveKB(newName);
            if (headerTitle) headerTitle.textContent = currentKB;
            restoreChat(currentKB);
        }
        loadKBs();
    } else {
        const err = await res.json();
        alert("실패: " + err.error);
    }
};

// Create KB
createKbBtn.onclick = async () => {
    const name = prompt("새로운 지침서 공간 이름을 입력하세요:");
    if (!name) return;
    const hadExistingRecord = kbRecordByName.has(normalizeKBName(name));
    const res = await fetch(apiUrl('kbs'), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
    });
    const data = await res.json();
    if (data.status === 'success') {
        if (data.kb_record) rememberKBRecord(data.kb_record);
        if (!hadExistingRecord) removePersistedChatState(name);
        await loadKBs();
        selectKB(name);
    } else alert("생성 실패: " + data.error);
};

// Upload
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getUploadPollingIntervalMs(job = null) {
    const status = String((job && job.status) || '').trim().toLowerCase();
    const stage = String((job && job.progress_stage) || '').trim().toLowerCase();

    if (status === 'queued' || stage === 'preparing' || stage === 'prepare_kb') {
        return 2500;
    }
    if (
        stage === 'inspect_pdf' ||
        stage === 'extract_pdf_text' ||
        stage === 'load_pdf_ocr_model' ||
        stage === 'run_pdf_ocr' ||
        stage === 'fallback_pdf_ocr'
    ) {
        return 5000;
    }
    if (stage === 'merge_pdf_ocr' || stage === 'chunk_pdf') {
        return 3000;
    }
    if (
        stage === 'store_cache' ||
        stage === 'persist_meta' ||
        stage === 'store_chunks' ||
        stage === 'refresh_index'
    ) {
        return 2000;
    }
    return 1200;
}

function getUploadLongPollWaitSeconds(job = null, idlePollCount = 0) {
    const intervalMs = getUploadPollingIntervalMs(job);
    const hidden = document.visibilityState === 'hidden';
    const baseSeconds = hidden
        ? Math.max(15, Math.min(45, Math.ceil((intervalMs * 4) / 1000)))
        : Math.max(8, Math.min(25, Math.ceil((intervalMs * 2) / 1000)));
    const normalizedIdlePollCount = Math.max(0, Math.min(8, Math.floor(Number(idlePollCount) || 0)));
    const multiplier = 1 + (normalizedIdlePollCount * 0.75);
    const maxWaitSeconds = hidden ? 120 : 75;
    return Math.max(baseSeconds, Math.min(maxWaitSeconds, Math.ceil(baseSeconds * multiplier)));
}

async function pollUploadJob(jobId, maxAttempts = 900, intervalMs = 1200, onUpdate = null, initialJobVersion = 0, initialJob = null) {
    let latestJob = initialJob;
    let jobVersion = Math.max(0, Number(initialJobVersion) || 0);
    let idlePollCount = 0;
    for (let i = 0; i < maxAttempts; i++) {
        try {
            const pollUrl = new URL(apiUrl(`upload/jobs/${encodeURIComponent(jobId)}`));
            if (jobVersion > 0) {
                pollUrl.searchParams.set('since_version', String(jobVersion));
            }
            pollUrl.searchParams.set('wait_seconds', String(getUploadLongPollWaitSeconds(latestJob, idlePollCount)));
            const res = await fetch(pollUrl.toString());
            if (res.status === 404) {
                return {
                    status: 'not_found',
                    message: '작업 정보를 찾지 못했습니다. 서버 재시작 또는 작업 만료 후 남은 이전 진행 상태일 수 있습니다.',
                    failure_code: 'upload_job_not_found'
                };
            }
            if (res.status === 204) {
                idlePollCount += 1;
                continue;
            }
            if (!res.ok) {
                await sleep(getUploadPollingIntervalMs(latestJob));
                continue;
            }
            const job = await res.json();
            latestJob = job;
            idlePollCount = 0;
            const nextVersion = Number(job && job.version);
            if (Number.isFinite(nextVersion) && nextVersion > 0) {
                jobVersion = Math.max(jobVersion, nextVersion);
            }
            if (typeof onUpdate === 'function') {
                onUpdate(job);
            }
            const status = (job.status || '').toLowerCase();
            if (status === 'success' || status === 'error') {
                return job;
            }
            intervalMs = getUploadPollingIntervalMs(job);
        } catch (err) {
            await sleep(getUploadPollingIntervalMs(latestJob));
        }
    }
    return null;
}

function formatDurationSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds <= 0) return '';
    if (seconds < 60) return `${Math.round(seconds)}초`;
    const minutes = Math.floor(seconds / 60);
    const remainSeconds = Math.round(seconds % 60);
    if (minutes < 60) return `${minutes}분 ${remainSeconds}초`;
    const hours = Math.floor(minutes / 60);
    const remainMinutes = minutes % 60;
    return `${hours}시간 ${remainMinutes}분`;
}

function uploadRoleLabel(role) {
    const v = (role || '').trim().toLowerCase();
    if (v === 'guide') return '지침서';
    if (v === 'casebook') return '사례집(Q&A)';
    return '미분류';
}

function formatUploadStageLabel(stage) {
    const value = String(stage || '').trim().toLowerCase();
    const labels = {
        queued: '대기',
        preparing: '준비',
        prepare_kb: '공간 준비',
        inspect_pdf: 'PDF 확인',
        extract_pdf_text: '텍스트 추출',
        load_pdf_ocr_model: 'OCR 모델 로드',
        run_pdf_ocr: 'OCR 실행',
        fallback_pdf_ocr: 'CPU 재시도',
        merge_pdf_ocr: 'OCR 정리',
        chunk_pdf: '청킹',
        store_cache: '캐시 저장',
        persist_meta: '메타데이터 저장',
        store_chunks: '조각 저장',
        refresh_index: '임베딩/인덱스 갱신',
        done: '완료',
        error: '오류'
    };
    return labels[value] || value;
}

function formatUploadDeviceLabel(device) {
    const value = String(device || '').trim().toLowerCase();
    if (!value) return '';
    if (value.startsWith('gpu') || value.startsWith('cuda')) return 'GPU 사용';
    if (value === 'cpu') return 'CPU';
    return value;
}

function canShowExactUploadPageProgress(job) {
    const status = String((job && job.status) || '').trim().toLowerCase();
    if (status === 'queued' || status === 'not_found') return false;
    const currentPage = Number(job && job.current_page);
    const totalPages = Number(job && job.total_pages);
    return Number.isFinite(totalPages) && totalPages > 0 && Number.isFinite(currentPage) && currentPage >= 0;
}

function formatUploadProgressDetails(job) {
    const currentPage = Number(job && job.current_page);
    const totalPages = Number(job && job.total_pages);
    if (canShowExactUploadPageProgress(job) && Number.isFinite(totalPages) && totalPages > 0 && Number.isFinite(currentPage) && currentPage >= 0) {
        const normalizedTotalPages = Math.max(1, Math.round(totalPages));
        const normalizedCurrentPage = Math.max(0, Math.min(Math.round(currentPage), normalizedTotalPages));
        return `${normalizedCurrentPage}/${normalizedTotalPages}페이지`;
    }
    return '';
}

function isUploadOcrStage(job) {
    const stage = String((job && job.progress_stage) || '').trim().toLowerCase();
    return stage === 'load_pdf_ocr_model' || stage === 'run_pdf_ocr' || stage === 'fallback_pdf_ocr' || stage === 'merge_pdf_ocr';
}

function formatSimpleUploadBody(job) {
    const status = String((job && job.status) || '').trim().toLowerCase();
    const stage = String((job && job.progress_stage) || '').trim().toLowerCase();
    const detailLabel = formatUploadProgressDetails(job);
    if (status === 'success') return '문서 정리가 끝났습니다.';
    if (status === 'queued') return '업로드 대기 중입니다.';
    if (isUploadOcrStage(job)) {
        return detailLabel ? `PDF OCR 실행 중입니다. ${detailLabel}` : 'PDF OCR 실행 중입니다.';
    }
    if (stage === 'chunk_pdf' || stage === 'store_cache' || stage === 'persist_meta' || stage === 'store_chunks' || stage === 'refresh_index') {
        return '문서 정리 중입니다.';
    }
    return (job && job.message ? String(job.message) : '').trim() || '문서를 분석하는 중입니다.';
}

function formatUploadStatusMessage(fileName, roleLabel, job) {
    const safeName = fileName || '문서';
    const safeRoleLabel = roleLabel || '문서';
    const message = formatSimpleUploadBody(job);
    const status = ((job && job.status) || '').toLowerCase();
    const percent = Number(job && job.progress_percent);
    const hasPercent = Number.isFinite(percent);
    const percentLabel = hasPercent ? ` (${Math.max(0, Math.min(100, Math.round(percent)))}%)` : '';
    const elapsedLabel = formatDurationSeconds(job && job.elapsed_seconds);
    const timingLabel = elapsedLabel ? ` [${elapsedLabel}]` : '';

    if (status === 'success') {
        return `✅ ${safeName} (${safeRoleLabel}) 완료! ${message}${timingLabel}`.trim();
    }
    if (status === 'error') {
        return `❌ ${safeName} (${safeRoleLabel}) 오류: ${message}${percentLabel}${timingLabel}`.trim();
    }
    if (status === 'not_found') {
        return `⚠️ ${safeName} (${safeRoleLabel}) 진행 상태를 더 이상 찾지 못했습니다. 파일 목록을 확인해 주세요.${timingLabel}`.trim();
    }
    return `🕒 ${safeName} (${safeRoleLabel}) ${message}${percentLabel}${timingLabel}`.trim();
}

function formatOcrJobLine(job) {
    const safeName = String((job && job.original_filename) || 'PDF 문서').trim() || 'PDF 문서';
    const status = String((job && job.status) || '').trim().toLowerCase();
    const percent = Number(job && job.progress_percent);
    const percentLabel = Number.isFinite(percent)
        ? ` (${Math.max(0, Math.min(100, Math.round(percent)))}%)`
        : '';
    const currentPage = Number(job && job.current_page);
    const totalPages = Number(job && job.total_pages);
    const pageLabel = Number.isFinite(totalPages) && totalPages > 0 && Number.isFinite(currentPage)
        ? ` ${Math.max(0, Math.min(Math.round(currentPage), Math.round(totalPages)))}/${Math.round(totalPages)}페이지`
        : '';
    if (status === 'queued') {
        return `${safeName} OCR 대기중${percentLabel}`;
    }
    if (status === 'error') {
        return `${safeName} OCR 오류: ${String((job && job.message) || '').trim() || '작업 실패'}`;
    }
    return `${safeName} OCR 추출중${pageLabel}${percentLabel}`;
}

function renderBackgroundOcrStatus(jobs = []) {
    if (!ocrStatusWidget || !ocrStatusTrigger || !ocrStatusPopover) return;
    const activeJobs = (Array.isArray(jobs) ? jobs : []).filter(job => {
        const status = String((job && job.status) || '').trim().toLowerCase();
        return status && !['success', 'skipped'].includes(status);
    });
    if (activeJobs.length === 0) {
        ocrStatusWidget.hidden = true;
        ocrStatusPopover.textContent = '';
        ocrStatusTrigger.setAttribute('aria-expanded', 'false');
        return;
    }
    ocrStatusWidget.hidden = false;
    ocrStatusTrigger.textContent = String(activeJobs.length);
    ocrStatusTrigger.title = activeJobs.map(formatOcrJobLine).join('\n');
    ocrStatusTrigger.setAttribute('aria-expanded', ocrStatusPopoverOpen ? 'true' : 'false');
    ocrStatusPopover.innerHTML = '';
    const title = document.createElement('div');
    title.className = 'ocr-status-title';
    title.textContent = '백그라운드 OCR';
    ocrStatusPopover.appendChild(title);
    activeJobs.slice(0, 4).forEach(job => {
        const row = document.createElement('div');
        row.className = 'ocr-status-row';
        row.textContent = formatOcrJobLine(job);
        ocrStatusPopover.appendChild(row);
    });
    if (activeJobs.length > 4) {
        const more = document.createElement('div');
        more.className = 'ocr-status-row muted';
        more.textContent = `${activeJobs.length - 4}개 작업 더 있음`;
        ocrStatusPopover.appendChild(more);
    }
}

async function pollBackgroundOcrJobs() {
    if (!ocrStatusWidget) return;
    try {
        const pollUrl = new URL(apiUrl('ocr/jobs'));
        pollUrl.searchParams.set('kb_name', normalizeKBName(currentKB));
        const res = await fetch(pollUrl.toString());
        if (res.status === 401) {
            stopBackgroundOcrPolling();
            stopAllUploadPolling();
            applyCurrentUser(null);
            resetChatStateForLoggedOutUser();
            showLogin();
            return;
        }
        if (!res.ok) return;
        const data = await res.json();
        renderBackgroundOcrStatus(data && data.jobs);
    } catch (err) {
        // Keep this indicator best-effort; upload/chat flows must not depend on it.
    }
}

function startBackgroundOcrPolling() {
    if (ocrPollTimerId || !ocrStatusWidget) return;
    pollBackgroundOcrJobs();
    ocrPollTimerId = window.setInterval(pollBackgroundOcrJobs, 5000);
}

function stopBackgroundOcrPolling() {
    if (ocrPollTimerId) {
        window.clearInterval(ocrPollTimerId);
        ocrPollTimerId = 0;
    }
    renderBackgroundOcrStatus([]);
}

function applyUploadJobState({ kbName, messageId, jobId, fileName, roleLabel, job }) {
    const safeKB = normalizeKBName(kbName);
    const nextStatus = String((job && job.status) || '').toLowerCase();
    const nextStage = String((job && job.progress_stage) || '').toLowerCase();
    const nextVersion = Math.max(0, Number(job && job.version) || 0);
    const nextText = formatUploadStatusMessage(fileName, roleLabel, job);
    const updated = updatePersistedMessage(safeKB, messageId, {
        text: nextText,
        kind: 'upload',
        uploadJobId: jobId,
        uploadJobVersion: nextVersion,
        uploadStatus: nextStatus,
        uploadProgressStage: nextStage,
        fileName,
        roleLabel
    });
    if (updated) {
        syncRenderedMessage(updated);
    }
    return updated;
}

function ensureUploadJobPolling({ kbName, messageId, jobId, fileName, roleLabel, uploadStatus = '', uploadProgressStage = '', uploadJobVersion = 0 }) {
    const safeKB = normalizeKBName(kbName);
    if (!jobId || !messageId) return;
    if (activeUploadPolls.has(jobId)) return;
    activeUploadPolls.set(jobId, {
        kbName: safeKB,
        messageId,
        uploadJobVersion,
        fileName,
        roleLabel,
        latestJob: null,
        elapsedBaseAtMs: 0,
        renderedElapsedSeconds: 0,
        elapsedTimerId: 0
    });

    (async () => {
        let lastProgressKey = '';
        const initialJob = { status: uploadStatus, progress_stage: uploadProgressStage, version: uploadJobVersion, elapsed_seconds: 0 };
        rememberLatestUploadJob(jobId, initialJob);
        ensureUploadElapsedTimer(jobId);
        const finalJob = await pollUploadJob(jobId, 900, getUploadPollingIntervalMs({ status: uploadStatus, progress_stage: uploadProgressStage }), (job) => {
            if (!activeUploadPolls.has(jobId)) return;
            rememberLatestUploadJob(jobId, job);
            ensureUploadElapsedTimer(jobId);
            const progressKey = [
                (job && job.status) || '',
                (job && job.progress_stage) || '',
                String(job && job.progress_percent),
                (job && job.message) || '',
                String(job && job.elapsed_seconds)
            ].join('|');
            if (progressKey === lastProgressKey) return;
            lastProgressKey = progressKey;
            applyUploadJobState({
                kbName: safeKB,
                messageId,
                jobId,
                fileName,
                roleLabel,
                job
            });
        }, uploadJobVersion, initialJob);

        if (!activeUploadPolls.has(jobId)) return;
        if (!finalJob) {
            const timeoutText = `⚠️ ${fileName} 자동 확인이 시간 제한을 넘었습니다. 실제 업로드가 멈췄을 수 있습니다. 파일 목록과 backend 로그, /upload/jobs/${jobId} 응답을 확인해 주세요.`;
            const updated = updatePersistedMessage(safeKB, messageId, {
                text: timeoutText,
                kind: 'upload',
                uploadJobId: jobId,
                uploadJobVersion: Math.max(0, Number(uploadJobVersion) || 0),
                uploadStatus: 'timeout',
                uploadProgressStage: '',
                fileName,
                roleLabel
            });
            if (updated) {
                syncRenderedMessage(updated);
            }
            stopUploadElapsedTimer(jobId);
            activeUploadPolls.delete(jobId);
            return;
        }

        rememberLatestUploadJob(jobId, finalJob);
        stopUploadElapsedTimer(jobId);
        applyUploadJobState({
            kbName: safeKB,
            messageId,
            jobId,
            fileName,
            roleLabel,
            job: finalJob
        });
        activeUploadPolls.delete(jobId);
        loadKBs();
        pollBackgroundOcrJobs();
    })();
}

function stopAllUploadPolling() {
    activeUploadPolls.forEach((pollState, jobId) => {
        if (pollState && pollState.elapsedTimerId) {
            clearInterval(pollState.elapsedTimerId);
            pollState.elapsedTimerId = 0;
        }
        activeUploadPolls.delete(jobId);
    });
}

function stopUploadPollingForKB(kbName) {
    const safeKB = normalizeKBName(kbName);
    activeUploadPolls.forEach((pollState, jobId) => {
        if (!pollState || normalizeKBName(pollState.kbName) !== safeKB) return;
        if (pollState.elapsedTimerId) {
            clearInterval(pollState.elapsedTimerId);
            pollState.elapsedTimerId = 0;
        }
        activeUploadPolls.delete(jobId);
    });
}

function resumePendingUploadJobsForKB(kbName) {
    loadPersistedMessages(kbName).forEach(message => {
        if (message.kind !== 'upload') return;
        if (!message.uploadJobId || isTerminalUploadStatus(message.uploadStatus)) return;
        ensureUploadJobPolling({
            kbName,
            messageId: message.id,
            jobId: message.uploadJobId,
            fileName: message.fileName || '문서',
            roleLabel: message.roleLabel || '문서',
            uploadJobVersion: message.uploadJobVersion || 0,
            uploadStatus: message.uploadStatus || '',
            uploadProgressStage: message.uploadProgressStage || ''
        });
    });
}

function resumeAllPendingUploadJobs() {
    Object.keys(chatStateCache.chats || {}).forEach(kbName => {
        resumePendingUploadJobsForKB(kbName);
    });
}

fileInput.addEventListener('change', async () => {
    if (fileInput.files.length === 0) return;
    const file = fileInput.files[0];
    fileInput.value = '';
    const targetKB = normalizeKBName(currentKB);
    const selectedRole = (uploadDocRoleSelect && uploadDocRoleSelect.value || '').trim();
    if (!selectedRole) {
        appendMessage('assistant', '⚠️ 업로드 전에 문서 유형(지침서/사례집)을 먼저 선택해 주세요.');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    formData.append('kb_name', targetKB);
    formData.append('document_role', selectedRole);
    const initialRoleLabel = uploadRoleLabel(selectedRole);

    const progressMsgDiv = appendMessage(
        'assistant',
        `📂 '${targetKB}'에 ${initialRoleLabel} 파일을 올리는 중...`,
        true,
        {
            kbName: targetKB,
            kind: 'upload',
            uploadJobVersion: 0,
            uploadStatus: 'uploading',
            uploadProgressStage: '',
            fileName: file.name,
            roleLabel: initialRoleLabel
        }
    );

    try {
        const res = await fetch(apiUrl('upload'), { method: 'POST', body: formData });
        const data = await res.json();

        if (data.status !== 'success') {
            setMessageText(progressMsgDiv, `❌ ${file.name} 오류: ${data.message || '업로드 실패'}`, {
                kbName: targetKB,
                kind: 'upload',
                uploadJobVersion: 0,
                uploadStatus: 'error',
                uploadProgressStage: '',
                fileName: file.name,
                roleLabel: initialRoleLabel
            });
            return;
        }

        const roleLabel = data.document_role_label || initialRoleLabel;

        if (data.queued && data.job_id) {
            setMessageText(
                progressMsgDiv,
                formatUploadStatusMessage(file.name, roleLabel, {
                    status: 'queued',
                    message: data.message || '파일이 접수되었습니다.',
                    progress_percent: 0
                }),
                {
                    kbName: targetKB,
                    kind: 'upload',
                    uploadJobId: data.job_id,
                    uploadJobVersion: Math.max(0, Number(data.version) || 0),
                    uploadStatus: 'queued',
                    uploadProgressStage: '',
                    fileName: file.name,
                    roleLabel
                }
            );

            ensureUploadJobPolling({
                kbName: targetKB,
                messageId: progressMsgDiv.dataset.messageId,
                jobId: data.job_id,
                fileName: file.name,
                roleLabel,
                uploadJobVersion: Math.max(0, Number(data.version) || 0),
                uploadStatus: 'queued',
                uploadProgressStage: ''
            });
        } else {
            setMessageText(
                progressMsgDiv,
                formatUploadStatusMessage(file.name, roleLabel, {
                    status: 'success',
                    message: data.message || '문서 처리가 완료되었습니다.',
                    progress_percent: 100
                }),
                {
                    kbName: targetKB,
                    kind: 'upload',
                    uploadJobVersion: 0,
                    uploadStatus: 'success',
                    uploadProgressStage: '',
                    fileName: file.name,
                    roleLabel
                }
            );
            pollBackgroundOcrJobs();
        }
    } catch (error) {
        setMessageText(progressMsgDiv, `❌ ${file.name} 오류: ${error.message || '업로드 실패'}`, {
            kbName: targetKB,
            kind: 'upload',
            uploadJobVersion: 0,
            uploadStatus: 'error',
            uploadProgressStage: '',
            fileName: file.name,
            roleLabel: initialRoleLabel
        });
    }
    loadKBs(); // Refresh file lists
    persistChatState(targetKB);
});

[uploadTrigger, uploadBtn].forEach(b => b && b.addEventListener('click', () => fileInput.click()));

if (ocrStatusTrigger) {
    ocrStatusTrigger.addEventListener('click', () => {
        ocrStatusPopoverOpen = !ocrStatusPopoverOpen;
        if (ocrStatusWidget) {
            ocrStatusWidget.classList.toggle('open', ocrStatusPopoverOpen);
        }
        ocrStatusTrigger.setAttribute('aria-expanded', ocrStatusPopoverOpen ? 'true' : 'false');
    });
}

// Init
async function initApp() {
    const loggedIn = await requireLoggedIn();
    if (!loggedIn) return;
    if (headerTitle) headerTitle.textContent = currentKB;
    await loadKBs();
    await restoreChat(currentKB);
    resumeAllPendingUploadJobs();
    startBackgroundOcrPolling();
}

initApp();
