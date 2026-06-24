// CNIRS app-like guided demo assistant.

(function () {
    const ACTIVE_KEY = 'cnirs_demo_assistant_active';
    const STEP_KEY = 'cnirs_demo_assistant_step';
    const SEEN_KEY = 'cnirs_demo_assistant_seen';
    const BASE_QUERY = '半導體 人工智慧';

    const copy = {
        'zh-TW': {
            appTitle: '小幫手',
            appSubtitle: '互動式作品導覽',
            open: '開啟導覽',
            collapse: '收合',
            back: '上一步',
            next: '下一步',
            finish: '完成',
            recenter: '重新定位',
            exit: '結束導覽',
            step: '步驟',
            currentArea: '目前區域',
            waiting: '正在載入目標區塊...',
            missing: '此區塊尚未載入，請等待或重新執行。',
            steps: {
                overview: ['系統總覽', '先看完整展示路線：從新聞查詢、模型解釋、資料探索、評估到 feedback learning。', '這頁可以當作面試 demo 的口頭流程稿。'],
                search: ['執行新聞查詢', '使用 Hybrid 查詢真實新聞語料，結果會包含 snippet、score、metadata 與 explanation。', '確認搜尋框、模型選擇與結果列表都已載入。'],
                facets: ['分面篩選 Facets', '用 source、taxonomy、date、tags、publisher metadata 即時縮小結果，不需要改 query。', '觀察左側 facets；這裡已清理雜訊值並標示 metadata 品質。'],
                explain: ['Why this result?', '每筆結果會揭露 matched terms、field boost、component scores 與 ranking signals。', '小幫手會定位到第一筆 explanation panel。'],
                detail: ['文章詳情', '文章 modal 會展示完整 metadata、summary、KWIC、keywords、related news 與 taxonomy。', '點進文章可以展示檢索結果如何被解釋與延伸探索。'],
                compare: ['模型對比', '同一個 query 可比較 BM25、TF-IDF、Hybrid、LM、BIM、WAND 與 MaxScore。', '看 overlap、unique documents、timing 與 optimization diagnostics。'],
                corpus: ['語料庫儀表板', '展示資料規模、來源分布、taxonomy coverage、facet quality 與 index cache 狀態。', '這頁用來證明系統不是小型課堂 demo。'],
                topics: ['Topic Explorer', '對 query-focused sample 做輕量 clustering，產生 topic cards 與代表文章。', '檢視主題卡片，並可接著開啟代表新聞。'],
                evaluation: ['評估儀表板', '使用 demo qrels 比較 Precision@K、Recall@K、MAP、MRR、nDCG 與 PR curves。', '這裡明確標示是 demo evaluation，不假裝完整 benchmark。'],
                diagnostics: ['排序診斷', '拆解 BM25、TF-IDF、LM 與 field-aware signals，說明分數如何形成。', '這是展示 IR 可解釋性的核心頁面。'],
                analysis: ['分析圖譜', '用節點圖呈現 query、processing、index、ranking、model、document、metadata 與 feedback 的資料流。', 'Hover 節點可預覽，點擊文章節點可接到 document detail。'],
                feedback: ['Feedback + LTR', '回饋儀表板展示 clicks、relevance labels、zero-result queries、quality controls 與 LTR sandbox。', '用來說明搜尋系統如何從使用者行為改善排序。'],
                wrap: ['展示總結', '完整 demo 已涵蓋資料、檢索、解釋、評估、診斷與回饋學習。', '照這份流程走，就能完整呈現作品集亮點。'],
            },
        },
        en: {
            appTitle: 'Demo Helper',
            appSubtitle: 'Interactive portfolio walkthrough',
            open: 'Open Tour',
            collapse: 'Collapse',
            back: 'Back',
            next: 'Next',
            finish: 'Finish',
            recenter: 'Recenter',
            exit: 'Exit Tour',
            step: 'Step',
            currentArea: 'Current Area',
            waiting: 'Loading target section...',
            missing: 'This section is not loaded yet. Please wait or run the demo action again.',
            steps: {
                overview: ['System Overview', 'Start with the complete demo route: news search, model explanations, corpus exploration, evaluation, and feedback learning.', 'Use this page as the interview demo script.'],
                search: ['Search News', 'Run a Hybrid query over the real news corpus. Results include snippets, scores, metadata, and explanations.', 'Check the search box, selected model, and ranked result list.'],
                facets: ['Faceted Search', 'Use source, taxonomy, date, tags, and publisher metadata to narrow results without changing the query.', 'Inspect the left sidebar. Facets are cleaned and sorted by metadata quality.'],
                explain: ['Why this result?', 'Each result exposes matched terms, field boosts, component scores, and ranking signals.', 'The helper anchors to the first explanation panel.'],
                detail: ['Document Detail', 'The modal shows metadata, summary, KWIC, keywords, related news, and taxonomy context.', 'Open a result to show how one retrieved article can be explained and explored.'],
                compare: ['Model Comparison', 'Compare BM25, TF-IDF, Hybrid, LM, BIM, WAND, and MaxScore on the same query.', 'Review overlap, unique documents, timing, and optimization diagnostics.'],
                corpus: ['Corpus Dashboard', 'Show data scale, source distribution, taxonomy coverage, facet quality, and index cache state.', 'Use this page to prove the system is more than a small class demo.'],
                topics: ['Topic Explorer', 'Cluster a query-focused sample into topic cards and representative documents.', 'Inspect generated topic cards and continue into representative news.'],
                evaluation: ['Evaluation Dashboard', 'Run demo qrels for Precision@K, Recall@K, MAP, MRR, nDCG, and PR curves.', 'The page labels this as demo evaluation, not a full benchmark.'],
                diagnostics: ['Ranking Diagnostics', 'Break down BM25, TF-IDF, LM, and field-aware signals to explain how scores are formed.', 'This is the core explainable IR page.'],
                analysis: ['Analysis Graph', 'Visualize the data flow from query, processing, index, ranking, models, documents, metadata, and feedback.', 'Hover nodes for previews and click document nodes to open document details.'],
                feedback: ['Feedback + LTR', 'Feedback analytics show clicks, relevance labels, zero-result queries, quality controls, and the LTR sandbox.', 'Use it to show how search can improve from user behavior.'],
                wrap: ['Wrap-up', 'The demo covers data, retrieval, explanation, evaluation, diagnostics, and feedback learning.', 'Follow this flow to present the portfolio highlights.'],
            },
        },
    };

    const steps = [
        { id: 'overview', path: 'guide', target: '.guide-hero', url: '/guide?tour=1&step=overview' },
        { id: 'search', path: '', target: '.search-box', waitFor: '#results-list', url: `/?q=${encodeURIComponent(BASE_QUERY)}&model=hybrid&run=1&tour=1&step=search` },
        { id: 'facets', path: '', target: '#filter-sidebar', waitFor: '#facet-groups', url: '/?q=%E5%8F%B0%E7%81%A3%20%E7%B6%93%E6%BF%9F&model=bm25&taxonomy_topic=business&run=1&tour=1&step=facets' },
        { id: 'explain', path: '', target: '.explain-panel', waitFor: '.result-item, .model-result-item, .explain-panel', action: openFirstExplanation, url: `/?q=${encodeURIComponent(BASE_QUERY)}&model=hybrid&run=1&tour=1&step=explain` },
        { id: 'detail', path: '', target: '#doc-modal .modal-content, .result-item', waitFor: '.result-item', action: openFirstDocument, url: '/?q=%E5%8F%B0%E7%81%A3%20%E7%B6%93%E6%BF%9F&model=bm25&run=1&tour=1&step=detail' },
        { id: 'compare', path: 'compare', target: '#comparison-container', waitFor: '#comparison-container .model-results', url: '/compare?q=%E7%BE%8E%E5%9C%8B%20%E4%B8%AD%E5%9C%8B&models=bm25,tfidf,hybrid,lm,bim,wand_bm25,maxscore_bm25&run=1&tour=1&step=compare' },
        { id: 'corpus', path: 'corpus', target: '.corpus-page', waitFor: '#corpus-content', url: '/corpus?tour=1&step=corpus' },
        { id: 'topics', path: 'corpus', target: '#topic-results', waitFor: '.topic-card, #topic-results', url: `/corpus?tour=1&step=topics&topic_query=${encodeURIComponent(BASE_QUERY)}&run_topic=1` },
        { id: 'evaluation', path: 'evaluation', target: '#eval-results', waitFor: '#eval-results', url: '/evaluation?query_set=news_demo&models=bm25,tfidf,hybrid,lm&top_k=10&run=1&tour=1&step=evaluation' },
        { id: 'diagnostics', path: 'diagnostics', target: '#diagnostics-results', waitFor: '#diagnostics-results', url: '/diagnostics?q=%E5%8D%8A%E5%B0%8E%E9%AB%94&doc_id=1&models=bm25,tfidf,lm&run=1&tour=1&step=diagnostics' },
        { id: 'analysis', path: 'analysis-graph', target: '.analysis-graph-stage', waitFor: '.graph-node', url: '/analysis-graph?query=%E5%8F%B0%E7%81%A3%20%E7%B6%93%E6%BF%9F&models=bm25,tfidf,hybrid,lm&top_k=6&tour=1&step=analysis' },
        { id: 'feedback', path: 'feedback', target: '#feedback-dashboard', waitFor: '#feedback-dashboard', url: '/feedback?tour=1&step=feedback' },
        { id: 'wrap', path: 'guide', target: '.pipeline-card', url: '/guide?tour=1&step=wrap' },
    ];

    let currentStepId = 'overview';
    let currentTarget = null;
    let repositionTimer = null;

    window.addEventListener('DOMContentLoaded', init);
    window.addEventListener('resize', () => schedulePosition());
    window.addEventListener('scroll', () => schedulePosition(), { passive: true });
    window.addEventListener('cnirs:langchange', () => {
        if (isActive()) {
            renderAssistant();
            positionCoach();
        }
    });

    function init() {
        const params = new URLSearchParams(window.location.search);
        const requestedTour = params.get('tour') === '1' || params.get('tour') === 'true';
        const firstVisit = localStorage.getItem(SEEN_KEY) !== '1';
        if (firstVisit && !requestedTour && currentPath() !== 'guide') {
            localStorage.setItem(ACTIVE_KEY, '1');
            localStorage.setItem(SEEN_KEY, '1');
            localStorage.setItem(STEP_KEY, 'overview');
            window.location.href = withBase('/guide?tour=1&step=overview');
            return;
        }

        currentStepId = params.get('step') || localStorage.getItem(STEP_KEY) || 'overview';
        if (requestedTour || localStorage.getItem(ACTIVE_KEY) === '1' || firstVisit) {
            localStorage.setItem(ACTIVE_KEY, '1');
            localStorage.setItem(SEEN_KEY, '1');
            localStorage.setItem(STEP_KEY, currentStepId);
            renderAssistant();
            activateCurrentStep();
        } else {
            renderLauncher();
        }
    }

    async function activateCurrentStep() {
        const step = currentStep();
        renderAssistant(true);
        await new Promise(resolve => window.setTimeout(resolve, 0));
        if (step.action) {
            window.setTimeout(step.action, step.id === 'detail' ? 900 : 500);
        }
        currentTarget = await waitForTarget(step.target, step.waitFor || step.target, 22000);
        highlightTarget(currentTarget);
        renderAssistant(false);
        positionCoach();
    }

    function renderLauncher() {
        removeElement('demo-assistant-app');
        removeElement('demo-assistant-coach');
        clearHighlights();
        const launcher = document.createElement('button');
        launcher.id = 'demo-assistant-launcher';
        launcher.className = 'demo-assistant-launcher app-like';
        launcher.type = 'button';
        launcher.innerHTML = `<span>?</span><strong>${escapeHtml(labels().appTitle)}</strong>`;
        launcher.addEventListener('click', () => startTour('overview'));
        document.body.appendChild(launcher);
    }

    function renderAssistant(waiting = false) {
        removeElement('demo-assistant-launcher');
        removeElement('demo-assistant-app');
        removeElement('demo-assistant-coach');
        const label = labels();
        const step = currentStep();
        const stepText = stepCopy(step.id);
        const index = steps.findIndex(item => item.id === step.id);

        const app = document.createElement('aside');
        app.id = 'demo-assistant-app';
        app.className = 'demo-assistant-app';
        app.innerHTML = `
            <div class="demo-assistant-app-head">
                <div>
                    <div class="demo-assistant-app-title">${escapeHtml(label.appTitle)}</div>
                    <div class="demo-assistant-app-subtitle">${escapeHtml(label.appSubtitle)}</div>
                </div>
                <button type="button" class="demo-assistant-icon-btn" data-tour-collapse aria-label="${escapeAttr(label.collapse)}">−</button>
            </div>
            <div class="demo-assistant-progress">
                <span>${escapeHtml(label.step)} ${index + 1} / ${steps.length}</span>
                <div class="demo-assistant-meter"><span style="width:${((index + 1) / steps.length) * 100}%"></span></div>
            </div>
            <div class="demo-assistant-current">
                <span>${escapeHtml(label.currentArea)}</span>
                <strong>${escapeHtml(stepText[0])}</strong>
            </div>
            <div class="demo-assistant-mini-steps">
                ${steps.map((item, itemIndex) => `
                    <button type="button" data-tour-step="${escapeAttr(item.id)}" class="${item.id === step.id ? 'active' : ''}" title="${escapeAttr(stepCopy(item.id)[0])}">
                        ${itemIndex + 1}
                    </button>
                `).join('')}
            </div>
            <div class="demo-assistant-controls">
                <button type="button" class="btn btn-secondary" data-tour-prev ${index === 0 ? 'disabled' : ''}>${escapeHtml(label.back)}</button>
                <button type="button" class="btn btn-secondary" data-tour-recenter>${escapeHtml(label.recenter)}</button>
                <button type="button" class="btn btn-primary" data-tour-next>${escapeHtml(index === steps.length - 1 ? label.finish : label.next)}</button>
            </div>
            <button type="button" class="demo-assistant-exit" data-tour-exit>${escapeHtml(label.exit)}</button>
        `;
        wireControls(app);
        document.body.appendChild(app);

        const coach = document.createElement('section');
        coach.id = 'demo-assistant-coach';
        coach.className = 'demo-assistant-coach';
        coach.innerHTML = `
            <div class="demo-assistant-coach-kicker">${escapeHtml(label.step)} ${index + 1}</div>
            <h2>${escapeHtml(stepText[0])}</h2>
            <p>${escapeHtml(waiting ? label.waiting : stepText[1])}</p>
            <div class="demo-assistant-action">${escapeHtml(waiting ? label.missing : stepText[2])}</div>
            <div class="demo-assistant-coach-actions">
                <button type="button" class="btn btn-secondary" data-tour-recenter>${escapeHtml(label.recenter)}</button>
                <button type="button" class="btn btn-primary" data-tour-next>${escapeHtml(index === steps.length - 1 ? label.finish : label.next)}</button>
            </div>
        `;
        wireControls(coach);
        document.body.appendChild(coach);
        if (window.CNIRS_I18N) window.CNIRS_I18N.applyI18n(document.body);
    }

    function wireControls(root) {
        root.querySelector('[data-tour-collapse]')?.addEventListener('click', renderLauncher);
        root.querySelector('[data-tour-prev]')?.addEventListener('click', previousStep);
        root.querySelector('[data-tour-next]')?.addEventListener('click', nextStep);
        root.querySelector('[data-tour-recenter]')?.addEventListener('click', () => {
            if (currentTarget) {
                currentTarget.scrollIntoView({ behavior: 'smooth', block: 'center' });
                positionCoach();
            }
        });
        root.querySelector('[data-tour-exit]')?.addEventListener('click', exitTour);
        root.querySelectorAll('[data-tour-step]').forEach(button => {
            button.addEventListener('click', () => navigateTo(button.dataset.tourStep));
        });
    }

    function startTour(stepId) {
        localStorage.setItem(ACTIVE_KEY, '1');
        localStorage.setItem(SEEN_KEY, '1');
        navigateTo(stepId);
    }

    function nextStep() {
        const index = steps.findIndex(item => item.id === currentStepId);
        if (index >= steps.length - 1) {
            exitTour();
            return;
        }
        navigateTo(steps[index + 1].id);
    }

    function previousStep() {
        const index = steps.findIndex(item => item.id === currentStepId);
        if (index > 0) navigateTo(steps[index - 1].id);
    }

    function navigateTo(stepId) {
        const step = steps.find(item => item.id === stepId) || steps[0];
        localStorage.setItem(STEP_KEY, step.id);
        localStorage.setItem(ACTIVE_KEY, '1');
        window.location.href = withBase(step.url);
    }

    function exitTour() {
        localStorage.removeItem(ACTIVE_KEY);
        localStorage.removeItem(STEP_KEY);
        localStorage.setItem(SEEN_KEY, '1');
        currentTarget = null;
        removeElement('demo-assistant-app');
        removeElement('demo-assistant-coach');
        clearHighlights();
        renderLauncher();
    }

    function currentStep() {
        return steps.find(item => item.id === currentStepId) || steps[0];
    }

    function labels() {
        const lang = window.CNIRS_I18N?.currentLang?.() || 'zh-TW';
        return copy[lang] || copy['zh-TW'];
    }

    function stepCopy(stepId) {
        return labels().steps[stepId] || copy['zh-TW'].steps[stepId] || copy['zh-TW'].steps.overview;
    }

    function isActive() {
        return localStorage.getItem(ACTIVE_KEY) === '1';
    }

    function waitForTarget(targetSelector, readySelector, timeoutMs) {
        const started = Date.now();
        return new Promise(resolve => {
            const tick = () => {
                const ready = document.querySelector(readySelector);
                const target = document.querySelector(targetSelector);
                if ((ready && target) || Date.now() - started > timeoutMs) {
                    resolve(target || ready || document.body);
                    return;
                }
                window.setTimeout(tick, 250);
            };
            tick();
        });
    }

    function highlightTarget(target) {
        clearHighlights();
        if (!target || target === document.body) return;
        target.classList.add('demo-assistant-highlight');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function positionCoach() {
        const coach = document.getElementById('demo-assistant-coach');
        if (!coach || !currentTarget || currentTarget === document.body) {
            if (coach) coach.classList.add('centered');
            return;
        }
        coach.classList.remove('centered', 'place-left', 'place-right', 'place-top', 'place-bottom');
        const rect = currentTarget.getBoundingClientRect();
        const margin = 18;
        const width = Math.min(360, window.innerWidth - 28);
        coach.style.width = `${width}px`;
        const coachHeight = Math.min(coach.offsetHeight || 260, window.innerHeight - 28);
        let left = rect.right + margin;
        let top = rect.top + (rect.height / 2) - (coachHeight / 2);
        let placement = 'place-right';

        if (left + width > window.innerWidth - margin) {
            left = rect.left - width - margin;
            placement = 'place-left';
        }
        if (left < margin) {
            left = Math.min(Math.max(rect.left, margin), window.innerWidth - width - margin);
            top = rect.bottom + margin;
            placement = 'place-bottom';
        }
        if (top + coachHeight > window.innerHeight - margin) top = window.innerHeight - coachHeight - margin;
        if (top < margin) {
            top = rect.top - coachHeight - margin;
            placement = 'place-top';
        }
        if (top < margin) top = margin;

        coach.style.left = `${Math.round(left)}px`;
        coach.style.top = `${Math.round(top)}px`;
        coach.classList.add(placement);
    }

    function schedulePosition() {
        window.clearTimeout(repositionTimer);
        repositionTimer = window.setTimeout(positionCoach, 80);
    }

    function openFirstExplanation() {
        const panel = document.querySelector('.explain-panel');
        if (panel && panel.tagName === 'DETAILS') panel.open = true;
    }

    function openFirstDocument() {
        if (document.querySelector('#doc-modal[style*="block"] .modal-content')) return;
        const doc = document.querySelector('.result-item[data-doc-id], .model-result-item[data-doc-id]');
        const docId = doc?.dataset?.docId;
        if (docId && typeof window.openDocumentModal === 'function') {
            window.openDocumentModal(docId);
        } else if (doc) {
            doc.click();
        }
    }

    function clearHighlights() {
        document.querySelectorAll('.demo-assistant-highlight').forEach(item => {
            item.classList.remove('demo-assistant-highlight');
        });
    }

    function removeElement(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    function currentPath() {
        const path = window.location.pathname.replace(/^\/+|\/+$/g, '');
        const base = basePath().replace(/^\/+|\/+$/g, '');
        if (base && path.startsWith(`${base}/`)) {
            return path.slice(base.length + 1);
        }
        if (base && path === base) return '';
        return path;
    }

    function basePath() {
        const marker = '/projects/information-retrieval';
        const path = window.location.pathname;
        const index = path.indexOf(marker);
        return index >= 0 ? path.slice(0, index + marker.length) : '';
    }

    function withBase(url) {
        if (!url || /^[a-z]+:/i.test(url)) return url;
        const base = basePath();
        if (!base || !url.startsWith('/')) return url;
        return `${base}${url}`;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeAttr(value) {
        return escapeHtml(value).replace(/`/g, '&#096;');
    }
})();
