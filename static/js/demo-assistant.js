// CNIRS guided demo assistant.

(function () {
    const STORAGE_KEY = 'cnirs_demo_assistant_active';
    const STEP_KEY = 'cnirs_demo_assistant_step';
    const SEEN_KEY = 'cnirs_demo_assistant_seen';
    const BASE_QUERY = '半導體 人工智慧';

    const steps = [
        {
            id: 'overview',
            path: 'guide',
            target: '.guide-hero',
            title: 'Overview',
            body: 'Start here. This walkthrough shows the full portfolio flow from search to evaluation and feedback learning.',
            action: 'Review the demo goals, then continue to run a real news search.',
            url: 'guide?tour=1&step=overview',
        },
        {
            id: 'search',
            path: '',
            target: '.search-box',
            title: 'Search News',
            body: 'Run a Hybrid query over the unified Taiwanese news corpus. The results include snippets, scores, metadata, and explanations.',
            action: 'Inspect the search box, selected model, and returned result list.',
            url: `?q=${encodeURIComponent(BASE_QUERY)}&model=hybrid&run=1&tour=1&step=search`,
        },
        {
            id: 'facets',
            path: '',
            target: '#filter-sidebar',
            title: 'Faceted Search',
            body: 'Use source, taxonomy, date, tags, and publisher metadata to narrow results without changing the query.',
            action: 'Look at the left sidebar. Facets are cleaned and sorted by metadata quality.',
            url: '?q=%E5%8F%B0%E7%81%A3%20%E7%B6%93%E6%BF%9F&model=bm25&taxonomy_topic=business&run=1&tour=1&step=facets',
        },
        {
            id: 'explain',
            path: '',
            target: '.explain-panel',
            title: 'Why This Result?',
            body: 'Each result exposes matched terms, field boosts, component scores, and ranking signals.',
            action: 'Open or scan the explanation panel under a result.',
            url: `?q=${encodeURIComponent(BASE_QUERY)}&model=hybrid&run=1&tour=1&step=explain`,
        },
        {
            id: 'detail',
            path: '',
            target: '.result-item',
            title: 'Document Detail',
            body: 'Document detail shows full metadata, summary, KWIC, keywords, related news, and taxonomy context.',
            action: 'Click the first result title to open the detail modal.',
            url: '?q=%E5%8F%B0%E7%81%A3%20%E7%B6%93%E6%BF%9F&model=bm25&run=1&tour=1&step=detail',
        },
        {
            id: 'compare',
            path: 'compare',
            target: '#comparison-container',
            title: 'Model Comparison',
            body: 'Compare BM25, TF-IDF, Hybrid, LM, BIM, WAND, and MaxScore on the same query.',
            action: 'Review overlap, unique documents, timing, and optimization diagnostics.',
            url: 'compare?q=%E7%BE%8E%E5%9C%8B%20%E4%B8%AD%E5%9C%8B&models=bm25,tfidf,hybrid,lm,bim,wand_bm25,maxscore_bm25&run=1&tour=1&step=compare',
        },
        {
            id: 'corpus',
            path: 'corpus',
            target: '.corpus-page',
            title: 'Corpus Explorer',
            body: 'Show the data scale, source distribution, taxonomy coverage, facet quality, and index cache status.',
            action: 'Use this page to prove the system uses a real multi-source news corpus.',
            url: 'corpus?tour=1&step=corpus',
        },
        {
            id: 'topics',
            path: 'corpus',
            target: '#topic-results',
            title: 'Topic Explorer',
            body: 'Cluster a query-focused sample to reveal topical groups and representative documents.',
            action: 'Inspect generated topic cards and click a representative document to continue exploring.',
            url: `corpus?tour=1&step=topics&topic_query=${encodeURIComponent(BASE_QUERY)}&run_topic=1`,
        },
        {
            id: 'evaluation',
            path: 'evaluation',
            target: '#eval-results',
            title: 'Evaluation Dashboard',
            body: 'Run the small demo qrels to compare Precision@K, Recall@K, MAP, MRR, nDCG, and PR curves.',
            action: 'Metrics are explicitly labeled as demo evaluation, not a full academic benchmark.',
            url: 'evaluation?query_set=news_demo&models=bm25,tfidf,hybrid,lm&top_k=10&run=1&tour=1&step=evaluation',
        },
        {
            id: 'diagnostics',
            path: 'diagnostics',
            target: '#diagnostics-results',
            title: 'Ranking Diagnostics',
            body: 'Break down BM25, TF-IDF, and LM contributions for one result and inspect field-aware signals.',
            action: 'Use this page to explain how ranking scores are formed.',
            url: 'diagnostics?q=%E5%8D%8A%E5%B0%8E%E9%AB%94&doc_id=1&models=bm25,tfidf,lm&run=1&tour=1&step=diagnostics',
        },
        {
            id: 'feedback',
            path: 'feedback',
            target: '#feedback-dashboard',
            title: 'Feedback + LTR',
            body: 'Feedback analytics show clicks, relevance labels, zero-result queries, quality controls, and the weak-supervision LTR sandbox.',
            action: 'This demonstrates how the search system can improve from user behavior.',
            url: 'feedback?tour=1&step=feedback',
        },
        {
            id: 'wrap',
            path: 'guide',
            target: '.pipeline-card',
            title: 'Wrap-up',
            body: 'The demo now covers data, retrieval models, explanation, evaluation, diagnostics, and feedback learning.',
            action: 'Use this checklist as the interview presentation script.',
            url: 'guide?tour=1&step=wrap',
        },
    ];

    let currentStepId = null;

    window.addEventListener('DOMContentLoaded', () => {
        const params = new URLSearchParams(window.location.search);
        const requestedTour = params.get('tour') === '1' || params.get('tour') === 'true';
        const firstVisit = localStorage.getItem(SEEN_KEY) !== '1';
        if (firstVisit && !requestedTour && currentPath() !== 'guide') {
            localStorage.setItem(STORAGE_KEY, '1');
            localStorage.setItem(SEEN_KEY, '1');
            localStorage.setItem(STEP_KEY, 'overview');
            window.location.href = 'guide?tour=1&step=overview';
            return;
        }
        const active = requestedTour || localStorage.getItem(STORAGE_KEY) === '1' || firstVisit;
        if (!active) {
            renderLauncher();
            return;
        }
        localStorage.setItem(STORAGE_KEY, '1');
        localStorage.setItem(SEEN_KEY, '1');
        currentStepId = params.get('step') || localStorage.getItem(STEP_KEY) || 'overview';
        localStorage.setItem(STEP_KEY, currentStepId);
        renderAssistant();
        waitForTarget(currentStep().target, 16000).then(highlightTarget);
    });

    function renderLauncher() {
        if (document.getElementById('demo-assistant-launcher')) return;
        const launcher = document.createElement('button');
        launcher.id = 'demo-assistant-launcher';
        launcher.className = 'demo-assistant-launcher';
        launcher.type = 'button';
        launcher.textContent = 'Demo Helper';
        launcher.addEventListener('click', () => startTour('overview'));
        document.body.appendChild(launcher);
    }

    function renderAssistant() {
        removeElement('demo-assistant-launcher');
        removeElement('demo-assistant-panel');
        const step = currentStep();
        const index = steps.findIndex(item => item.id === step.id);
        const panel = document.createElement('aside');
        panel.id = 'demo-assistant-panel';
        panel.className = 'demo-assistant-panel';
        panel.innerHTML = `
            <div class="demo-assistant-topline">
                <span>小幫手 Demo Assistant</span>
                <button type="button" class="demo-assistant-close" aria-label="Close demo assistant">×</button>
            </div>
            <div class="demo-assistant-progress">
                <span>Step ${index + 1} / ${steps.length}</span>
                <div class="demo-assistant-meter"><span style="width:${((index + 1) / steps.length) * 100}%"></span></div>
            </div>
            <h2>${escapeHtml(step.title)}</h2>
            <p>${escapeHtml(step.body)}</p>
            <div class="demo-assistant-action">${escapeHtml(step.action)}</div>
            <div class="demo-assistant-controls">
                <button type="button" class="btn btn-secondary" data-tour-prev ${index === 0 ? 'disabled' : ''}>Previous</button>
                <button type="button" class="btn btn-secondary" data-tour-open>Open Target</button>
                <button type="button" class="btn btn-primary" data-tour-next>${index === steps.length - 1 ? 'Finish' : 'Next'}</button>
            </div>
            <div class="demo-assistant-step-list">
                ${steps.map((item, itemIndex) => `
                    <button type="button" data-tour-step="${escapeAttr(item.id)}" class="${item.id === step.id ? 'active' : ''}">
                        ${itemIndex + 1}. ${escapeHtml(item.title)}
                    </button>
                `).join('')}
            </div>
        `;
        panel.querySelector('.demo-assistant-close').addEventListener('click', exitTour);
        panel.querySelector('[data-tour-prev]')?.addEventListener('click', previousStep);
        panel.querySelector('[data-tour-next]')?.addEventListener('click', nextStep);
        panel.querySelector('[data-tour-open]')?.addEventListener('click', () => navigateTo(step.id));
        panel.querySelectorAll('[data-tour-step]').forEach(button => {
            button.addEventListener('click', () => navigateTo(button.dataset.tourStep));
        });
        document.body.appendChild(panel);
    }

    function startTour(stepId) {
        localStorage.setItem(STORAGE_KEY, '1');
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
        if (index > 0) {
            navigateTo(steps[index - 1].id);
        }
    }

    function navigateTo(stepId) {
        const step = steps.find(item => item.id === stepId) || steps[0];
        localStorage.setItem(STEP_KEY, step.id);
        localStorage.setItem(STORAGE_KEY, '1');
        window.location.href = step.url;
    }

    function exitTour() {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(STEP_KEY);
        localStorage.setItem(SEEN_KEY, '1');
        document.querySelectorAll('.demo-assistant-highlight').forEach(item => {
            item.classList.remove('demo-assistant-highlight');
        });
        removeElement('demo-assistant-panel');
        renderLauncher();
    }

    function currentStep() {
        return steps.find(item => item.id === currentStepId) || steps[0];
    }

    function waitForTarget(selector, timeoutMs) {
        const started = Date.now();
        return new Promise(resolve => {
            const tick = () => {
                const target = document.querySelector(selector);
                if (target || Date.now() - started > timeoutMs) {
                    resolve(target || document.body);
                    return;
                }
                window.setTimeout(tick, 250);
            };
            tick();
        });
    }

    function highlightTarget(target) {
        document.querySelectorAll('.demo-assistant-highlight').forEach(item => {
            item.classList.remove('demo-assistant-highlight');
        });
        if (!target || target === document.body) return;
        target.classList.add('demo-assistant-highlight');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function removeElement(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    function currentPath() {
        return window.location.pathname.replace(/^\/+|\/+$/g, '');
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
