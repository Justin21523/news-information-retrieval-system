// CNIRS frontend internationalization.

(function () {
    const STORAGE_KEY = 'cnirs_lang';
    const DEFAULT_LANG = 'zh-TW';
    const SUPPORTED = ['zh-TW', 'en'];

    const entries = [
        ['回到作品集', 'Back to Portfolio'],
        ['搜尋', 'Search'],
        ['模型對比', 'Model Comparison'],
        ['查詢擴展', 'Query Expansion'],
        ['評估分析', 'Evaluation'],
        ['排序診斷', 'Ranking Diagnostics'],
        ['回饋分析', 'Feedback Analytics'],
        ['語料庫', 'Corpus'],
        ['導覽', 'Guide'],
        ['關於', 'About'],
        ['篩選', 'Filters'],
        ['篩選條件', 'Filters'],
        ['載入篩選選項...', 'Loading filter options...'],
        ['清除所有篩選', 'Clear all filters'],
        ['中文新聞智能檢索系統', 'Chinese News Intelligent Retrieval System'],
        ['檢索模型:', 'Retrieval Model:'],
        ['結果數量:', 'Result Count:'],
        ['運算子:', 'Operator:'],
        ['搜尋中...', 'Searching...'],
        ['結果分析', 'Result Analytics'],
        ['來源分佈', 'Source Distribution'],
        ['類別分佈', 'Category Distribution'],
        ['分數分佈', 'Score Distribution'],
        ['模型效能快速比較', 'Quick Model Performance Comparison'],
        ['執行快速比較', 'Run Quick Compare'],
        ['搜尋結果', 'Search Results'],
        ['匯出為 JSON 格式', 'Export as JSON'],
        ['匯出為 CSV 格式', 'Export as CSV'],
        ['選擇模型 (可多選):', 'Select Models:'],
        ['開始對比', 'Compare Models'],
        ['正在比較模型...', 'Comparing models...'],
        ['比較不同檢索模型的效能與結果', 'Compare retrieval model performance and ranked results'],
        ['評估計算中...', 'Calculating evaluation metrics...'],
        ['圖表顯示選項', 'Chart Options'],
        ['疊加 Precision@K 圖表', 'Overlay Precision@K chart'],
        ['疊加 Recall@K 圖表', 'Overlay Recall@K chart'],
        ['疊加 F1-Score 圖表', 'Overlay F1-Score chart'],
        ['顯示 11-點插值 PR 曲線', 'Show 11-point interpolated PR curve'],
        ['原始 PR 曲線 (Raw PR Curve)', 'Raw PR Curve'],
        ['11-點插值 PR 曲線 (11-Point Interpolated)', '11-Point Interpolated PR Curve'],
        ['檢索系統效能評估', 'Retrieval System Evaluation'],
        ['文檔詳情', 'Document Details'],
        ['載入中...', 'Loading...'],
        ['摘要 Summary', 'Summary'],
        ['關鍵詞 Keywords', 'Keywords'],
        ['完整內容 Full Content', 'Full Content'],
        ['無內容', 'No content'],
        ['未命名文檔', 'Untitled document'],
        ['未知日期', 'Unknown date'],
        ['未分類', 'Uncategorized'],
        ['關鍵句摘要', 'Key Sentence Summary'],
        ['詞頻統計', 'Term Frequency'],
        ['生成摘要', 'Generate Summary'],
        ['正在生成摘要...', 'Generating summary...'],
        ['提取關鍵詞', 'Extract Keywords'],
        ['正在提取關鍵詞...', 'Extracting keywords...'],
        ['生成 KWIC', 'Generate KWIC'],
        ['輸入 KWIC query', 'Enter KWIC query'],
        ['請輸入查詢關鍵字', 'Please enter a query'],
        ['請至少選擇一個模型', 'Please select at least one model'],
        ['搜尋失敗，請稍後再試', 'Search failed. Please try again later.'],
        ['對比失敗，請稍後再試', 'Comparison failed. Please try again later.'],
        ['模型對比結果', 'Model Comparison Results'],
        ['效能比較', 'Performance Comparison'],
        ['模型', 'Model'],
        ['結果數', 'Results'],
        ['響應時間', 'Response Time'],
        ['平均分數', 'Average Score'],
        ['文檔數量', 'Documents'],
        ['詞彙數', 'Terms'],
        ['檢索模型', 'Retrieval Models'],
        ['Demo Readiness', 'Demo Readiness'],
        ['Source Distribution', 'Source Distribution'],
        ['Taxonomy Distribution', 'Taxonomy Distribution'],
        ['Date Coverage', 'Date Coverage'],
        ['Index Cache', 'Index Cache'],
        ['Metadata Completeness', 'Metadata Completeness'],
        ['Facet Quality', 'Facet Quality'],
        ['IR System Flow', 'IR System Flow'],
        ['Topic Explorer', 'Topic Explorer'],
        ['Lightweight clustering', 'Lightweight clustering'],
        ['Analyze Topics', 'Analyze Topics'],
        ['External Data Policy', 'External Data Policy'],
        ['Loading corpus audit...', 'Loading corpus audit...'],
        ['大型語料狀態、metadata 品質、taxonomy facets 與 IR pipeline 可視化', 'Large-corpus readiness, metadata quality, taxonomy facets, and IR pipeline visibility'],
        ['Window:', 'Window:'],
        ['Rows:', 'Rows:'],
        ['Refresh Analytics', 'Refresh Analytics'],
        ['Loading feedback analytics...', 'Loading feedback analytics...'],
        ['Last 7 days', 'Last 7 days'],
        ['Last 30 days', 'Last 30 days'],
        ['Last 90 days', 'Last 90 days'],
        ['Last 365 days', 'Last 365 days'],
        ['Run Diagnostics', 'Run Diagnostics'],
        ['Building ranking diagnostics...', 'Building ranking diagnostics...'],
        ['Document ID:', 'Document ID:'],
        ['Models:', 'Models:'],
        ['Recommended flow', 'Recommended flow'],
        ['Run a complete search engine demo in five minutes', 'Run a complete search engine demo in five minutes'],
        ['Use real news queries, inspect how each ranking model behaves, and show how feedback can become a future Learning-to-Rank feature set.', 'Use real news queries, inspect how each ranking model behaves, and show how feedback can become a future Learning-to-Rank feature set.'],
        ['Start Guided Demo', 'Start Guided Demo'],
        ['Start Search Demo', 'Start Search Demo'],
        ['Compare Models', 'Compare Models'],
        ['Operation Flow', 'Operation Flow'],
        ['Natural Search', 'Natural Search'],
        ['Facets + Metadata', 'Facets + Metadata'],
        ['Why This Result?', 'Why This Result?'],
        ['Evaluation', 'Evaluation'],
        ['Feedback + LTR', 'Feedback + LTR'],
        ['Run Hybrid search', 'Run Hybrid search'],
        ['Inspect corpus facets', 'Inspect corpus facets'],
        ['Open explainable results', 'Open explainable results'],
        ['Compare rankings', 'Compare rankings'],
        ['Run demo evaluation', 'Run demo evaluation'],
        ['Open feedback analytics', 'Open feedback analytics'],
        ['小幫手', 'Demo Helper'],
        ['下一步', 'Next'],
        ['上一步', 'Back'],
        ['重新定位', 'Recenter'],
        ['結束導覽', 'Exit Tour'],
        ['開啟導覽', 'Open Tour'],
        ['收合', 'Collapse'],
        ['步驟', 'Step'],
        ['目前區域', 'Current Area'],
        ['此區塊尚未載入，請等待或重新執行。', 'This section is not loaded yet. Please wait or run the demo action again.'],
        ['語料覆蓋範圍', 'Corpus Coverage'],
        ['來源', 'Sources'],
        ['主題', 'Topics'],
        ['點選即可套用篩選並更新結果', 'Click to apply this filter and update results'],
        ['時間範圍:', 'Window:'],
        ['列數:', 'Rows:'],
        ['最近 7 天', 'Last 7 days'],
        ['最近 30 天', 'Last 30 days'],
        ['最近 90 天', 'Last 90 days'],
        ['最近 365 天', 'Last 365 days'],
        ['重新整理分析', 'Refresh Analytics'],
        ['正在載入回饋分析...', 'Loading feedback analytics...'],
        ['執行診斷', 'Run Diagnostics'],
        ['正在建立排序診斷...', 'Building ranking diagnostics...'],
        ['執行評估', 'Run Evaluation'],
        ['評估設定', 'Evaluation Settings'],
        ['評估指標', 'Evaluation Metrics'],
        ['逐 query 分析', 'Per-Query Breakdown'],
        ['語料庫儀表板', 'Corpus Dashboard'],
        ['Demo 準備狀態', 'Demo Readiness'],
        ['來源分布', 'Source Distribution'],
        ['日期覆蓋', 'Date Coverage'],
        ['Metadata 完整度', 'Metadata Completeness'],
        ['Facet 品質', 'Facet Quality'],
        ['IR 系統流程', 'IR System Flow'],
        ['外部資料策略', 'External Data Policy'],
        ['分析 Topics', 'Analyze Topics'],
        ['輕量 clustering', 'Lightweight clustering'],
        ['開啟操作導覽', 'Open walkthrough'],
        ['體驗完整可解釋新聞檢索流程', 'Try a complete explainable news search flow'],
        ['從真實新聞查詢開始，接著檢視 facets、排序理由、文章詳情、模型對比與評估結果。', 'Start with a real query, then inspect facets, ranking reasons, document detail, comparison, and evaluation.'],
    ];

    const placeholders = [
        ['輸入查詢關鍵字... (例如: 人工智慧、颱風災害、台灣經濟)', 'Enter keywords... e.g. AI, typhoon damage, Taiwan economy'],
        ['輸入查詢關鍵字進行模型對比...', 'Enter a query to compare models...'],
        ['Leave blank to run the selected query set', 'Leave blank to run the selected query set'],
        ['Query for topic exploration, e.g. 半導體 人工智慧', 'Query for topic exploration, e.g. semiconductors AI'],
        ['輸入要探索 topic 的 query，例如：半導體 人工智慧', 'Query for topic exploration, e.g. semiconductors AI'],
        ['Query, e.g. 半導體 or information retrieval', 'Query, e.g. semiconductors or information retrieval'],
        ['輸入 query，例如：半導體 或 information retrieval', 'Query, e.g. semiconductors or information retrieval'],
        ['輸入 KWIC query', 'Enter KWIC query'],
        ['留空會執行選取的 query set', 'Leave blank to run the selected query set'],
    ];

    const zhText = {
        'CNIRS Demo Guide': 'CNIRS 操作導覽',
        'CNIRS Corpus Dashboard': 'CNIRS 語料庫儀表板',
        'Corpus Dashboard - CNIRS': '語料庫儀表板 - CNIRS',
        'Corpus Dashboard': '語料庫儀表板',
        'Ranking Diagnostics - CNIRS': '排序診斷 - CNIRS',
        'Ranking Diagnostics': '排序診斷',
        'Feedback Analytics - CNIRS': '回饋分析 - CNIRS',
        'Feedback Analytics': '回饋分析',
        'Search Logs, User Feedback, and LTR Feature Foundation': '搜尋紀錄、使用者回饋與 Learning-to-Rank 特徵基礎',
        'BM25 / TF-IDF / LM term contribution analysis': 'BM25 / TF-IDF / LM 詞項貢獻分析',
        'Evaluation & Performance Analysis': 'Evaluation 與檢索效能分析',
        'Evaluation Settings': '評估設定',
        'Query Set:': 'Query Set:',
        'Optional single-query override:': '單一查詢覆蓋:',
        'Retrieval Models:': 'Retrieval Models:',
        'Top-K Documents:': 'Top-K 文件數:',
        'Metric K values:': 'Metric K values:',
        'Run Evaluation': '執行評估',
        'Evaluation Metrics': '評估指標',
        'Precision-Recall Curves': 'Precision-Recall 曲線',
        'Model Metrics Table': '模型指標表',
        'Per-Query Breakdown': '逐查詢分析',
        'Model Agreement': '模型一致性',
        'Unique Documents': '各模型獨有文件',
        'Largest Rank Changes': '最大排名變化',
        'No overlap data': '沒有 overlap 資料',
        'No unique document data': '沒有獨有文件資料',
        'Why this rank?': '為什麼這個排名？',
        'Why this result?': '為什麼是這筆結果？',
        'Matched Terms': '命中詞',
        'Expanded Terms': '擴展詞',
        'Component Scores': '組成分數',
        'Field Matches': '欄位命中',
        'Field Boost': '欄位加權',
        'Ranking Features': '排序特徵',
        'Ranking Diagnostics': '排序診斷',
        'Feedback': '回饋',
        'Load diagnostics': '載入診斷',
        'Track click': '記錄點擊',
        'Relevant': '相關',
        'Not relevant': '不相關',
        'Related News': '相關新聞',
        'Metadata / Facets': 'Metadata / Facets',
        'Generate new analysis': '產生新的分析',
        'Demo Guide': 'Demo 導覽',
        'Recommended flow': '建議展示流程',
        'Start Guided Demo': '開始小幫手導覽',
        'Start Search Demo': '開始搜尋 Demo',
        'Run a complete search engine demo in five minutes': '五分鐘完成完整 Search Engine Demo',
        'Use real news queries, inspect how each ranking model behaves, and show how feedback can become a future Learning-to-Rank feature set.': '使用真實新聞查詢，檢視各種 ranking model 的行為，並展示 feedback 如何成為未來 Learning-to-Rank 的特徵基礎。',
        'Compare Models': '比較模型',
        'Operation Flow': '操作流程',
        'Search': '搜尋',
        'Explain': '解釋',
        'Compare': '比較',
        'Explore': '探索',
        'Evaluate': '評估',
        'Query, model, top-k, and facet filters': 'Query、model、top-k 與 facet filters',
        'Matched terms, field boosts, component scores': 'Matched terms、field boosts 與 component scores',
        'Corpus readiness, taxonomy facets, topic clusters': 'Corpus readiness、taxonomy facets 與 topic clusters',
        'Demo qrels, PR curves, diagnostics, feedback logs': 'Demo qrels、PR curves、diagnostics 與 feedback logs',
        'Natural Search': '自然語言查詢',
        'Start with Hybrid for normal language queries. Switch to BM25 or WAND when demonstrating classic lexical ranking and optimization.': '一般查詢先使用 Hybrid；展示傳統 lexical ranking 或 query optimization 時再切換 BM25、WAND 或 MaxScore。',
        'Run the small curated demo qrels. Metrics are labeled as a portfolio smoke evaluation, not a full benchmark.': '執行小型 curated demo qrels。Metrics 會明確標示為作品集 smoke evaluation，不是完整 benchmark。',
        'Run Hybrid search': '執行 Hybrid 搜尋',
        'Inspect corpus facets': '檢視語料 facets',
        'Open explainable results': '開啟可解釋結果',
        'Compare rankings': '比較排序結果',
        'Run demo evaluation': '執行 demo evaluation',
        'Open feedback analytics': '開啟回饋分析',
        'Searchable documents': '可搜尋文件',
        'Vocabulary': '詞彙量',
        'Dataset size': '資料集大小',
        'Audit time': '稽核時間',
        'Cache used': '使用快取',
        'Tokenizer': 'Tokenizer',
        'Avg doc length': '平均文件長度',
        'Manifest': 'Manifest',
        'Lexical cache': 'Lexical cache',
        'Yes': '是',
        'No, built during startup': '否，啟動時建立',
        'Present': '存在',
        'Missing': '缺漏',
        'Model Metrics': '模型指標',
        'Zero-Result Queries': '零結果查詢',
        'Top Queries': '熱門查詢',
        'Relevance Labels': '相關性標籤',
        'Feedback Quality Controls': '回饋品質控制',
        'Recent Feedback': '近期回饋',
        'Learning-to-Rank Feature Preview': 'Learning-to-Rank 特徵預覽',
        'Feature foundation only. This does not train or claim a production ranking model.': '這只是特徵基礎展示，不會訓練或宣稱已部署 production ranking model。',
        'LTR Training Sandbox': 'LTR 訓練沙盒',
        'Weak-supervision demo only. It uses click/relevance-derived labels and does not change production ranking.': '僅為 weak-supervision demo，使用 click/relevance 衍生標籤，不會改變 production ranking。',
        'Training rows': '訓練列數',
        'Train Demo Ranker': '訓練 Demo Ranker',
        'No model metrics yet.': '目前沒有模型指標。',
        'No zero-result queries in this window.': '此時間範圍沒有零結果查詢。',
        'No queries yet.': '目前沒有查詢資料。',
        'No labels yet.': '目前沒有標籤資料。',
        'No feedback events yet.': '目前沒有回饋事件。',
        'Training demo ranker...': '正在訓練 demo ranker...',
        'Feature Weights': '特徵權重',
        'Sample Predictions': '預測範例',
        'No coefficients.': '沒有係數資料。',
        'No predictions.': '沒有預測資料。',
        'Searches': '搜尋次數',
        'Clicks': '點擊',
        'Zero Rate': '零結果率',
        'Latency': '延遲',
        'Time': '時間',
        'Type': '類型',
        'Query': 'Query',
        'Doc': '文件',
        'Label': '標籤',
        'Rank Bucket': '排名區間',
        'Relevance Labels': '相關性標籤',
        'Field Boost': '欄位加權',
        'Feature': '特徵',
        'Coefficient': '係數',
        'Direction': '方向',
        'Predicted': '預測值',
        'Demo evaluation, not full benchmark.': 'Demo evaluation，不是完整 benchmark。',
        'Corpus': '語料庫',
        'Qrels Coverage': 'Qrels 覆蓋率',
        'Resolved Judgments': '已解析 judgments',
        'Cache': '快取',
        'hit': '命中',
        'miss': '未命中',
        'No per-query data.': '沒有逐 query 資料。',
        'No results': '沒有結果',
        'unavailable': '不可用',
        'Doc': '文件',
        'Ranks': '排名',
        'Span': '跨度',
        'Optimization': '最佳化',
        'No component scores': '沒有組成分數',
        'No field matches': '沒有欄位命中',
        'No field boost': '沒有欄位加權',
        'No ranking features': '沒有排序特徵',
        'No direct matched terms': '沒有直接命中詞',
        'Loading diagnostics...': '正在載入診斷...',
        'No diagnostics available': '沒有可用診斷。',
        'Field-Aware Signals': '欄位感知訊號',
        'Field-Aware Ranking Signals': '欄位感知排序訊號',
        'No field contributions.': '沒有欄位貢獻。',
        'Field Match Heatmap': '欄位命中熱圖',
        'No term contributions.': '沒有詞項貢獻。',
        'Term': '詞項',
        'Weight': '權重',
        'Matches': '命中',
        'Boost': '加權',
        'Contribution': '貢獻',
        'Available Analysis': '可用分析',
        'Metadata Signals': 'Metadata 訊號',
        'Relation Reason': '關聯原因',
        'Why this document?': '為什麼是這篇文章？',
        'No summary available.': '沒有可用摘要。',
        'No keywords available.': '沒有可用關鍵詞。',
        'No KWIC query is active.': '目前沒有啟用 KWIC query。',
        'same category': '同分類',
        'same taxonomy': '同 taxonomy',
        'article': '文章',
        'source': '來源',
        'query terms': 'query terms',
        'coverage': '覆蓋率',
        'matched': '命中',
        'missing': '缺漏',
        'doc length': '文件長度',
        'smoothing': '平滑方法',
    };

    const textLookup = new Map();
    entries.forEach(([zh, en]) => {
        textLookup.set(zh, {'zh-TW': zh, en});
        textLookup.set(en, {'zh-TW': zhText[en] || zh, en});
    });
    Object.entries(zhText).forEach(([en, zh]) => {
        textLookup.set(en, {'zh-TW': zh, en});
        textLookup.set(zh, {'zh-TW': zh, en});
    });

    const placeholderLookup = new Map();
    placeholders.forEach(([zh, en]) => {
        placeholderLookup.set(zh, {'zh-TW': zh, en});
        placeholderLookup.set(en, {'zh-TW': zh, en});
    });

    let applying = false;

    function currentLang() {
        const saved = localStorage.getItem(STORAGE_KEY);
        return SUPPORTED.includes(saved) ? saved : DEFAULT_LANG;
    }

    function translate(value, lang = currentLang()) {
        const original = String(value ?? '');
        const trimmed = original.trim();
        const entry = textLookup.get(trimmed);
        if (!entry) return original;
        return original.replace(trimmed, entry[lang] || entry[DEFAULT_LANG] || trimmed);
    }

    function translateAttr(value, lang = currentLang()) {
        const entry = placeholderLookup.get(String(value ?? '').trim()) || textLookup.get(String(value ?? '').trim());
        return entry ? (entry[lang] || entry[DEFAULT_LANG]) : value;
    }

    function applyI18n(root = document.body) {
        if (!root || applying) return;
        applying = true;
        const lang = currentLang();
        document.documentElement.lang = lang;

        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
                if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
                const parent = node.parentElement;
                if (!parent || ['SCRIPT', 'STYLE', 'TEXTAREA'].includes(parent.tagName)) {
                    return NodeFilter.FILTER_REJECT;
                }
                return NodeFilter.FILTER_ACCEPT;
            }
        });

        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);
        nodes.forEach(node => {
            const translated = translate(node.nodeValue, lang);
            if (translated !== node.nodeValue) node.nodeValue = translated;
        });

        root.querySelectorAll?.('[placeholder]').forEach(element => {
            element.setAttribute('placeholder', translateAttr(element.getAttribute('placeholder'), lang));
        });
        root.querySelectorAll?.('[title]').forEach(element => {
            element.setAttribute('title', translateAttr(element.getAttribute('title'), lang));
        });
        root.querySelectorAll?.('[aria-label]').forEach(element => {
            element.setAttribute('aria-label', translateAttr(element.getAttribute('aria-label'), lang));
        });
        root.querySelectorAll?.('[data-i18n]').forEach(element => {
            element.textContent = translate(element.dataset.i18n, lang);
        });
        renderLanguageToggle();
        applying = false;
    }

    function setLang(lang) {
        if (!SUPPORTED.includes(lang)) return;
        localStorage.setItem(STORAGE_KEY, lang);
        applyI18n(document.body);
        window.dispatchEvent(new CustomEvent('cnirs:langchange', { detail: { lang } }));
    }

    function renderLanguageToggle() {
        let toggle = document.getElementById('language-toggle');
        if (!toggle) {
            const nav = document.querySelector('.nav') || document.querySelector('.header');
            if (!nav) return;
            toggle = document.createElement('div');
            toggle.id = 'language-toggle';
            toggle.className = 'language-toggle';
            toggle.innerHTML = `
                <button type="button" data-lang="zh-TW">中文</button>
                <button type="button" data-lang="en">EN</button>
            `;
            nav.appendChild(toggle);
            toggle.querySelectorAll('button').forEach(button => {
                button.addEventListener('click', () => setLang(button.dataset.lang));
            });
        }
        const lang = currentLang();
        toggle.querySelectorAll('button').forEach(button => {
            button.classList.toggle('active', button.dataset.lang === lang);
        });
    }

    function observe() {
        const observer = new MutationObserver(records => {
            if (applying) return;
            records.forEach(record => {
                record.addedNodes.forEach(node => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        applyI18n(node);
                    } else if (node.nodeType === Node.TEXT_NODE && node.parentElement) {
                        const translated = translate(node.nodeValue);
                        if (translated !== node.nodeValue) node.nodeValue = translated;
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    }

    window.CNIRS_I18N = {
        currentLang,
        setLang,
        t: translate,
        applyI18n,
    };

    window.addEventListener('DOMContentLoaded', () => {
        renderLanguageToggle();
        applyI18n(document.body);
        observe();
    });
})();
