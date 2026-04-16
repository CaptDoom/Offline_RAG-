document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-item');
    const topTabs = document.querySelectorAll('.top-tab');
    const views = document.querySelectorAll('.view');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatClearBtn = document.getElementById('chat-clear-btn');
    const chatHistory = document.getElementById('chat-history');
    const runtimePill = document.getElementById('python-runtime');

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderRichText(text) {
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    function switchView(target) {
        views.forEach((view) => view.classList.remove('active'));
        const activeView = document.getElementById(target);
        if (activeView) activeView.classList.add('active');

        navItems.forEach((item) => item.classList.toggle('active', item.dataset.target === target));
        topTabs.forEach((item) => item.classList.toggle('active', item.dataset.target === target));
    }

    navItems.forEach((item) => item.addEventListener('click', (event) => {
        event.preventDefault();
        switchView(item.dataset.target);
    }));

    topTabs.forEach((item) => item.addEventListener('click', (event) => {
        event.preventDefault();
        switchView(item.dataset.target);
    }));

    async function fetchStatus() {
        try {
            const response = await fetch('/api/status');
            const payload = await response.json();
            const data = payload.success ? payload : { index: {}, python_runtime: {}, ollama_active: false };

            document.getElementById('files-found').textContent = data.index.file_count || 0;
            document.getElementById('dataset-stats').textContent = `DATASET: ${data.index.file_count || 0} HIGH-FIDELITY DOCUMENTS PARSED`;
            document.getElementById('total-indexed-count').textContent = data.index.chunk_count || 0;

            if (data.ollama_active) {
                document.getElementById('system-readiness').innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="8"></circle></svg> SYSTEM_READY';
                document.getElementById('system-readiness').style.background = '#ecfdf5';
                document.getElementById('system-readiness').style.color = '#065f46';
            } else {
                document.getElementById('system-readiness').textContent = 'SYSTEM_OFFLINE';
                document.getElementById('system-readiness').style.background = '#fee2e2';
                document.getElementById('system-readiness').style.color = '#991b1b';
            }

            if (runtimePill) {
                const version = data.python_runtime?.version || 'unknown';
                runtimePill.textContent = `PY ${version}`;
                runtimePill.classList.toggle('runtime-pill-warning', !data.python_runtime?.supported);
            }
        } catch (error) {
            console.error('Status fetch failed', error);
        }
    }

    setInterval(fetchStatus, 5000);
    fetchStatus();

    function appendBubble(text, role, rich = false) {
        const bubble = document.createElement('div');
        bubble.className = role === 'assistant' ? 'assistant-bubble' : 'user-bubble';
        if (rich) {
            bubble.innerHTML = renderRichText(text);
        } else {
            bubble.textContent = text;
        }
        chatHistory.appendChild(bubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return bubble;
    }

    async function sendChat() {
        const query = chatInput.value.trim();
        if (!query) return;

        appendBubble(query, 'user');
        chatInput.value = '';
        const assistantBubble = appendBubble('Retrieving local context...', 'assistant');
        assistantBubble.classList.add('loading');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await response.json();

            assistantBubble.classList.remove('loading');
            if (!response.ok || data.success === false) {
                assistantBubble.textContent = `Error: ${data.error || 'Request failed.'}`;
                return;
            }

            assistantBubble.innerHTML = renderRichText(data.answer_markdown || '');
            updateDebugLogs(data.debug_payload || {});
        } catch (error) {
            assistantBubble.classList.remove('loading');
            assistantBubble.textContent = 'Connection error.';
        }
    }

    chatSendBtn.addEventListener('click', sendChat);
    chatInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') sendChat();
    });
    chatClearBtn.addEventListener('click', () => {
        chatHistory.innerHTML = '';
    });

    function updateDebugLogs(debug) {
        if (!debug || !Object.keys(debug).length) return;

        if (debug.embedding_preview) {
            const sliced = debug.embedding_preview.slice(0, 8).map((n) => Number.parseFloat(n).toFixed(4));
            document.getElementById('debug-embedding').textContent = `[ ${sliced.join(', ')} ... ]`;
        }

        if (debug.prompt_text) {
            document.getElementById('debug-prompt').textContent = String(debug.prompt_text).trim();
        }

        const chunksContainer = document.getElementById('debug-chunks-grid');
        chunksContainer.innerHTML = '';
        const retrieved = debug.retrieved_chunks || [];
        document.getElementById('debug-k').textContent = retrieved.length;
        retrieved.forEach((chunk) => {
            const card = document.createElement('div');
            card.className = 'chunk-card';
            card.innerHTML = `
                <div class="chunk-top">
                    <span class="chunk-name">${escapeHtml(chunk.file_name)}</span>
                    <span class="chunk-sim">SIM: ${Number.parseFloat(chunk.score || 0).toFixed(3)}</span>
                </div>
                <div class="chunk-id">CHUNK_ID: ${escapeHtml(chunk.chunk_index)}</div>
                <div class="chunk-text">"...${escapeHtml(chunk.text_preview)}..."</div>
            `;
            chunksContainer.appendChild(card);
        });
    }

    const indexBtn = document.getElementById('index-files-btn');
    indexBtn.addEventListener('click', async () => {
        const path = document.getElementById('folder-path').value.trim();
        const sidebarStatus = document.getElementById('sidebar-index-ready');
        sidebarStatus.textContent = 'INDEXING...';

        const formData = new FormData();
        formData.append('folder_path', path);

        try {
            const response = await fetch('/api/index', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok || data.success === false) {
                sidebarStatus.textContent = 'ERROR';
                alert(data.error || 'Indexing failed.');
                return;
            }
            sidebarStatus.textContent = 'READY';
            fetchStatus();
        } catch (error) {
            sidebarStatus.textContent = 'ERROR';
        }
    });

    const batchRunBtn = document.getElementById('batch-run-btn');
    const batchClearBtn = document.getElementById('batch-clear-btn');
    batchRunBtn.addEventListener('click', async () => {
        const queries = document.getElementById('batch-queries').value.split('\n').filter((query) => query.trim());
        if (!queries.length) return;

        document.getElementById('queue-depth').textContent = queries.length;
        document.getElementById('batch-tbody').innerHTML = '';

        try {
            const response = await fetch('/api/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ queries }),
            });
            const data = await response.json();
            const results = data.results || [];
            window.lastBatchResults = results;

            results.forEach((result) => {
                document.getElementById('batch-tbody').innerHTML += `
                    <tr>
                        <td class="td-id">${escapeHtml(result.id)}</td>
                        <td title="${escapeHtml(result.query)}">${escapeHtml((result.query || '').slice(0, 50))}...</td>
                        <td><span class="status-badge ${String(result.status || '').toLowerCase()}">${escapeHtml(result.status)}</span></td>
                    </tr>
                `;
            });
        } catch (error) {
            console.error('Batch request failed', error);
        }
    });

    batchClearBtn.addEventListener('click', () => {
        document.getElementById('batch-queries').value = '';
        document.getElementById('batch-tbody').innerHTML = '';
        document.getElementById('queue-depth').textContent = '0';
        window.lastBatchResults = null;
    });

    const downloadBtn = document.querySelector('.download-btn');
    downloadBtn.addEventListener('click', () => {
        if (!window.lastBatchResults || window.lastBatchResults.length === 0) {
            alert('No batch results to download.');
            return;
        }
        let csvContent = 'ID,Query,Status,Target,Answer\n';
        window.lastBatchResults.forEach((result) => {
            const escapeCsv = (value) => `"${String(value || '').replace(/"/g, '""')}"`;
            csvContent += `${result.id},${escapeCsv(result.query)},${result.status},${result.target},${escapeCsv(result.answer)}\n`;
        });
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'batch_results.csv';
        link.click();
    });
});
