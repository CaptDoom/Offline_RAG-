document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-item');
    const topTabs = document.querySelectorAll('.top-tab');
    const views = document.querySelectorAll('.view');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatClearBtn = document.getElementById('chat-clear-btn');
    const chatHistory = document.getElementById('chat-history');
    const runtimePill = document.getElementById('python-runtime');
    const imageIndexBtn = document.getElementById('image-index-btn');
    const imageIndexClearBtn = document.getElementById('image-index-clear-btn');
    const imageBrowseFileBtn = document.getElementById('image-browse-file-btn');
    const imageBrowseDirBtn = document.getElementById('image-browse-dir-btn');
    const imageFolderPath = document.getElementById('image-folder-path');
    const imageFilesInput = document.getElementById('image-file-input');
    const imageDirInput = document.getElementById('image-dir-input');
    const imageSearchBtn = document.getElementById('image-search-btn');
    const imageQueryInput = document.getElementById('image-query');
    const imageResultsGrid = document.getElementById('image-results-grid');
    const imageChunkCount = document.getElementById('image-chunk-count');
    const imageIndexStatus = document.getElementById('image-index-status');
    const mainSourceSummary = document.getElementById('main-source-summary');
    const imageSourceSummary = document.getElementById('image-source-summary');
    const folderPathInput = document.getElementById('folder-path');

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

    function detectDataType(file) {
        const name = String(file?.name || '').toLowerCase();
        if (name.endsWith('.pdf')) return 'pdf';
        if (/\.(png|jpg|jpeg|gif|bmp|tiff|tif|webp|svg)$/i.test(name)) return 'image';
        if (name.endsWith('.zip')) return 'archive';
        return 'document';
    }

    function summarizeSources(files, labelOverrides = {}) {
        const summary = {
            total: files.length,
            folders: 0,
            pdfs: 0,
            images: 0,
            archives: 0,
            documents: 0,
        };

        files.forEach((file) => {
            if (file?.webkitRelativePath) summary.folders += 1;
            const type = detectDataType(file);
            if (type === 'pdf') summary.pdfs += 1;
            else if (type === 'image') summary.images += 1;
            else if (type === 'archive') summary.archives += 1;
            else summary.documents += 1;
        });

        return [
            { label: labelOverrides.total || 'TOTAL', value: summary.total },
            { label: labelOverrides.folders || 'FOLDER ITEMS', value: summary.folders },
            { label: labelOverrides.pdfs || 'PDFS', value: summary.pdfs },
            { label: labelOverrides.images || 'IMAGES', value: summary.images },
            { label: labelOverrides.documents || 'DOCS', value: summary.documents },
            { label: labelOverrides.archives || 'ARCHIVES', value: summary.archives },
        ].filter((item) => item.value > 0);
    }

    function renderSourceSummary(container, items) {
        if (!container) return;
        if (!items.length) {
            container.innerHTML = '';
            return;
        }
        container.innerHTML = items
            .map((item) => `<span class="source-pill"><strong>${escapeHtml(item.value)}</strong> ${escapeHtml(item.label)}</span>`)
            .join('');
    }

    function fileListToArray(fileList) {
        return Array.from(fileList || []);
    }

    function updateMainSelectionSummary() {
        const allFiles = [
            ...fileListToArray(mainFileInput?.files),
            ...fileListToArray(mainDirInput?.files),
        ];
        folderPathInput.value = allFiles.length > 0 ? `[${allFiles.length} sources selected]` : '';
        renderSourceSummary(mainSourceSummary, summarizeSources(allFiles));
    }

    function updateImageSelectionSummary() {
        const allFiles = [
            ...fileListToArray(imageFilesInput?.files),
            ...fileListToArray(imageDirInput?.files),
        ];
        imageFolderPath.value = allFiles.length > 0 ? `[${allFiles.length} image sources selected]` : '';
        renderSourceSummary(
            imageSourceSummary,
            summarizeSources(allFiles, { documents: 'NON-IMAGE' }).filter((item) => item.label !== 'DOCS' || item.value > 0),
        );
    }

    function appendStructuredUploads(formData, files, inputKind, indexOffset = 0) {
        const sources = [];
        let currentIndex = indexOffset;
        files.forEach((file) => {
            formData.append('files', file);
            sources.push({
                kind: 'upload',
                input_kind: inputKind,
                data_type: detectDataType(file),
                file_index: currentIndex,
                file_name: file.name,
                relative_path: file.webkitRelativePath || file.name,
                mime_type: file.type || '',
                size: Number(file.size || 0),
            });
            currentIndex += 1;
        });
        return { sources, nextIndex: currentIndex };
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
            document.getElementById('image-chunk-count').textContent = data.image_index?.chunk_count || 0;
            imageIndexStatus.textContent = data.image_index?.exists ? 'READY' : 'OFFLINE';

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

    imageSearchBtn.addEventListener('click', searchImages);
    imageIndexBtn.addEventListener('click', indexImages);
    imageIndexClearBtn.addEventListener('click', () => {
        imageFolderPath.value = '';
        imageFilesInput.value = '';
        if (imageDirInput) imageDirInput.value = '';
        imageResultsGrid.innerHTML = '';
        imageIndexStatus.textContent = 'OFFLINE';
        renderSourceSummary(imageSourceSummary, []);
    });
    imageBrowseFileBtn.addEventListener('click', (event) => {
        event.preventDefault();
        imageFilesInput.click();
    });
    imageBrowseDirBtn.addEventListener('click', (event) => {
        event.preventDefault();
        if (imageDirInput) imageDirInput.click();
    });
    
    imageFilesInput.addEventListener('change', () => {
        updateImageSelectionSummary();
    });
    if (imageDirInput) {
        imageDirInput.addEventListener('change', () => {
            updateImageSelectionSummary();
        });
    }

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

    function renderImageResults(results) {
        imageResultsGrid.innerHTML = '';
        if (!results || results.length === 0) {
            imageResultsGrid.innerHTML = '<div class="empty-state">No image search results found.</div>';
            return;
        }

        results.forEach((result, index) => {
            const card = document.createElement('div');
            card.className = 'chunk-card';
            card.innerHTML = `
                <div class="chunk-top">
                    <span class="chunk-name">${escapeHtml(result.file_name)}</span>
                    <span class="chunk-sim">SIM: ${Number.parseFloat(result.score || 0).toFixed(3)}</span>
                </div>
                <div class="chunk-id">CHUNK_ID: ${escapeHtml(result.chunk_index)}</div>
                <div class="chunk-text">${escapeHtml((result.text || '').slice(0, 450))}</div>
                <div class="chunk-meta"><span>${escapeHtml(result.file_path)}</span></div>
            `;
            imageResultsGrid.appendChild(card);
        });
    }

    async function searchImages() {
        const query = imageQueryInput.value.trim();
        if (!query) return;

        imageResultsGrid.innerHTML = '<div class="empty-state">Searching images...</div>';

        try {
            const response = await fetch('/api/image/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    top_k: Number(document.getElementById('image-top-k').value) || 5,
                    retrieval_mode: document.getElementById('image-retrieval-mode').value,
                    bm25_weight: 0.4,
                }),
            });
            const data = await response.json();
            if (!response.ok || data.success === false) {
                imageResultsGrid.innerHTML = `<div class="empty-state">Error: ${escapeHtml(data.error || 'Search failed.')}</div>`;
                return;
            }
            renderImageResults(data.results || []);
        } catch (error) {
            imageResultsGrid.innerHTML = '<div class="empty-state">Connection error while searching images.</div>';
        }
    }

    async function indexImages() {
        const sidebarStatus = document.getElementById('image-index-status');
        sidebarStatus.textContent = 'INDEXING...';

        const formData = new FormData();
        const imageFiles = [
            ...fileListToArray(imageFilesInput?.files),
            ...fileListToArray(imageDirInput?.files),
        ];
        if (!imageFiles.length) {
            sidebarStatus.textContent = 'OFFLINE';
            alert('Select image files or an image folder first.');
            return;
        }
        const payload = appendStructuredUploads(formData, imageFiles, 'image');
        formData.append('sources', JSON.stringify(payload.sources));

        try {
            const response = await fetch('/api/image/index', { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok || data.success === false) {
                sidebarStatus.textContent = 'ERROR';
                alert(data.error || 'Image indexing failed.');
                return;
            }
            sidebarStatus.textContent = 'READY';
            fetchStatus();
        } catch (error) {
            sidebarStatus.textContent = 'ERROR';
        }
    }

    const indexFilesBtn = document.getElementById('index-files-btn');
    const mainFileInput = document.getElementById('main-file-input');
    const mainDirInput = document.getElementById('main-dir-input');
    const browseFileBtn = document.getElementById('browse-file-btn');
    const browseDirBtn = document.getElementById('browse-dir-btn');
    
    if (browseFileBtn && mainFileInput) {
        browseFileBtn.addEventListener('click', (e) => {
            e.preventDefault();
            mainFileInput.click();
        });
        browseDirBtn.addEventListener('click', (e) => {
            e.preventDefault();
            if (mainDirInput) mainDirInput.click();
        });
        
        mainFileInput.addEventListener('change', () => {
            updateMainSelectionSummary();
        });
        if (mainDirInput) {
            mainDirInput.addEventListener('change', () => {
                updateMainSelectionSummary();
            });
        }
    }

    if (indexFilesBtn) {
        indexFilesBtn.addEventListener('click', async () => {
            const sidebarStatus = document.getElementById('sidebar-index-ready');
            sidebarStatus.textContent = 'INDEXING...';

            const formData = new FormData();
            const mainFiles = [
                ...fileListToArray(mainFileInput?.files),
                ...fileListToArray(mainDirInput?.files),
            ];
            if (!mainFiles.length) {
                sidebarStatus.textContent = 'READY';
                alert('Select files, images, PDFs, or folders first.');
                return;
            }
            const payload = appendStructuredUploads(formData, mainFiles, 'document');
            formData.append('sources', JSON.stringify(payload.sources));

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
    }

    updateMainSelectionSummary();
    updateImageSelectionSummary();

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
