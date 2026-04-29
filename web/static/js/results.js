// ============================================
// LuminaFix Pro - Results Page
// Version gallery with comparison & export
// ============================================

const METHOD_INFO = {
    reinhard: { name: 'Reinhard (LAB)', type: 'classic', color: '#6C8EF2' },
    nilut: { name: 'NILUT', type: 'neural', color: '#FBBF24' },
    nilut_contrast: { name: 'NILUT + CLAHE', type: 'neural', color: '#14b8a6' },
    nilut_tonecurve: { name: 'NILUT + ToneCurve', type: 'neural', color: '#8b5cf6' },
    nilut_tonecurve_sat: { name: 'NILUT + ToneCurve + Sat', type: 'neural', color: '#ec4899' },
    nilut_chroma: { name: 'NILUT + Chroma', type: 'neural', color: '#f97316' },
    xmp_preset: { name: 'XMP Preset', type: 'classic', color: '#a855f7' }
};

function getMethodInfo(methodId) {
    if (METHOD_INFO[methodId]) return METHOD_INFO[methodId];

    const parts = methodId.split('_');
    const lastPart = parts[parts.length - 1];
    const secondLastPart = parts.length >= 2 ? parts[parts.length - 2] : '';

    let baseMethodId = null;
    let timestamp = null;

    if (lastPart === 'latest') {
        baseMethodId = parts.slice(0, -1).join('_');
        timestamp = 'latest';
    } else if (secondLastPart && `${secondLastPart}_${lastPart}`.match(/^\d{8}_\d{6}$/)) {
        baseMethodId = parts.slice(0, -2).join('_');
        timestamp = `${secondLastPart}_${lastPart}`;
    }

    if (baseMethodId) {
        const baseInfo = METHOD_INFO[baseMethodId];
        if (baseInfo) {
            let modelDisplay = '';
            if (timestamp === 'latest') {
                modelDisplay = 'Latest';
            } else {
                try {
                    const year = timestamp.substr(0, 4);
                    const month = timestamp.substr(4, 2);
                    const day = timestamp.substr(6, 2);
                    const hour = timestamp.substr(9, 2);
                    const min = timestamp.substr(11, 2);
                    const date = new Date(year, month - 1, day, hour, min);
                    const monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
                    const hr = date.getHours() % 12 || 12;
                    const ampm = date.getHours() < 12 ? 'AM' : 'PM';
                    modelDisplay = `${monthNames[date.getMonth()]} ${date.getDate()}, ${hr}:${date.getMinutes().toString().padStart(2,'0')} ${ampm}`;
                } catch (e) { modelDisplay = timestamp; }
            }
            return { name: `${baseInfo.name} (${modelDisplay})`, type: baseInfo.type, color: baseInfo.color };
        }
    }

    return { name: methodId.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), type: 'unknown', color: '#9ca3af' };
}

let state = {
    results: [],
    baselineResults: [],
    currentIndex: 0,
    currentRefIndex: 0,
    references: [],
    isProcessing: false,
    isPreprocessing: false,
    debounceTimer: null,
    comparisonMode: false,
    selectedStrengths: [90],
    strengthCache: {},
    perImageStrengths: {} // key: "refName" -> strength value (for per-image overrides)
};

// ============================================
// Helpers
// ============================================
function getSelectedStrengths() {
    const slider = document.getElementById('color-strength');
    return slider ? [parseInt(slider.value)] : [90];
}

function buildCacheKey(targetFilename, refName, strength, settings) {
    return `${targetFilename}_${refName}_${strength}_${settings.luminanceStrength}_${settings.skinProtection}_${settings.neonProtection}_${settings.lipProtection}`;
}

function getBaseMethodId(methodId) {
    const parts = methodId.split('_');
    const lastPart = parts[parts.length - 1];
    const secondLastPart = parts.length >= 2 ? parts[parts.length - 2] : '';
    if (lastPart === 'latest') return parts.slice(0, -1).join('_');
    if (secondLastPart && `${secondLastPart}_${lastPart}`.match(/^\d{8}_\d{6}$/)) return parts.slice(0, -2).join('_');
    return methodId;
}

function findMethodOutput(methodsMap, methodId) {
    if (!methodsMap) return null;
    if (methodsMap[methodId]) return methodsMap[methodId];
    const targetBase = getBaseMethodId(methodId);
    for (const key in methodsMap) {
        if (getBaseMethodId(key) === targetBase && methodsMap[key]) return methodsMap[key];
    }
    return null;
}

function extractNilutModelIds(results) {
    const modelIds = new Set();
    results.forEach(result => {
        Object.keys(result.referenceResults).forEach(refName => {
            const methods = result.referenceResults[refName].methods || {};
            Object.keys(methods).forEach(methodId => {
                if (!methodId.startsWith('nilut')) return;
                const parts = methodId.split('_');
                const lastPart = parts[parts.length - 1];
                const secondLastPart = parts.length >= 2 ? parts[parts.length - 2] : '';
                if (lastPart === 'latest') modelIds.add('latest');
                else if (secondLastPart && `${secondLastPart}_${lastPart}`.match(/^\d{8}_\d{6}$/)) modelIds.add(`${secondLastPart}_${lastPart}`);
            });
        });
    });
    return modelIds.size > 0 ? JSON.stringify(Array.from(modelIds)) : '["latest"]';
}

function getCurrentSettings() {
    return {
        luminanceStrength: document.getElementById('luminance-strength').value,
        skinProtection: document.getElementById('skin-protection').checked,
        neonProtection: document.getElementById('neon-protection').checked,
        lipProtection: document.getElementById('lip-protection').checked
    };
}

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', async () => {
    const storedResults = sessionStorage.getItem('styleTransferResults');
    const storedSettings = sessionStorage.getItem('styleTransferSettings');

    if (!storedResults) {
        alert('No results found. Please process images first.');
        window.location.href = '/';
        return;
    }

    state.results = JSON.parse(storedResults);
    state.baselineResults = JSON.parse(JSON.stringify(state.results));
    state.currentIndex = 0;

    // Clean up results
    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard', 'nilut'];

    state.results.forEach(result => {
        Object.keys(result.referenceResults).forEach(refName => {
            const refData = result.referenceResults[refName];
            const allMethodIds = [...Object.keys(refData.methods || {}), ...Object.keys(refData.errors || {})];
            allMethodIds.forEach(methodId => {
                const baseMethodId = getBaseMethodId(methodId);
                if (!selectedMethods.includes(baseMethodId)) {
                    delete refData.methods[methodId];
                    delete refData.errors[methodId];
                }
            });
        });
    });
    sessionStorage.setItem('styleTransferResults', JSON.stringify(state.results));

    // Restore settings
    let initialStrength = 90;
    if (storedSettings) {
        const settings = JSON.parse(storedSettings);

        // Restore color strength slider
        if (settings.selectedStrengths && settings.selectedStrengths.length > 0) {
            initialStrength = settings.selectedStrengths[0];
        } else if (settings.colorStrength) {
            initialStrength = parseInt(settings.colorStrength);
        }
        const colorSlider = document.getElementById('color-strength');
        if (colorSlider) {
            colorSlider.value = initialStrength;
            document.getElementById('color-strength-value').textContent = initialStrength + '%';
        }

        document.getElementById('luminance-strength').value = settings.luminanceStrength || 0;
        document.getElementById('skin-protection').checked = settings.skinProtection !== false;
        document.getElementById('neon-protection').checked = settings.neonProtection !== false;
        document.getElementById('lip-protection').checked = settings.lipProtection || false;
        document.getElementById('luminance-value').textContent = (settings.luminanceStrength || 0) + '%';
    }

    state.selectedStrengths = [initialStrength];
    state.baselineSettings = {
        selectedStrengths: [...state.selectedStrengths],
        luminanceStrength: document.getElementById('luminance-strength').value,
        skinProtection: document.getElementById('skin-protection').checked,
        neonProtection: document.getElementById('neon-protection').checked,
        lipProtection: document.getElementById('lip-protection').checked
    };

    // Seed cache
    const initSettings = getCurrentSettings();
    state.results.forEach(result => {
        const targetFn = result.filename || result.inputUrl.split('/').pop();
        Object.keys(result.referenceResults).forEach(refName => {
            const cacheKey = buildCacheKey(targetFn, refName, initialStrength, initSettings);
            state.strengthCache[cacheKey] = {
                methods: { ...result.referenceResults[refName].methods },
                errors: { ...result.referenceResults[refName].errors }
            };
        });
    });

    // Update version count
    const result = state.results[0];
    const refCount = Object.keys(result.referenceResults).length;
    const versionCountEl = document.getElementById('version-count');
    if (versionCountEl) versionCountEl.textContent = `${refCount} version${refCount !== 1 ? 's' : ''}`;

    await loadReferences();
    showCurrentResult();
    setupDownloadAll();
    setupExportZip();
    setupLiveSettings();
    setupComparisonMode();

    if (state.selectedStrengths.length > 1 || (state.selectedStrengths.length === 1 && state.selectedStrengths[0] !== initialStrength)) {
        reprocessAllStrengths();
    }
});

async function loadReferences() {
    try {
        const response = await fetch('/api/references');
        const data = await response.json();
        state.references = data.references;
    } catch (error) {
        console.error('Failed to load references:', error);
    }
}

// ============================================
// Live Settings
// ============================================
function setupLiveSettings() {
    // Color strength slider (universal)
    const colorSlider = document.getElementById('color-strength');
    const colorValue = document.getElementById('color-strength-value');
    colorSlider.addEventListener('input', (e) => {
        colorValue.textContent = e.target.value + '%';
    });
    colorSlider.addEventListener('change', (e) => {
        const newStrength = parseInt(e.target.value);
        state.selectedStrengths = [newStrength];
        // Reset per-image overrides when universal changes
        state.perImageStrengths = {};
        debouncedReprocess();
    });

    document.getElementById('luminance-strength').addEventListener('input', (e) => {
        document.getElementById('luminance-value').textContent = e.target.value + '%';
        state.strengthCache = {};
        state.isPreprocessing = false;
        state.perImageStrengths = {};
        debouncedReprocess();
    });

    ['skin-protection', 'neon-protection', 'lip-protection'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            state.strengthCache = {};
            state.isPreprocessing = false;
            state.perImageStrengths = {};
            debouncedReprocess();
        });
    });
}

function updateComparisonToggleState() {
    // Single slider now, comparison is always available
}

function setupComparisonMode() {
    const toggle = document.getElementById('comparison-mode-toggle');
    const baselineInfo = document.getElementById('baseline-info');
    const resetBtn = document.getElementById('reset-baseline-btn');

    toggle.addEventListener('change', (e) => {
        state.comparisonMode = e.target.checked;
        if (e.target.checked) {
            baselineInfo.style.display = 'block';
            updateSettingsDisplay();
        } else {
            baselineInfo.style.display = 'none';
        }

        const result = state.results[state.currentIndex];
        const baselineResult = state.baselineResults[state.currentIndex];
        const refNames = Object.keys(result.referenceResults);
        let hasChanges = false;
        for (const refName of refNames) {
            const currentMethods = result.referenceResults[refName].methods;
            const baselineMethods = baselineResult.referenceResults[refName].methods;
            for (const methodId in currentMethods) {
                if (currentMethods[methodId]?.url !== baselineMethods[methodId]?.url) { hasChanges = true; break; }
            }
            if (hasChanges) break;
        }

        if (e.target.checked && !hasChanges) {
            showStatus('Adjust parameters to see comparison.', 'var(--warning)', 4000);
        }
        showCurrentResult();
    });

    resetBtn.addEventListener('click', () => {
        state.baselineResults = JSON.parse(JSON.stringify(state.results));
        state.baselineSettings = {
            selectedStrengths: [...state.selectedStrengths],
            luminanceStrength: document.getElementById('luminance-strength').value,
            skinProtection: document.getElementById('skin-protection').checked,
            neonProtection: document.getElementById('neon-protection').checked,
            lipProtection: document.getElementById('lip-protection').checked
        };
        updateSettingsDisplay();
        showCurrentResult();
        showStatus('Baseline reset', 'var(--success)', 2000);
    });

    updateComparisonToggleState();
}

function updateSettingsDisplay() {
    if (!state.baselineSettings) return;
    const baselineDisplay = document.getElementById('baseline-settings-display');
    const currentDisplay = document.getElementById('current-settings-display');
    const baselineStrength = (state.baselineSettings.selectedStrengths || [90])[0];
    const currentStrength = state.selectedStrengths[0];
    baselineDisplay.textContent = `Color: ${baselineStrength}%, Lum: ${state.baselineSettings.luminanceStrength}%`;
    currentDisplay.textContent = `Color: ${currentStrength}%, Lum: ${document.getElementById('luminance-strength').value}%`;
    const isDifferent = baselineStrength !== currentStrength ||
                        state.baselineSettings.luminanceStrength !== document.getElementById('luminance-strength').value;
    currentDisplay.style.color = isDifferent ? 'var(--success)' : 'var(--warning)';
}

function showStatus(message, color, timeout) {
    const status = document.getElementById('reprocess-status');
    status.className = 'reprocess-status';
    status.textContent = message;
    status.style.color = color || '';
    if (timeout) setTimeout(() => { status.textContent = ''; }, timeout);
}

function setProcessingBanner(active, text) {
    const banner = document.getElementById('processing-banner');
    const bannerText = document.getElementById('processing-banner-text');
    const main = document.getElementById('results-main');
    if (!banner) return;
    if (active) {
        if (text && bannerText) bannerText.textContent = text;
        banner.classList.add('active');
        if (main) main.classList.add('processing');
    } else {
        banner.classList.remove('active');
        if (main) main.classList.remove('processing');
    }
}

function debouncedReprocess() {
    if (state.debounceTimer) clearTimeout(state.debounceTimer);
    const status = document.getElementById('reprocess-status');
    status.textContent = 'Settings changed...';
    status.style.color = '';
    setProcessingBanner(true, 'Updating — settings changed...');
    state.debounceTimer = setTimeout(() => reprocessAllStrengths(), 500);
}

// ============================================
// Reprocessing
// ============================================
async function reprocessAllStrengths() {
    if (state.references.length === 0) { setProcessingBanner(false); return; }
    if (state.isProcessing) { state.pendingReprocess = true; return; }
    if (state.selectedStrengths.length === 0) {
        setProcessingBanner(false);
        showStatus('Select at least one color strength.', 'var(--warning)', 3000);
        return;
    }

    state.isProcessing = true;
    const status = document.getElementById('reprocess-status');
    const result = state.results[state.currentIndex];
    const settings = getCurrentSettings();

    sessionStorage.setItem('styleTransferSettings', JSON.stringify({
        selectedStrengths: state.selectedStrengths,
        luminanceStrength: settings.luminanceStrength,
        skinProtection: settings.skinProtection,
        neonProtection: settings.neonProtection,
        lipProtection: settings.lipProtection
    }));

    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard', 'nilut'];
    const nilutMode = sessionStorage.getItem('nilutMode') || 'universal';
    const nilutModels = sessionStorage.getItem('nilutModels') || extractNilutModelIds(state.results);
    const targetFilename = result.filename || result.inputUrl.split('/').pop();
    const refNames = Object.keys(result.referenceResults);

    const tasks = [];
    for (const strength of state.selectedStrengths) {
        for (const refName of refNames) {
            const cacheKey = buildCacheKey(targetFilename, refName, strength, settings);
            if (!state.strengthCache[cacheKey]) tasks.push({ strength, refName, cacheKey });
        }
    }

    if (tasks.length === 0) {
        updateResultsFromCache();
        showCurrentResult();
        state.isProcessing = false;
        setProcessingBanner(false);
        showStatus('Updated!', 'var(--success)', 2000);
        return;
    }

    let completed = 0;
    status.className = 'reprocess-status processing';
    status.innerHTML = `<span class="spinner-small"></span> Processing 0/${tasks.length}...`;
    setProcessingBanner(true, `Processing 0/${tasks.length}...`);

    const CONCURRENCY = 3;
    let taskIndex = 0;

    async function processNext() {
        while (taskIndex < tasks.length) {
            const task = tasks[taskIndex++];
            const refData = result.referenceResults[task.refName];

            const formData = new FormData();
            formData.append('target_filename', targetFilename);
            formData.append('reference_filename', refData.refFilename);
            formData.append('color_strength', task.strength / 100);
            formData.append('luminance_strength', settings.luminanceStrength / 100);
            formData.append('skin_protection', settings.skinProtection);
            formData.append('neon_protection', settings.neonProtection);
            formData.append('lip_protection', settings.lipProtection);
            formData.append('methods', JSON.stringify(selectedMethods));
            formData.append('nilut_mode', nilutMode);
            formData.append('nilut_models', nilutModels);

            try {
                const response = await fetch('/api/process-all', { method: 'POST', body: formData });
                if (!response.ok) throw new Error('Processing failed');
                const data = await response.json();
                const cached = { methods: data.outputs || {}, errors: data.errors || {} };
                for (const methodId in cached.methods) {
                    if (cached.methods[methodId]?.url) cached.methods[methodId].url += '?t=' + Date.now();
                }
                state.strengthCache[task.cacheKey] = cached;
            } catch (error) {
                state.strengthCache[task.cacheKey] = { methods: {}, errors: { _all: { error: 'Request failed' } } };
            }

            completed++;
            status.innerHTML = `<span class="spinner-small"></span> Processing ${completed}/${tasks.length}...`;
            const bannerText = document.getElementById('processing-banner-text');
            if (bannerText) bannerText.textContent = `Processing ${completed}/${tasks.length}...`;
            updateResultsFromCache();
            showCurrentResult();
        }
    }

    const workers = [];
    for (let i = 0; i < Math.min(CONCURRENCY, tasks.length); i++) workers.push(processNext());
    await Promise.all(workers);

    updateResultsFromCache();
    sessionStorage.setItem('styleTransferResults', JSON.stringify(state.results));
    state.isProcessing = false;
    setProcessingBanner(false);
    showStatus('Updated!', 'var(--success)', 2000);

    if (state.pendingReprocess) {
        state.pendingReprocess = false;
        reprocessAllStrengths();
    }
}

function updateResultsFromCache() {
    if (state.selectedStrengths.length === 0) return;
    const firstStrength = state.selectedStrengths[0];
    const settings = getCurrentSettings();
    const result = state.results[state.currentIndex];
    const targetFilename = result.filename || result.inputUrl.split('/').pop();

    Object.keys(result.referenceResults).forEach(refName => {
        const cacheKey = buildCacheKey(targetFilename, refName, firstStrength, settings);
        const cached = state.strengthCache[cacheKey];
        if (cached) {
            result.referenceResults[refName].methods = { ...cached.methods };
            result.referenceResults[refName].errors = { ...cached.errors };
        }
    });
}

// ============================================
// Render Results
// ============================================
function showCurrentResult() {
    const result = state.results[state.currentIndex];
    const baselineResult = state.baselineResults[state.currentIndex];
    const refNames = Object.keys(result.referenceResults);

    document.getElementById('original-image').src = result.inputUrl;
    document.getElementById('image-name').textContent = result.originalName;

    const grid = document.getElementById('results-grid');
    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedBaseMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard'];

    // Collect method IDs
    const allMethodIds = new Set();
    refNames.forEach(refName => {
        const refData = result.referenceResults[refName];
        Object.keys(refData.methods || {}).forEach(id => allMethodIds.add(id));
        Object.keys(refData.errors || {}).forEach(id => allMethodIds.add(id));
    });

    const targetFilename = result.filename || result.inputUrl.split('/').pop();
    if (state.selectedStrengths.length > 1) {
        const settings = getCurrentSettings();
        for (const strength of state.selectedStrengths) {
            for (const refName of refNames) {
                const cacheKey = buildCacheKey(targetFilename, refName, strength, settings);
                const cached = state.strengthCache[cacheKey];
                if (cached) {
                    Object.keys(cached.methods || {}).forEach(id => allMethodIds.add(id));
                    Object.keys(cached.errors || {}).forEach(id => allMethodIds.add(id));
                }
            }
        }
    }

    let methods = Array.from(allMethodIds).filter(methodId => selectedBaseMethods.includes(getBaseMethodId(methodId)));
    const seenBases = new Map();
    methods.forEach(methodId => {
        const base = getBaseMethodId(methodId);
        if (!seenBases.has(base)) seenBases.set(base, methodId);
    });
    methods = Array.from(seenBases.values());

    if (methods.length === 0) {
        grid.innerHTML = '<p class="no-results">No results available.</p>';
        return;
    }

    grid.style.setProperty('--method-count', methods.length);
    const isMultiStrength = false; // Single slider now

    // Header row
    let html = `
        <div class="grid-header-row ${state.comparisonMode && !isMultiStrength ? 'comparison-mode' : ''}">
            <div class="grid-corner">Reference</div>
            ${methods.map(methodId => {
                const info = getMethodInfo(methodId);
                if (state.comparisonMode && !isMultiStrength) {
                    return `<div class="grid-header-cell comparison-header">
                        <div class="method-header-info">
                            <span class="method-name">${info.name}</span>
                            <span class="method-badge ${info.type}">${info.type}</span>
                        </div>
                        <div class="comparison-labels">
                            <span class="baseline-label">Baseline</span>
                            <span class="tweaked-label">Tweaked</span>
                        </div>
                    </div>`;
                }
                return `<div class="grid-header-cell">
                    <span class="method-name">${info.name}</span>
                    <span class="method-badge ${info.type}">${info.type}</span>
                </div>`;
            }).join('')}
        </div>
    `;

    const settings = getCurrentSettings();

    // Rows
    refNames.forEach(refName => {
        const refData = result.referenceResults[refName];
        html += `<div class="grid-row">`;

        // Ref cell
        html += `<div class="grid-ref-cell">
            ${refData.refUrl ? `<img src="${refData.refUrl}" alt="${refName}" class="ref-thumb">` : `<div class="ref-thumb xmp-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>`}
            <span class="ref-name">${refName}</span>
        </div>`;

        // Method cells
        methods.forEach(methodId => {
            const methodInfo = getMethodInfo(methodId);

            if (isMultiStrength) {
                html += `<div class="grid-cell multi-strength-cell" data-method="${methodInfo.name}"><div class="strength-strip">`;
                html += `<div class="strength-strip-item strength-strip-original">
                    <div class="strength-label strength-label--original">Original</div>
                    <img src="${result.inputUrl}" alt="Original" onclick="openFullscreen('${result.inputUrl}')" class="strength-image">
                </div>`;

                for (const strength of state.selectedStrengths) {
                    const cacheKey = buildCacheKey(targetFilename, refName, strength, settings);
                    const cached = state.strengthCache[cacheKey];
                    const output = cached ? findMethodOutput(cached.methods, methodId) : null;
                    const error = cached ? (findMethodOutput(cached.errors, methodId) || cached.errors['_all']) : null;

                    if (!cached) {
                        html += `<div class="strength-strip-item loading"><div class="strength-label">${strength}%</div><span class="spinner-small"></span></div>`;
                    } else if (error) {
                        html += `<div class="strength-strip-item error"><div class="strength-label">${strength}%</div><div class="strength-error">${error.error || 'Error'}</div></div>`;
                    } else if (output) {
                        html += `<div class="strength-strip-item">
                            <div class="strength-label">${strength}%</div>
                            <img src="${output.url}" alt="${strength}%" onclick="openFullscreen('${output.url}')" class="strength-image">
                            <button class="btn-icon-tiny" onclick="downloadImage('${output.url}', '${methodId}_${refName}_${strength}pct_${result.originalName}')" title="Download">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                            </button>
                        </div>`;
                    } else {
                        html += `<div class="strength-strip-item"><div class="strength-label">${strength}%</div><div class="strength-error">N/A</div></div>`;
                    }
                }
                html += `</div></div>`;
            } else {
                const output = findMethodOutput(refData.methods, methodId);
                const error = findMethodOutput(refData.errors, methodId);
                const baselineRefData = state.comparisonMode ? baselineResult.referenceResults[refName] : null;
                const baselineOutput = baselineRefData ? findMethodOutput(baselineRefData.methods, methodId) : null;
                const baselineError = baselineRefData ? findMethodOutput(baselineRefData.errors, methodId) : null;

                if (error && (!state.comparisonMode || baselineError)) {
                    html += `<div class="grid-cell grid-cell--error">
                        <div class="error-icon-small"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                        <p class="error-text">${error.error || 'Error'}</p>
                    </div>`;
                } else if (!output && (!state.comparisonMode || !baselineOutput)) {
                    html += `<div class="grid-cell grid-cell--unavailable"><p>N/A</p></div>`;
                } else if (state.comparisonMode && baselineOutput && output) {
                    const urlsMatch = baselineOutput.url === output.url;
                    html += `<div class="grid-cell comparison-cell">
                        ${urlsMatch ? '<div class="comparison-warning">Same - adjust settings first</div>' : ''}
                        <div class="comparison-slider-wrapper" data-comparison-slider="${methodId}-${refName}">
                            <img class="comparison-img-baseline" src="${baselineOutput.url}" alt="Baseline">
                            <img class="comparison-img-tweaked" src="${output.url}" alt="Tweaked">
                            <div class="comparison-slider-handle"><div class="comparison-slider-line"></div>
                                <div class="comparison-slider-circle"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/><polyline points="9 18 15 12 9 6" transform="translate(6,0)"/></svg></div>
                            </div>
                            <div class="comparison-labels-overlay"><span class="baseline-side-label">Baseline</span><span class="tweaked-side-label">Tweaked</span></div>
                        </div>
                        <div class="cell-actions">
                            <button class="btn-icon btn-hold-comparison" data-slider="${methodId}-${refName}" title="Toggle view">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                            </button>
                            <button class="btn-icon btn-toggle-comparison" data-slider="${methodId}-${refName}" title="Restore slider">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
                            </button>
                            <div class="btn-separator"></div>
                            <button class="btn-icon" onclick="downloadImage('${baselineOutput.url}', 'baseline_${methodId}_${refName}_${result.originalName}')" title="Download Baseline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg><span class="btn-label-small">Base</span></button>
                            <button class="btn-icon" onclick="downloadImage('${output.url}', 'tweaked_${methodId}_${refName}_${result.originalName}')" title="Download Tweaked"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg><span class="btn-label-small">Tweak</span></button>
                            <button class="btn-icon" onclick="openFullscreen('${output.url}')" title="Fullscreen"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg></button>
                        </div>
                    </div>`;
                } else if (!state.comparisonMode && output) {
                    let beforeUrl = result.inputUrl;
                    const baseId = getBaseMethodId(methodId);
                    if (['nilut_contrast','nilut_tonecurve','nilut_tonecurve_sat'].includes(baseId)) {
                        const nilutOutput = findMethodOutput(refData.methods, 'nilut');
                        if (nilutOutput?.url) beforeUrl = nilutOutput.url;
                    }

                    html += `<div class="grid-cell" data-method="${methodInfo.name}">
                        <div class="slider-wrapper" data-slider="${methodId}-${refName}">
                            <img class="img-before" src="${beforeUrl}" alt="Original">
                            <img class="img-after" src="${output.url}" alt="${methodInfo.name}">
                        </div>
                        <div class="cell-actions">
                            <button class="btn-icon btn-compare" data-slider="${methodId}-${refName}" title="Hold to see original">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                            </button>
                            <button class="btn-icon" onclick="downloadImage('${output.url}', '${methodId}_${refName}_${result.originalName}')" title="Download"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>
                            <button class="btn-icon" onclick="openFullscreen('${output.url}')" title="Fullscreen"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg></button>
                        </div>
                        <div class="per-image-strength">
                            <label>Strength</label>
                            <input type="range" min="0" max="200" value="${state.perImageStrengths[refName] || state.selectedStrengths[0]}"
                                   data-ref="${refName}" data-method="${methodId}"
                                   oninput="updatePerImageStrengthLabel(this)"
                                   onchange="reprocessSingleImage(this)">
                            <span class="per-strength-val">${state.perImageStrengths[refName] || state.selectedStrengths[0]}%</span>
                        </div>
                    </div>`;
                }
            }
        });

        html += `</div>`;
    });

    grid.innerHTML = html;
    setupSliders();
    setupComparisonSliders();
    setupCompareDropdowns();
    setupCompareButtons();
    setupComparisonButtons();
}

// ============================================
// Slider Setup
// ============================================
function setupSliders() {
    document.querySelectorAll('.slider-wrapper').forEach(slider => {
        let isDragging = false;
        let currentPosition = 50;
        const handle = slider.querySelector('.slider-handle');
        const beforeImg = slider.querySelector('.img-before');
        if (!handle || !beforeImg) return;

        const updatePosition = (clientX) => {
            const rect = slider.getBoundingClientRect();
            let pos = (clientX - rect.left) / rect.width;
            pos = Math.max(0, Math.min(1, pos));
            currentPosition = pos * 100;
            handle.style.left = currentPosition + '%';
            beforeImg.style.clipPath = `inset(0 ${100 - currentPosition}% 0 0)`;
        };

        slider.addEventListener('mousedown', (e) => { if (!slider.classList.contains('compare-active')) return; isDragging = true; updatePosition(e.clientX); e.preventDefault(); });
        document.addEventListener('mousemove', (e) => { if (isDragging && slider.classList.contains('compare-active')) updatePosition(e.clientX); });
        document.addEventListener('mouseup', () => isDragging = false);
        slider.addEventListener('touchstart', (e) => { if (!slider.classList.contains('compare-active')) return; isDragging = true; updatePosition(e.touches[0].clientX); e.preventDefault(); });
        document.addEventListener('touchmove', (e) => { if (isDragging && slider.classList.contains('compare-active')) updatePosition(e.touches[0].clientX); });
        document.addEventListener('touchend', () => isDragging = false);

        slider._sliderState = {
            getCurrentPosition: () => currentPosition,
            setPosition: (pos) => { currentPosition = pos; handle.style.left = pos + '%'; beforeImg.style.clipPath = `inset(0 ${100 - pos}% 0 0)`; }
        };
    });
}

function setupComparisonSliders() {
    document.querySelectorAll('.comparison-slider-wrapper').forEach(slider => {
        let isDragging = false;
        let currentPosition = 50;
        const handle = slider.querySelector('.comparison-slider-handle');
        const baselineImg = slider.querySelector('.comparison-img-baseline');
        if (!handle || !baselineImg) return;

        const updatePosition = (clientX) => {
            const rect = slider.getBoundingClientRect();
            let pos = (clientX - rect.left) / rect.width;
            pos = Math.max(0, Math.min(1, pos));
            currentPosition = pos * 100;
            handle.style.left = currentPosition + '%';
            baselineImg.style.clipPath = `inset(0 ${100 - currentPosition}% 0 0)`;
        };

        slider.addEventListener('mousedown', (e) => { if (slider.classList.contains('button-mode-active')) return; isDragging = true; updatePosition(e.clientX); e.preventDefault(); });
        document.addEventListener('mousemove', (e) => { if (isDragging) updatePosition(e.clientX); });
        document.addEventListener('mouseup', () => isDragging = false);
        slider.addEventListener('touchstart', (e) => { if (slider.classList.contains('button-mode-active')) return; isDragging = true; updatePosition(e.touches[0].clientX); e.preventDefault(); });
        document.addEventListener('touchmove', (e) => { if (isDragging) updatePosition(e.touches[0].clientX); });
        document.addEventListener('touchend', () => isDragging = false);

        slider._sliderState = {
            getCurrentPosition: () => currentPosition,
            setPosition: (pos) => { currentPosition = pos; handle.style.left = pos + '%'; baselineImg.style.clipPath = `inset(0 ${100 - pos}% 0 0)`; }
        };
    });
}

function setupComparisonButtons() {
    document.querySelectorAll('.btn-toggle-comparison').forEach(btn => {
        const sliderId = btn.dataset.slider;
        const slider = document.querySelector(`.comparison-slider-wrapper[data-comparison-slider="${sliderId}"]`);
        if (!slider) return;
        const handle = slider.querySelector('.comparison-slider-handle');
        const labels = slider.querySelector('.comparison-labels-overlay');
        const eyeBtn = document.querySelector(`.btn-hold-comparison[data-slider="${sliderId}"]`);

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            slider.classList.remove('button-mode-active');
            btn.classList.remove('active');
            if (eyeBtn) eyeBtn.classList.remove('active');
            if (handle) handle.style.display = 'block';
            if (labels) labels.style.display = 'flex';
            slider.style.cursor = 'ew-resize';
            if (slider._sliderState) slider._sliderState.setPosition(50);
        });
    });

    document.querySelectorAll('.btn-hold-comparison').forEach(btn => {
        const sliderId = btn.dataset.slider;
        const slider = document.querySelector(`.comparison-slider-wrapper[data-comparison-slider="${sliderId}"]`);
        if (!slider) return;
        const baselineImg = slider.querySelector('.comparison-img-baseline');
        const handle = slider.querySelector('.comparison-slider-handle');
        const labels = slider.querySelector('.comparison-labels-overlay');
        if (!baselineImg || !handle) return;

        let showingBaseline = false;
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            showingBaseline = !showingBaseline;
            btn.classList.toggle('active', showingBaseline);
            slider.classList.add('button-mode-active');
            baselineImg.style.clipPath = showingBaseline ? 'inset(0 0 0 0)' : 'inset(0 100% 0 0)';
            handle.style.display = 'none';
            if (labels) labels.style.display = 'none';
            slider.style.cursor = 'default';
        });
    });
}

function setupCompareDropdowns() {
    document.querySelectorAll('.btn-compare-trigger').forEach(trigger => {
        const dropdown = trigger.closest('.compare-dropdown');
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.compare-dropdown.open').forEach(d => { if (d !== dropdown) d.classList.remove('open'); });
            dropdown.classList.toggle('open');
        });
    });
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.compare-dropdown')) document.querySelectorAll('.compare-dropdown.open').forEach(d => d.classList.remove('open'));
    });
}

function setupCompareButtons() {
    document.querySelectorAll('.btn-compare').forEach(btn => {
        const sliderId = btn.dataset.slider;
        const slider = document.querySelector(`.slider-wrapper[data-slider="${sliderId}"]`);
        if (!slider) return;
        const beforeImg = slider.querySelector('.img-before');
        if (!beforeImg) return;

        const showOriginal = () => { slider.classList.add('hold-compare'); beforeImg.style.clipPath = 'inset(0 0 0 0)'; };
        const showEdited = () => {
            slider.classList.remove('hold-compare');
            if (slider.classList.contains('compare-active') && slider._sliderState) {
                const pos = slider._sliderState.getCurrentPosition();
                beforeImg.style.clipPath = `inset(0 ${100 - pos}% 0 0)`;
            } else {
                beforeImg.style.clipPath = 'inset(0 100% 0 0)';
            }
        };

        btn.addEventListener('mousedown', (e) => { e.preventDefault(); showOriginal(); });
        btn.addEventListener('mouseup', showEdited);
        btn.addEventListener('mouseleave', showEdited);
        btn.addEventListener('touchstart', (e) => { e.preventDefault(); showOriginal(); });
        btn.addEventListener('touchend', showEdited);
        btn.addEventListener('touchcancel', showEdited);
    });

    document.querySelectorAll('.btn-toggle').forEach(btn => {
        const sliderId = btn.dataset.slider;
        const slider = document.querySelector(`.slider-wrapper[data-slider="${sliderId}"]`);
        if (!slider) return;
        const handle = slider.querySelector('.slider-handle');
        const beforeImg = slider.querySelector('.img-before');
        if (!handle || !beforeImg) return;

        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const isActive = slider.classList.toggle('compare-active');
            btn.classList.toggle('active', isActive);
            if (isActive) {
                if (slider._sliderState) slider._sliderState.setPosition(50);
                else { handle.style.left = '50%'; beforeImg.style.clipPath = 'inset(0 50% 0 0)'; }
            } else {
                beforeImg.style.clipPath = 'inset(0 100% 0 0)';
            }
        });
    });
}

// ============================================
// Downloads & Exports
// ============================================
window.downloadImage = function(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};

window.openFullscreen = function(url) {
    window.open(url, '_blank');
};

function setupDownloadAll() {
    document.getElementById('download-all-btn').addEventListener('click', () => {
        let delay = 0;
        const settings = getCurrentSettings();
        state.results.forEach(result => {
            const targetFn = result.filename || result.inputUrl.split('/').pop();
            const refNames = Object.keys(result.referenceResults);
            for (const strength of state.selectedStrengths) {
                refNames.forEach(refName => {
                    const cacheKey = buildCacheKey(targetFn, refName, strength, settings);
                    const cached = state.strengthCache[cacheKey];
                    if (cached) {
                        Object.entries(cached.methods).forEach(([methodId, output]) => {
                            if (output?.url) {
                                setTimeout(() => downloadImage(output.url, `${methodId}_${refName}_${strength}pct_${result.originalName}`), delay);
                                delay += 400;
                            }
                        });
                    }
                });
            }
        });
    });
}

window.downloadXMP = async function(targetFn, refFn, methodId, strength, outputFn, strategy) {
    try {
        const formData = new FormData();
        formData.append('target_filename', targetFn);
        formData.append('reference_filename', refFn);
        formData.append('method_id', methodId);
        formData.append('color_strength', strength);
        formData.append('xmp_strategy', strategy || 'color_science');
        if (outputFn) formData.append('output_filename', outputFn);

        const response = await fetch('/api/export/xmp', { method: 'POST', body: formData });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Download failed' }));
            alert('XMP download failed: ' + (err.detail || 'Unknown error'));
            return;
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        const refName = refFn.replace(/\.[^.]+$/, '');
        const pct = Math.round(strength * 100);
        const strategyLabel = {
            'color_science': 'colorsci', 'basic': 'basic', 'basic_optimized': 'basic+opt',
            'darktable': 'darktable', 'rawtherapee': 'rawtherapee', 'rapidraw': 'rapidraw',
            'darktable_optimized': 'darktable+opt', 'rawtherapee_optimized': 'rawtherapee+opt',
            'rapidraw_optimized': 'rapidraw+opt', 'rapidraw_exact_inverse': 'rapidraw-exact',
            'rapidraw_exact_inverse_optimized': 'rapidraw-exact+opt'
        }[strategy] || strategy;
        link.href = url;
        link.download = `${refName}_${strategyLabel}.xmp`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } catch (e) {
        alert('XMP download failed: ' + e.message);
    }
};

window.toggleXmpDropdown = function(btn) {
    const wrap = btn.closest('.xmp-dropdown-wrap');
    const dropdown = wrap.querySelector('.xmp-dropdown');
    document.querySelectorAll('.xmp-dropdown.open').forEach(d => { if (d !== dropdown) d.classList.remove('open'); });
    dropdown.classList.toggle('open');
};

document.addEventListener('click', (e) => {
    if (!e.target.closest('.xmp-dropdown-wrap')) document.querySelectorAll('.xmp-dropdown.open').forEach(d => d.classList.remove('open'));
});

function setupExportZip() {
    const btn = document.getElementById('export-zip-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
        btn.disabled = true;
        const origText = btn.innerHTML;
        btn.textContent = 'Exporting...';

        try {
            const settings = getCurrentSettings();
            const items = [];
            state.results.forEach(result => {
                const targetFn = result.filename || result.inputUrl.split('/').pop();
                const refNames = Object.keys(result.referenceResults);
                for (const strength of state.selectedStrengths) {
                    refNames.forEach(refName => {
                        const refData = result.referenceResults[refName];
                        const cacheKey = buildCacheKey(targetFn, refName, strength, settings);
                        const cached = state.strengthCache[cacheKey];
                        if (cached) {
                            Object.entries(cached.methods).forEach(([methodId, output]) => {
                                if (output?.filename) {
                                    items.push({
                                        output_filename: output.filename, method_id: methodId,
                                        reference_name: refName, strength, target_filename: targetFn,
                                        reference_filename: refData.refFilename,
                                    });
                                }
                            });
                        }
                    });
                }
            });

            if (items.length === 0) { alert('No results to export.'); return; }

            const xmpStrategy = document.getElementById('xmp-strategy-select')?.value || 'color_science';
            const response = await fetch('/api/export/zip', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ items, xmp_strategy: xmpStrategy }),
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: 'Export failed' }));
                alert('Export failed: ' + (err.detail || 'Unknown error'));
                return;
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'LuminaFix_Export.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (e) {
            alert('Export failed: ' + e.message);
        } finally {
            btn.disabled = false;
            btn.innerHTML = origText;
        }
    });
}

// Parameters panel
window.toggleParams = async function(panelKey, targetFn, refFn, strength, outputFn) {
    const panel = document.getElementById('params-' + panelKey);
    if (!panel) return;
    if (panel.style.display !== 'none') { panel.style.display = 'none'; return; }

    panel.style.display = 'block';
    panel.innerHTML = '<div class="params-loading">Loading parameters...</div>';

    try {
        const formData = new FormData();
        formData.append('target_filename', targetFn);
        formData.append('reference_filename', refFn);
        formData.append('color_strength', strength);
        if (outputFn) formData.append('output_filename', outputFn);

        const response = await fetch('/api/export/params', { method: 'POST', body: formData });
        if (!response.ok) { panel.innerHTML = '<div class="params-error">Failed to load</div>'; return; }

        const data = await response.json();
        const p = data.params;
        panel.innerHTML = `
            <div class="params-grid">
                <div class="params-title">Lightroom Parameters</div>
                <div class="param-row"><span class="param-name">Temperature</span><span class="param-value">${p.temperature}K</span><div class="param-bar"><div class="param-bar-fill param-bar-temp" style="width:${Math.min(100, ((p.temperature - 2000) / 48000) * 100)}%"></div></div></div>
                <div class="param-row"><span class="param-name">Tint</span><span class="param-value">${p.tint > 0 ? '+' : ''}${p.tint}</span><div class="param-bar"><div class="param-bar-fill param-bar-tint" style="width:${50 + (p.tint / 300) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Exposure</span><span class="param-value">${p.exposure > 0 ? '+' : ''}${p.exposure.toFixed(2)}</span><div class="param-bar"><div class="param-bar-fill param-bar-exp" style="width:${50 + (p.exposure / 10) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Contrast</span><span class="param-value">${p.contrast > 0 ? '+' : ''}${p.contrast}</span><div class="param-bar"><div class="param-bar-fill" style="width:${50 + (p.contrast / 200) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Highlights</span><span class="param-value">${p.highlights > 0 ? '+' : ''}${p.highlights}</span><div class="param-bar"><div class="param-bar-fill" style="width:${50 + (p.highlights / 200) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Shadows</span><span class="param-value">${p.shadows > 0 ? '+' : ''}${p.shadows}</span><div class="param-bar"><div class="param-bar-fill" style="width:${50 + (p.shadows / 200) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Saturation</span><span class="param-value">${p.saturation > 0 ? '+' : ''}${p.saturation}</span><div class="param-bar"><div class="param-bar-fill param-bar-sat" style="width:${50 + (p.saturation / 200) * 50}%"></div></div></div>
                <div class="param-row"><span class="param-name">Vibrance</span><span class="param-value">${p.vibrance > 0 ? '+' : ''}${p.vibrance}</span><div class="param-bar"><div class="param-bar-fill param-bar-vib" style="width:${50 + (p.vibrance / 200) * 50}%"></div></div></div>
            </div>
        `;
    } catch (e) {
        panel.innerHTML = '<div class="params-error">Failed to load</div>';
    }
};

// ============================================
// Per-image strength controls
// ============================================
window.updatePerImageStrengthLabel = function(slider) {
    const valSpan = slider.parentElement.querySelector('.per-strength-val');
    if (valSpan) valSpan.textContent = slider.value + '%';
};

window.reprocessSingleImage = async function(slider) {
    const refName = slider.dataset.ref;
    const methodId = slider.dataset.method;
    const strength = parseInt(slider.value);

    // Store the per-image override
    state.perImageStrengths[refName] = strength;

    const result = state.results[state.currentIndex];
    const settings = getCurrentSettings();
    const targetFilename = result.filename || result.inputUrl.split('/').pop();
    const refData = result.referenceResults[refName];
    if (!refData) return;

    const cacheKey = buildCacheKey(targetFilename, refName, strength, settings);

    // Check cache first
    if (state.strengthCache[cacheKey]) {
        result.referenceResults[refName].methods = { ...state.strengthCache[cacheKey].methods };
        result.referenceResults[refName].errors = { ...state.strengthCache[cacheKey].errors };
        showCurrentResult();
        return;
    }

    // Show processing on this cell
    showStatus(`Reprocessing ${refName} at ${strength}%...`, 'var(--accent)', 0);
    setProcessingBanner(true, `Reprocessing ${refName} at ${strength}%...`);

    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard'];
    const nilutMode = sessionStorage.getItem('nilutMode') || 'universal';
    const nilutModels = sessionStorage.getItem('nilutModels') || extractNilutModelIds(state.results);

    const formData = new FormData();
    formData.append('target_filename', targetFilename);
    formData.append('reference_filename', refData.refFilename);
    formData.append('color_strength', strength / 100);
    formData.append('luminance_strength', settings.luminanceStrength / 100);
    formData.append('skin_protection', settings.skinProtection);
    formData.append('neon_protection', settings.neonProtection);
    formData.append('lip_protection', settings.lipProtection);
    formData.append('methods', JSON.stringify(selectedMethods));
    formData.append('nilut_mode', nilutMode);
    formData.append('nilut_models', nilutModels);

    try {
        const response = await fetch('/api/process-all', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Processing failed');
        const data = await response.json();
        const cached = { methods: data.outputs || {}, errors: data.errors || {} };
        for (const mid in cached.methods) {
            if (cached.methods[mid]?.url) cached.methods[mid].url += '?t=' + Date.now();
        }
        state.strengthCache[cacheKey] = cached;

        result.referenceResults[refName].methods = { ...cached.methods };
        result.referenceResults[refName].errors = { ...cached.errors };
        showCurrentResult();
        setProcessingBanner(false);
        showStatus('Updated!', 'var(--success)', 2000);
    } catch (error) {
        console.error('Per-image reprocess failed:', error);
        setProcessingBanner(false);
        showStatus('Failed to reprocess', 'var(--danger)', 3000);
    }
};
