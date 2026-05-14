// ============================================
// LuminaFix Pro - Results Page
// One reference + many target photos
// state.results = [{ refUrl, refName, refFilename, targetResults: {filename: {targetUrl, targetFilename, originalName, methods, errors}} }]
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
    results: [],         // see top-of-file shape
    baselineResults: [],
    currentIndex: 0,     // always 0 — single reference
    isProcessing: false,
    isPreprocessing: false,
    debounceTimer: null,
    comparisonMode: false,
    selectedStrengths: [90],
    strengthCache: {},
    perTargetStrengths: {}, // key: targetFilename -> strength value
    segmentNames: [],
    segmentStrengths: {},
    segmentTouched: {}
};

// ============================================
// Helpers
// ============================================
function getCurrentResult() {
    return state.results && state.results[0];
}

function getTargetFilenames(result) {
    return Object.keys((result || {}).targetResults || {});
}

function buildCacheKey(targetFilename, refFilename, strength, settings) {
    return `${targetFilename}_${refFilename}_${strength}_${settings.luminanceStrength}_${settings.skinProtection}_${settings.neonProtection}_${settings.lipProtection}_${settings.curveStrength}_${settings.saturationBoost}`;
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
        Object.keys(result.targetResults || {}).forEach(targetFn => {
            const methods = result.targetResults[targetFn].methods || {};
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
    const curveEl = document.getElementById('curve-strength');
    const satEl = document.getElementById('saturation-boost');
    return {
        luminanceStrength: document.getElementById('luminance-strength').value,
        skinProtection: document.getElementById('skin-protection').checked,
        neonProtection: document.getElementById('neon-protection').checked,
        lipProtection: document.getElementById('lip-protection').checked,
        curveStrength: curveEl ? parseInt(curveEl.value) / 100 : 0.5,
        saturationBoost: satEl ? parseInt(satEl.value) / 100 : 1.4
    };
}

// Per-segment defaults (percent) chosen by class name.
const DEFAULT_FALLBACK_STRENGTH = 70;
const SEGMENT_DEFAULT_BY_NAME = {
    sky: 100, cloud: 100, wall: 90, building: 90, house: 90, ceiling: 90,
    floor: 70, road: 70, sidewalk: 70, earth: 70, sand: 70, rock: 80,
    water: 90, sea: 100, river: 90, tree: 25, grass: 20, plant: 25,
    field: 25, flower: 60, person: 30, skin: 30, other: 70
};

function defaultStrengthFor(name) {
    if (name in SEGMENT_DEFAULT_BY_NAME) return SEGMENT_DEFAULT_BY_NAME[name];
    return DEFAULT_FALLBACK_STRENGTH;
}

function getSegmentStrengthsPayload() {
    if (!state.segmentNames || state.segmentNames.length === 0) return '';
    const payload = {};
    for (const name of state.segmentNames) {
        if (!state.segmentTouched[name]) continue;
        const pct = state.segmentStrengths[name];
        if (pct == null) continue;
        payload[name] = pct / 100;
    }
    return Object.keys(payload).length ? JSON.stringify(payload) : '';
}

async function initSegmentSliders() {
    // With multiple target photos, segmentation is ambiguous (each photo has different content),
    // so run analysis on the first uploaded photo. Sliders still apply to every photo via the API.
    const container = document.getElementById('segment-sliders');
    const statusEl = document.getElementById('segments-status');
    const group = document.getElementById('segment-strengths-group');
    if (!container) return;

    const result = getCurrentResult();
    const targetFilenames = getTargetFilenames(result);
    const firstTarget = targetFilenames[0];
    if (!firstTarget) {
        if (group) group.style.display = 'none';
        return;
    }

    if (statusEl) statusEl.textContent = 'analyzing…';
    let data;
    try {
        const url = `/api/analyze-segments?target_filename=${encodeURIComponent(firstTarget)}`;
        const res = await fetch(url);
        data = await res.json();
    } catch (e) {
        if (group) group.style.display = 'none';
        return;
    }

    if (!data.available || !data.segments || data.segments.length === 0) {
        if (group) group.style.display = 'none';
        return;
    }

    const visibleSegments = data.segments.filter(s => s.name !== 'other');
    state.segmentNames = visibleSegments.map(s => s.name);
    state.segmentStrengths = {};
    state.segmentTouched = {};
    if (statusEl) statusEl.textContent = '';

    container.innerHTML = '';
    for (const seg of visibleSegments) {
        const name = seg.name;
        const pct = seg.pixel_pct != null ? seg.pixel_pct : null;
        const def = defaultStrengthFor(name);
        state.segmentStrengths[name] = def;

        const row = document.createElement('div');
        row.className = 'setting-group-header';
        row.style.marginTop = '8px';
        const coverage = pct != null ? ` <span style="opacity:0.6;font-weight:400;">(${pct}%)</span>` : '';
        row.innerHTML = `
            <span style="text-transform:capitalize;">${name}${coverage}</span>
            <span class="setting-val" id="seg-val-${name}">${def}%</span>
        `;
        container.appendChild(row);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '150';
        slider.value = String(def);
        slider.id = `seg-strength-${name}`;
        slider.addEventListener('input', () => {
            const v = parseInt(slider.value, 10);
            state.segmentStrengths[name] = v;
            state.segmentTouched[name] = true;
            const valEl = document.getElementById(`seg-val-${name}`);
            if (valEl) valEl.textContent = v + '%';
            if (state.debounceTimer) clearTimeout(state.debounceTimer);
            state.debounceTimer = setTimeout(() => {
                state.strengthCache = {};
                reprocessAllStrengths();
            }, 500);
        });
        container.appendChild(slider);
    }
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

    // Clean up results: prune any method outputs we didn't select on the index page
    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard', 'nilut'];

    state.results.forEach(result => {
        Object.keys(result.targetResults || {}).forEach(targetFn => {
            const tData = result.targetResults[targetFn];
            const allMethodIds = [...Object.keys(tData.methods || {}), ...Object.keys(tData.errors || {})];
            allMethodIds.forEach(methodId => {
                const baseMethodId = getBaseMethodId(methodId);
                if (!selectedMethods.includes(baseMethodId)) {
                    delete tData.methods[methodId];
                    delete tData.errors[methodId];
                }
            });
        });
    });
    sessionStorage.setItem('styleTransferResults', JSON.stringify(state.results));

    // Restore settings
    let initialStrength = 90;
    if (storedSettings) {
        const settings = JSON.parse(storedSettings);
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

    // Seed cache from the initial run
    const initSettings = getCurrentSettings();
    const result = getCurrentResult();
    if (result) {
        const refFilename = result.refFilename;
        Object.keys(result.targetResults).forEach(targetFn => {
            const cacheKey = buildCacheKey(targetFn, refFilename, initialStrength, initSettings);
            state.strengthCache[cacheKey] = {
                methods: { ...result.targetResults[targetFn].methods },
                errors: { ...result.targetResults[targetFn].errors }
            };
        });
    }

    // Update version count + nav title + sidebar reference image
    if (result) {
        const targetCount = Object.keys(result.targetResults).length;
        const versionCountEl = document.getElementById('version-count');
        if (versionCountEl) versionCountEl.textContent = `${targetCount} photo${targetCount !== 1 ? 's' : ''}`;

        const nameEl = document.getElementById('image-name');
        if (nameEl) nameEl.textContent = result.refName || 'Results';

        const refImg = document.getElementById('original-image');
        if (refImg) refImg.src = result.refUrl || '';

        const refMeta = document.getElementById('reference-name');
        if (refMeta) refMeta.textContent = result.refName || '';
    }

    showCurrentResult();
    setupDownloadAll();
    setupExportZip();
    setupLiveSettings();
    setupComparisonMode();
    initSegmentSliders();

    if (state.selectedStrengths.length > 1 || (state.selectedStrengths.length === 1 && state.selectedStrengths[0] !== initialStrength)) {
        reprocessAllStrengths();
    }
});

// ============================================
// Live Settings
// ============================================
function setupLiveSettings() {
    const colorSlider = document.getElementById('color-strength');
    const colorValue = document.getElementById('color-strength-value');
    colorSlider.addEventListener('input', (e) => {
        colorValue.textContent = e.target.value + '%';
    });
    colorSlider.addEventListener('change', (e) => {
        const newStrength = parseInt(e.target.value);
        state.selectedStrengths = [newStrength];
        state.perTargetStrengths = {};
        state.strengthCache = {};
        state.isPreprocessing = false;
        debouncedReprocess();
    });

    document.getElementById('luminance-strength').addEventListener('input', (e) => {
        document.getElementById('luminance-value').textContent = e.target.value + '%';
        state.strengthCache = {};
        state.isPreprocessing = false;
        state.perTargetStrengths = {};
        debouncedReprocess();
    });

    const curveEl = document.getElementById('curve-strength');
    const curveValEl = document.getElementById('curve-strength-value');
    if (curveEl && curveValEl) {
        curveEl.addEventListener('input', (e) => {
            curveValEl.textContent = e.target.value + '%';
        });
        curveEl.addEventListener('change', () => {
            state.strengthCache = {};
            state.isPreprocessing = false;
            state.perTargetStrengths = {};
            debouncedReprocess();
        });
    }

    const satEl = document.getElementById('saturation-boost');
    const satValEl = document.getElementById('saturation-boost-value');
    if (satEl && satValEl) {
        satEl.addEventListener('input', (e) => {
            satValEl.textContent = (parseInt(e.target.value) / 100).toFixed(2) + '×';
        });
        satEl.addEventListener('change', () => {
            state.strengthCache = {};
            state.isPreprocessing = false;
            state.perTargetStrengths = {};
            debouncedReprocess();
        });
    }

    ['skin-protection', 'neon-protection', 'lip-protection'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            state.strengthCache = {};
            state.isPreprocessing = false;
            state.perTargetStrengths = {};
            debouncedReprocess();
        });
    });
}

function updateComparisonToggleState() {}

function setupComparisonMode() {
    const toggle = document.getElementById('comparison-mode-toggle');
    const baselineInfo = document.getElementById('baseline-info');
    const resetBtn = document.getElementById('reset-baseline-btn');

    if (!toggle) return;

    toggle.addEventListener('change', (e) => {
        state.comparisonMode = e.target.checked;
        if (e.target.checked) {
            baselineInfo.style.display = 'block';
            updateSettingsDisplay();
        } else {
            baselineInfo.style.display = 'none';
        }

        const result = getCurrentResult();
        const baseline = state.baselineResults[0];
        const targetFns = getTargetFilenames(result);
        let hasChanges = false;
        for (const tFn of targetFns) {
            const currentMethods = result.targetResults[tFn].methods;
            const baselineMethods = (baseline?.targetResults?.[tFn]?.methods) || {};
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

    if (resetBtn) {
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
    }

    updateComparisonToggleState();
}

function updateSettingsDisplay() {
    if (!state.baselineSettings) return;
    const baselineDisplay = document.getElementById('baseline-settings-display');
    const currentDisplay = document.getElementById('current-settings-display');
    if (!baselineDisplay || !currentDisplay) return;
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
    if (!status) return;
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
    if (status) {
        status.textContent = 'Settings changed...';
        status.style.color = '';
    }
    setProcessingBanner(true, 'Updating — settings changed...');
    state.debounceTimer = setTimeout(() => reprocessAllStrengths(), 500);
}

// ============================================
// Reprocessing
// ============================================
async function reprocessAllStrengths() {
    const result = getCurrentResult();
    if (!result) { setProcessingBanner(false); return; }
    if (state.isProcessing) { state.pendingReprocess = true; return; }
    if (state.selectedStrengths.length === 0) {
        setProcessingBanner(false);
        showStatus('Select at least one color strength.', 'var(--warning)', 3000);
        return;
    }

    state.isProcessing = true;
    const status = document.getElementById('reprocess-status');
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
    const refFilename = result.refFilename;
    const targetFns = getTargetFilenames(result);

    const tasks = [];
    for (const strength of state.selectedStrengths) {
        for (const targetFn of targetFns) {
            const cacheKey = buildCacheKey(targetFn, refFilename, strength, settings);
            if (!state.strengthCache[cacheKey]) tasks.push({ strength, targetFn, cacheKey });
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
    if (status) {
        status.className = 'reprocess-status processing';
        status.innerHTML = `<span class="spinner-small"></span> Processing 0/${tasks.length}...`;
    }
    setProcessingBanner(true, `Processing 0/${tasks.length}...`);

    const CONCURRENCY = 3;
    let taskIndex = 0;

    async function processNext() {
        while (taskIndex < tasks.length) {
            const task = tasks[taskIndex++];

            const formData = new FormData();
            formData.append('target_filename', task.targetFn);
            formData.append('reference_filename', refFilename);
            formData.append('color_strength', task.strength / 100);
            formData.append('luminance_strength', settings.luminanceStrength / 100);
            formData.append('skin_protection', settings.skinProtection);
            formData.append('neon_protection', settings.neonProtection);
            formData.append('lip_protection', settings.lipProtection);
            formData.append('methods', JSON.stringify(selectedMethods));
            formData.append('nilut_mode', nilutMode);
            formData.append('nilut_models', nilutModels);
            formData.append('curve_strength', settings.curveStrength);
            formData.append('saturation_boost', settings.saturationBoost);
            const segPayload = getSegmentStrengthsPayload();
            if (segPayload) formData.append('per_segment_strengths', segPayload);

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
            if (status) status.innerHTML = `<span class="spinner-small"></span> Processing ${completed}/${tasks.length}...`;
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
    const result = getCurrentResult();
    if (!result) return;
    const refFilename = result.refFilename;

    Object.keys(result.targetResults).forEach(targetFn => {
        const cacheKey = buildCacheKey(targetFn, refFilename, firstStrength, settings);
        const cached = state.strengthCache[cacheKey];
        if (cached) {
            result.targetResults[targetFn].methods = { ...cached.methods };
            result.targetResults[targetFn].errors = { ...cached.errors };
        }
    });
}

// ============================================
// Render Results
// ============================================
function showCurrentResult() {
    const result = getCurrentResult();
    if (!result) return;
    const baselineResult = state.baselineResults[0];
    const targetFns = getTargetFilenames(result);

    const grid = document.getElementById('results-grid');
    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedBaseMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard'];

    // Collect method IDs across all targets
    const allMethodIds = new Set();
    targetFns.forEach(tFn => {
        const tData = result.targetResults[tFn];
        Object.keys(tData.methods || {}).forEach(id => allMethodIds.add(id));
        Object.keys(tData.errors || {}).forEach(id => allMethodIds.add(id));
    });

    const refFilename = result.refFilename;
    if (state.selectedStrengths.length > 1) {
        const settings = getCurrentSettings();
        for (const strength of state.selectedStrengths) {
            for (const tFn of targetFns) {
                const cacheKey = buildCacheKey(tFn, refFilename, strength, settings);
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
    grid.setAttribute('data-cols', methods.length);
    const isMultiStrength = false;

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

    // One row per uploaded photo
    targetFns.forEach(targetFn => {
        const tData = result.targetResults[targetFn];
        html += `<div class="grid-row">`;

        // Left cell: the reference (style source) — same in every row.
        // Photo filename goes into the right cell as a caption so each row is identifiable.
        if (result.refUrl) {
            html += `<div class="grid-ref-cell grid-target-cell">
                <img src="${result.refUrl}" alt="${result.refName}" class="target-thumb" onclick="openFullscreen('${result.refUrl}')">
                <span class="target-name">${result.refName}</span>
                <span class="target-photo-caption">→ ${tData.originalName}</span>
            </div>`;
        } else {
            html += `<div class="grid-ref-cell grid-target-cell">
                <div class="target-thumb xmp-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>
                <span class="target-name">${result.refName}</span>
                <span class="target-photo-caption">→ ${tData.originalName}</span>
            </div>`;
        }

        // Method cells
        methods.forEach(methodId => {
            const methodInfo = getMethodInfo(methodId);

            const output = findMethodOutput(tData.methods, methodId);
            const error = findMethodOutput(tData.errors, methodId);
            const baselineTData = state.comparisonMode ? (baselineResult?.targetResults?.[targetFn]) : null;
            const baselineOutput = baselineTData ? findMethodOutput(baselineTData.methods, methodId) : null;
            const baselineError = baselineTData ? findMethodOutput(baselineTData.errors, methodId) : null;

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
                    <div class="comparison-slider-wrapper" data-comparison-slider="${methodId}-${targetFn}">
                        <img class="comparison-img-baseline" src="${baselineOutput.url}" alt="Baseline">
                        <img class="comparison-img-tweaked" src="${output.url}" alt="Tweaked">
                        <div class="comparison-slider-handle"><div class="comparison-slider-line"></div>
                            <div class="comparison-slider-circle"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/><polyline points="9 18 15 12 9 6" transform="translate(6,0)"/></svg></div>
                        </div>
                        <div class="comparison-labels-overlay"><span class="baseline-side-label">Baseline</span><span class="tweaked-side-label">Tweaked</span></div>
                    </div>
                    <div class="cell-actions">
                        <button class="btn-icon btn-hold-comparison" data-slider="${methodId}-${targetFn}" title="Toggle view">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        </button>
                        <button class="btn-icon btn-toggle-comparison" data-slider="${methodId}-${targetFn}" title="Restore slider">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>
                        </button>
                        <div class="btn-separator"></div>
                        <button class="btn-icon" onclick="downloadImage('${baselineOutput.url}', 'baseline_${methodId}_${tData.originalName}')" title="Download Baseline"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg><span class="btn-label-small">Base</span></button>
                        <button class="btn-icon" onclick="downloadImage('${output.url}', 'tweaked_${methodId}_${tData.originalName}')" title="Download Tweaked"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg><span class="btn-label-small">Tweak</span></button>
                        <button class="btn-icon" onclick="openFullscreen('${output.url}')" title="Fullscreen"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg></button>
                    </div>
                </div>`;
            } else if (!state.comparisonMode && output) {
                let beforeUrl = tData.targetUrl;
                const baseId = getBaseMethodId(methodId);
                if (['nilut_contrast','nilut_tonecurve','nilut_tonecurve_sat'].includes(baseId)) {
                    const nilutOutput = findMethodOutput(tData.methods, 'nilut');
                    if (nilutOutput?.url) beforeUrl = nilutOutput.url;
                }

                html += `<div class="grid-cell" data-method="${methodInfo.name}">
                    <div class="slider-wrapper" data-slider="${methodId}-${targetFn}">
                        <img class="img-before" src="${beforeUrl}" alt="Original">
                        <img class="img-after" src="${output.url}" alt="${methodInfo.name}">
                    </div>
                    <div class="cell-actions">
                        <button class="btn-icon btn-compare" data-slider="${methodId}-${targetFn}" title="Hold to see original">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                        </button>
                        <button class="btn-icon" onclick="downloadImage('${output.url}', '${methodId}_${tData.originalName}')" title="Download"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></button>
                        <button class="btn-icon" onclick="openFullscreen('${output.url}')" title="Fullscreen"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg></button>
                    </div>
                    <div class="per-image-strength">
                        <label>Strength</label>
                        <input type="range" min="0" max="200" value="${state.perTargetStrengths[targetFn] || state.selectedStrengths[0]}"
                               data-target="${targetFn}" data-method="${methodId}"
                               oninput="updatePerImageStrengthLabel(this)"
                               onchange="reprocessSingleImage(this)">
                        <span class="per-strength-val">${state.perTargetStrengths[targetFn] || state.selectedStrengths[0]}%</span>
                    </div>
                </div>`;
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
    const btn = document.getElementById('download-all-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
        let delay = 0;
        const settings = getCurrentSettings();
        const result = getCurrentResult();
        if (!result) return;
        const refFilename = result.refFilename;
        const targetFns = getTargetFilenames(result);
        for (const strength of state.selectedStrengths) {
            targetFns.forEach(targetFn => {
                const tData = result.targetResults[targetFn];
                const cacheKey = buildCacheKey(targetFn, refFilename, strength, settings);
                const cached = state.strengthCache[cacheKey];
                if (cached) {
                    Object.entries(cached.methods).forEach(([methodId, output]) => {
                        if (output?.url) {
                            setTimeout(() => downloadImage(output.url, `${methodId}_${strength}pct_${tData.originalName}`), delay);
                            delay += 400;
                        }
                    });
                }
            });
        }
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
            const result = getCurrentResult();
            if (!result) { alert('No results to export.'); return; }
            const refFilename = result.refFilename;
            const refName = result.refName;
            const targetFns = getTargetFilenames(result);

            for (const strength of state.selectedStrengths) {
                targetFns.forEach(targetFn => {
                    const cacheKey = buildCacheKey(targetFn, refFilename, strength, settings);
                    const cached = state.strengthCache[cacheKey];
                    if (cached) {
                        Object.entries(cached.methods).forEach(([methodId, output]) => {
                            if (output?.filename) {
                                items.push({
                                    output_filename: output.filename,
                                    method_id: methodId,
                                    reference_name: refName,
                                    strength,
                                    target_filename: targetFn,
                                    reference_filename: refFilename,
                                });
                            }
                        });
                    }
                });
            }

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
// Per-target strength controls
// ============================================
window.updatePerImageStrengthLabel = function(slider) {
    const valSpan = slider.parentElement.querySelector('.per-strength-val');
    if (valSpan) valSpan.textContent = slider.value + '%';
};

window.reprocessSingleImage = async function(slider) {
    const targetFn = slider.dataset.target;
    const strength = parseInt(slider.value);
    state.perTargetStrengths[targetFn] = strength;

    const result = getCurrentResult();
    if (!result) return;
    const settings = getCurrentSettings();
    const refFilename = result.refFilename;
    const tData = result.targetResults[targetFn];
    if (!tData) return;

    const cacheKey = buildCacheKey(targetFn, refFilename, strength, settings);

    if (state.strengthCache[cacheKey]) {
        result.targetResults[targetFn].methods = { ...state.strengthCache[cacheKey].methods };
        result.targetResults[targetFn].errors = { ...state.strengthCache[cacheKey].errors };
        showCurrentResult();
        return;
    }

    showStatus(`Reprocessing ${tData.originalName} at ${strength}%...`, 'var(--accent)', 0);
    setProcessingBanner(true, `Reprocessing ${tData.originalName} at ${strength}%...`);

    const storedMethods = sessionStorage.getItem('selectedMethods');
    const selectedMethods = storedMethods ? JSON.parse(storedMethods) : ['reinhard'];
    const nilutMode = sessionStorage.getItem('nilutMode') || 'universal';
    const nilutModels = sessionStorage.getItem('nilutModels') || extractNilutModelIds(state.results);

    const formData = new FormData();
    formData.append('target_filename', targetFn);
    formData.append('reference_filename', refFilename);
    formData.append('color_strength', strength / 100);
    formData.append('luminance_strength', settings.luminanceStrength / 100);
    formData.append('skin_protection', settings.skinProtection);
    formData.append('neon_protection', settings.neonProtection);
    formData.append('lip_protection', settings.lipProtection);
    formData.append('methods', JSON.stringify(selectedMethods));
    formData.append('nilut_mode', nilutMode);
    formData.append('nilut_models', nilutModels);
    const segPayload2 = getSegmentStrengthsPayload();
    if (segPayload2) formData.append('per_segment_strengths', segPayload2);

    try {
        const response = await fetch('/api/process-all', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Processing failed');
        const data = await response.json();
        const cached = { methods: data.outputs || {}, errors: data.errors || {} };
        for (const mid in cached.methods) {
            if (cached.methods[mid]?.url) cached.methods[mid].url += '?t=' + Date.now();
        }
        state.strengthCache[cacheKey] = cached;

        result.targetResults[targetFn].methods = { ...cached.methods };
        result.targetResults[targetFn].errors = { ...cached.errors };
        showCurrentResult();
        setProcessingBanner(false);
        showStatus('Updated!', 'var(--success)', 2000);
    } catch (error) {
        console.error('Per-target reprocess failed:', error);
        setProcessingBanner(false);
        showStatus('Failed to reprocess', 'var(--danger)', 3000);
    }
};
