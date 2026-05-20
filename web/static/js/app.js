// ============================================
// LuminaFix Pro - Main Application JS
// Multiple photos + single reference workflow
// ============================================

const MAX_UPLOADS = 25;

const state = {
    references: [],
    selectedReference: null, // single filename
    uploadedFiles: [], // [{file, filename, url, originalName}]
    processedResults: [],
    nilutModels: [],
    nilutMode: null,
    universalModelAvailable: false,
    styleSource: 'reference',
    xmpPreset: null,
    selectedMethod: 'nilut_tonecurve_sat',
    categories: [],
    activeCategory: null, // null = no category selected, grid hidden
};

let elements = {};

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    elements = {
        // Upload
        heroUpload: document.getElementById('hero-upload'),
        uploadDropzone: document.getElementById('upload-dropzone'),
        fileInput: document.getElementById('file-input'),
        imagePreviewContainer: document.getElementById('image-preview-container'),
        uploadedGrid: document.getElementById('uploaded-grid'),
        uploadCount: document.getElementById('upload-count'),
        btnClearAll: document.getElementById('btn-clear-all'),
        // Source toggle
        sourceRefBtn: document.getElementById('source-ref-btn'),
        sourceXmpBtn: document.getElementById('source-xmp-btn'),
        // Reference
        referenceGrid: document.getElementById('reference-grid'),
        referenceUploadArea: document.getElementById('reference-upload-area'),
        referenceFileInput: document.getElementById('reference-file-input'),
        selectionCount: document.getElementById('selection-count'),
        // Method
        methodPills: document.getElementById('method-pills'),
        // NILUT
        nilutSection: document.getElementById('nilut-section'),
        nilutToggle: document.getElementById('nilut-toggle'),
        nilutContent: document.getElementById('nilut-content'),
        nilutModeUniversal: document.getElementById('nilut-mode-universal'),
        nilutModePerRef: document.getElementById('nilut-mode-per-ref'),
        universalModelSection: document.getElementById('universal-model-section'),
        perRefModelSection: document.getElementById('per-ref-model-section'),
        universalStatus: document.getElementById('universal-status'),
        trainUniversalBtn: document.getElementById('train-universal-btn'),
        nilutStatus: document.getElementById('nilut-status'),
        trainAllNilutBtn: document.getElementById('train-all-nilut-btn'),
        refreshNilutBtn: document.getElementById('refresh-nilut-btn'),
        nilutModelSelector: document.getElementById('nilut-model-selector'),
        nilutModelOptions: document.getElementById('nilut-model-options'),
        // Settings
        settingsToggle: document.getElementById('settings-toggle'),
        settingsPanel: document.getElementById('settings-panel'),
        colorStrength: document.getElementById('color-strength'),
        colorValue: document.getElementById('color-value'),
        luminanceStrength: document.getElementById('luminance-strength'),
        luminanceValue: document.getElementById('luminance-value'),
        skinProtection: document.getElementById('skin-protection'),
        neonProtection: document.getElementById('neon-protection'),
        lipProtection: document.getElementById('lip-protection'),
        // Action bar
        processBtn: document.getElementById('process-btn'),
        btnText: document.getElementById('btn-text'),
        btnLoading: document.getElementById('btn-loading'),
        progressBarContainer: document.getElementById('progress-bar-container'),
        progressBarFill: document.getElementById('progress-bar-fill'),
        progressLabel: document.getElementById('progress-label'),
        progressText: document.getElementById('progress-text'),
        actionHint: document.getElementById('action-hint'),
        navStatus: document.getElementById('nav-status'),
        // Blocks
        referenceBlock: document.getElementById('reference-block'),
        methodBlock: document.getElementById('method-block'),
    };

    loadNilutStatus();
    loadUniversalNilutStatus();
    setupEventListeners();
    updateNilutSectionVisibility();
    updateModelSelectorVisibility();
    updateProcessButton();
});

// ============================================
// Style Reference (single custom upload)
// ============================================
function toggleReferenceSelection(filename, event) {
    if (event && event.target.closest('.btn-delete-ref')) return;

    const prev = state.selectedReference;
    state.selectedReference = (prev === filename) ? null : filename;

    document.querySelectorAll('.custom-ref-item').forEach(item => {
        const isSel = item.dataset.filename === state.selectedReference;
        item.classList.toggle('selected', isSel);
    });

    updateProcessButton();
}

window.toggleReferenceSelection = toggleReferenceSelection;

// Reference upload
async function uploadReferenceImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/references/upload', { method: 'POST', body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();

        // Add the uploaded ref directly to state and auto-select (replacing any prior selection)
        const newRef = {
            name: data.name,
            filename: data.filename,
            url: data.url,
            type: 'user',
            category: 'Custom',
            deletable: true,
        };

        state.references.push(newRef);
        state.selectedReference = data.filename;

        renderCustomRefGrid();
        updateProcessButton();

    } catch (error) {
        console.error('Reference upload failed:', error);
        alert('Failed to upload reference: ' + error.message);
    }
}

// Remove a custom reference uploaded in this session
window.removeCustomRef = function(filename) {
    state.references = state.references.filter(r => r.filename !== filename);
    if (state.selectedReference === filename) state.selectedReference = null;
    renderCustomRefGrid();
    updateProcessButton();
};

// Render the grid of custom-uploaded references
function renderCustomRefGrid() {
    const customCard = document.getElementById('custom-ref-card');
    if (!customCard) return;

    const customRefs = state.references.filter(r => r.type === 'user');

    // Get or create the grid container
    let grid = customCard.querySelector('.custom-ref-grid');
    if (!grid) {
        grid = document.createElement('div');
        grid.className = 'custom-ref-grid';
        customCard.appendChild(grid);
    }

    if (customRefs.length === 0) {
        grid.innerHTML = '';
        return;
    }

    grid.innerHTML = customRefs.map(ref => `
        <div class="custom-ref-item ${state.selectedReference === ref.filename ? 'selected' : ''}" data-filename="${ref.filename}" onclick="toggleReferenceSelection('${ref.filename}', event)">
            <img src="${ref.url}" alt="${ref.name}">
            <button class="btn-delete-ref" onclick="event.stopPropagation(); removeCustomRef('${ref.filename}')" title="Remove">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
            <span class="ref-name">${ref.name}</span>
        </div>
    `).join('');
}

// ============================================
// Multi-photo upload
// ============================================
async function handleFiles(fileList) {
    const files = Array.from(fileList).filter(f => f && f.type.startsWith('image/'));
    if (files.length === 0) {
        alert('Please upload image files');
        return;
    }

    const remaining = MAX_UPLOADS - state.uploadedFiles.length;
    if (remaining <= 0) {
        alert(`You can upload at most ${MAX_UPLOADS} photos.`);
        return;
    }

    const toUpload = files.slice(0, remaining);
    if (files.length > remaining) {
        alert(`Only the first ${remaining} of ${files.length} photos will be added (max ${MAX_UPLOADS}).`);
    }

    for (const file of toUpload) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('/api/upload', { method: 'POST', body: formData });
            if (!response.ok) throw new Error('Upload failed');
            const data = await response.json();
            state.uploadedFiles.push({
                file,
                filename: data.filename,
                url: data.url,
                originalName: data.original_name,
                size: file.size,
            });
        } catch (error) {
            console.error('Upload failed:', error);
            alert(`Failed to upload ${file.name}: ${error.message}`);
        }
    }

    renderUploadedGrid();
    updateProcessButton();
}

function renderUploadedGrid() {
    if (!elements.uploadedGrid || !elements.imagePreviewContainer) return;

    const count = state.uploadedFiles.length;
    const hero = elements.heroUpload;
    const dropzone = elements.uploadDropzone;

    if (count === 0) {
        // Full-size dropzone, no thumbnails
        if (hero) hero.classList.remove('compact');
        if (dropzone) dropzone.classList.remove('compact');
        elements.imagePreviewContainer.style.display = 'none';
        elements.uploadedGrid.innerHTML = '';
        if (elements.uploadCount) elements.uploadCount.textContent = '0 photos';
        // Restore full prompt
        setHeroPrompt('Drop your photos here, or <u>browse</u>', 'JPG · PNG · TIFF · WebP · up to 50 MB each', false);
        return;
    }

    // Compact dropzone + thumbnails strip
    if (hero) hero.classList.add('compact');
    if (dropzone) dropzone.classList.add('compact');
    setHeroPrompt('Drop more photos here or <u>browse to add</u>', `${count}/${MAX_UPLOADS} added`, true);
    elements.imagePreviewContainer.style.display = 'block';
    if (elements.uploadCount) {
        elements.uploadCount.textContent = `${count} photo${count !== 1 ? 's' : ''} (max ${MAX_UPLOADS})`;
    }

    elements.uploadedGrid.innerHTML = state.uploadedFiles.map(f => `
        <div class="uploaded-item" data-filename="${f.filename}">
            <img src="${f.url}" alt="${f.originalName}" loading="lazy" decoding="async">
            <span class="uploaded-name">${f.originalName}</span>
            <button class="btn-remove-upload" type="button" onclick="removeUploadedFile('${f.filename}')" title="Remove">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        </div>
    `).join('');
}

function setHeroPrompt(titleHtml, subText, compact) {
    if (!elements.heroUpload) return;
    const titleEl = elements.heroUpload.querySelector('.dropzone-title');
    const subEl = elements.heroUpload.querySelector('.dropzone-sub');
    if (titleEl) titleEl.innerHTML = titleHtml;
    if (subEl) subEl.textContent = subText;
}

window.removeUploadedFile = function(filename) {
    const item = state.uploadedFiles.find(f => f.filename === filename);
    state.uploadedFiles = state.uploadedFiles.filter(f => f.filename !== filename);
    renderUploadedGrid();
    updateProcessButton();

    if (item) {
        // Best-effort cleanup on server
        fetch(`/api/cleanup/${encodeURIComponent(filename)}`, { method: 'DELETE' }).catch(() => {});
    }
};

function clearAllUploads() {
    const toDelete = state.uploadedFiles.slice();
    state.uploadedFiles = [];
    renderUploadedGrid();
    updateProcessButton();
    for (const f of toDelete) {
        fetch(`/api/cleanup/${encodeURIComponent(f.filename)}`, { method: 'DELETE' }).catch(() => {});
    }
}

// ============================================
// Method Selection
// ============================================
function setSelectedMethod(method) {
    state.selectedMethod = method;
    document.querySelectorAll('.method-pill').forEach(pill => {
        pill.classList.toggle('active', pill.dataset.method === method);
    });
    updateModelSelectorVisibility();
}

function getSelectedMethods() {
    return [state.selectedMethod];
}

// ============================================
// NILUT Model Management
// ============================================
async function loadNilutStatus() {
    if (!elements.nilutStatus) return;
    try {
        const response = await fetch('/api/nilut/status');
        const data = await response.json();
        state.nilutModels = data.references || [];

        if (state.nilutModels.length === 0) {
            elements.nilutStatus.innerHTML = '<span class="loading-text">No reference images found</span>';
            return;
        }

        elements.nilutStatus.innerHTML = `
            <table class="nilut-table">
                <thead>
                    <tr><th>Reference</th><th>Status</th><th>Actions</th></tr>
                </thead>
                <tbody>
                    ${state.nilutModels.map(ref => `
                        <tr>
                            <td>${ref.name}</td>
                            <td><span class="status-badge ${ref.has_model ? 'trained' : 'not-trained'}">${ref.has_model ? 'Trained' : 'Untrained'}</span></td>
                            <td><button class="btn btn-sm ${ref.has_model ? 'btn-ghost' : 'btn-accent'}" onclick="trainNilutModel('${ref.filename}', '${ref.name}')">${ref.has_model ? 'Retrain' : 'Train'}</button></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (error) {
        console.error('Failed to load NILUT status:', error);
        elements.nilutStatus.innerHTML = '<span class="loading-text" style="color: var(--danger);">Failed to load</span>';
    }
}

function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

async function trainNilutModel(filename, name) {
    const btn = event.target;
    const originalText = btn.textContent;
    btn.textContent = 'Training...';
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('reference_filename', filename);
        formData.append('use_all_references_as_samples', 'true');

        const response = await fetch('/api/nilut/train', { method: 'POST', body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Training failed');

        alert(`NILUT model trained for ${name}`);
        loadNilutStatus();
    } catch (error) {
        console.error('Training failed:', error);
        alert('Training failed: ' + error.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

window.trainNilutModel = trainNilutModel;

async function trainAllNilutModels() {
    if (!elements.trainAllNilutBtn) return;
    const btn = elements.trainAllNilutBtn;
    const originalText = btn.textContent;
    btn.textContent = 'Training...';
    btn.disabled = true;

    try {
        for (const ref of state.nilutModels) {
            btn.textContent = `Training ${ref.name}...`;
            const formData = new FormData();
            formData.append('reference_filename', ref.filename);
            formData.append('use_all_references_as_samples', 'true');
            const response = await fetch('/api/nilut/train', { method: 'POST', body: formData });
            if (!response.ok) {
                const data = await response.json();
                console.error(`Failed to train ${ref.name}:`, data.detail);
            }
        }
        alert('All NILUT models trained!');
        loadNilutStatus();
    } catch (error) {
        alert('Training failed: ' + error.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

async function loadUniversalNilutStatus() {
    if (!elements.universalStatus) return;
    try {
        const response = await fetch('/api/nilut/universal/status');
        const data = await response.json();
        state.universalModelAvailable = data.available;

        if (data.available) {
            elements.universalStatus.innerHTML = `
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <span class="status-badge trained">Trained</span>
                    <span class="loading-text">${data.training_references || '?'} refs${data.last_trained ? ` — ${formatDate(data.last_trained)}` : ''}</span>
                </div>
            `;
            if (elements.trainUniversalBtn) {
                elements.trainUniversalBtn.textContent = 'Retrain';
                elements.trainUniversalBtn.className = 'btn btn-sm btn-ghost';
            }
        } else {
            elements.universalStatus.innerHTML = `
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    <span class="status-badge not-trained">Not Trained</span>
                    <span class="loading-text">Train to use universal mode</span>
                </div>
            `;
        }
    } catch (error) {
        elements.universalStatus.innerHTML = '<span class="loading-text" style="color: var(--danger);">Failed to load</span>';
    }
}

async function trainUniversalNilutModel() {
    if (!elements.trainUniversalBtn) return;
    const btn = elements.trainUniversalBtn;
    const originalText = btn.textContent;
    btn.textContent = 'Training...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/nilut/universal/train', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Training failed');
        alert(`Universal model trained on ${data.training_references} references!`);
        loadUniversalNilutStatus();
    } catch (error) {
        alert('Training failed: ' + error.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function getNilutMode() {
    return 'universal';
}

async function loadUniversalModelVersions() {
    if (!elements.nilutModelOptions) return;
    try {
        const response = await fetch('/api/nilut/universal/versions');
        const data = await response.json();
        if (!data.models || data.models.length === 0) {
            elements.nilutModelOptions.innerHTML = '<span class="loading-text">No models available. Train first.</span>';
            return;
        }
        elements.nilutModelOptions.innerHTML = data.models.map(model => `
            <label style="display:flex;align-items:center;gap:0.5rem;margin:0.4rem 0;cursor:pointer;font-size:0.8rem;">
                <input type="checkbox" class="nilut-model-checkbox" value="${model.id}" ${model.is_latest ? 'checked' : ''} style="accent-color:var(--accent);">
                <span style="color:${model.is_latest ? 'var(--accent)' : 'var(--text-secondary)'}">
                    ${model.name} ${model.is_latest ? '<strong>(Latest)</strong>' : ''}
                </span>
            </label>
        `).join('');
    } catch (error) {
        elements.nilutModelOptions.innerHTML = '<span class="loading-text" style="color:var(--danger);">Failed to load</span>';
    }
}

function getSelectedNilutModels() {
    // Hardcoded to Feb 16 model version
    return ['20260216_075617'];
}

function updateModelSelectorVisibility() {
    if (!elements.nilutModelSelector) return;
    const hasNilut = state.selectedMethod.startsWith('nilut');
    const isUniversal = getNilutMode() === 'universal';

    if (hasNilut && isUniversal) {
        elements.nilutModelSelector.style.display = 'block';
        loadUniversalModelVersions();
    } else {
        elements.nilutModelSelector.style.display = 'none';
    }
}

function updateNilutSectionVisibility() {
    const isUniversal = elements.nilutModeUniversal && elements.nilutModeUniversal.checked;
    const isPerRef = elements.nilutModePerRef && elements.nilutModePerRef.checked;

    if (isUniversal) {
        if (elements.universalModelSection) elements.universalModelSection.style.display = 'block';
        if (elements.perRefModelSection) elements.perRefModelSection.style.display = 'none';
        state.nilutMode = 'universal';
    } else if (isPerRef) {
        if (elements.universalModelSection) elements.universalModelSection.style.display = 'none';
        if (elements.perRefModelSection) elements.perRefModelSection.style.display = 'block';
        state.nilutMode = 'per_reference';
    } else {
        if (elements.universalModelSection) elements.universalModelSection.style.display = 'none';
        if (elements.perRefModelSection) elements.perRefModelSection.style.display = 'none';
        state.nilutMode = null;
    }
}

// ============================================
// XMP Functions
// ============================================
async function uploadXMPPreset(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/xmp/upload', { method: 'POST', body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        state.xmpPreset = { filename: data.filename, presetName: data.preset_name, params: data.params };

        document.getElementById('xmp-upload-area').style.display = 'none';
        document.getElementById('xmp-loaded-info').style.display = 'block';
        document.getElementById('xmp-filename').textContent = data.filename;
        document.getElementById('xmp-preset-name').textContent = data.preset_name || '';

        const p = data.params;
        document.getElementById('xmp-params-preview').innerHTML = `
            <div class="xmp-params-grid">
                <span>Temp: ${p.temperature}K</span>
                <span>Tint: ${p.tint >= 0 ? '+' : ''}${p.tint}</span>
                <span>Exp: ${p.exposure >= 0 ? '+' : ''}${p.exposure}</span>
                <span>Contrast: ${p.contrast >= 0 ? '+' : ''}${p.contrast}</span>
                <span>Highlights: ${p.highlights >= 0 ? '+' : ''}${p.highlights}</span>
                <span>Shadows: ${p.shadows >= 0 ? '+' : ''}${p.shadows}</span>
                <span>Vibrance: ${p.vibrance >= 0 ? '+' : ''}${p.vibrance}</span>
                <span>Saturation: ${p.saturation >= 0 ? '+' : ''}${p.saturation}</span>
            </div>
        `;

        updateProcessButton();
    } catch (error) {
        alert('Failed to parse XMP preset: ' + error.message);
    }
}

function removeXMPPreset() {
    state.xmpPreset = null;
    document.getElementById('xmp-upload-area').style.display = 'block';
    document.getElementById('xmp-loaded-info').style.display = 'none';
    updateProcessButton();
}
window.removeXMPPreset = removeXMPPreset;

function setupXMPEventListeners() {
    const xmpUploadArea = document.getElementById('xmp-upload-area');
    const xmpBrowseBtn = document.getElementById('xmp-browse-btn');
    const xmpFileInput = document.getElementById('xmp-file-input');

    if (xmpBrowseBtn) {
        xmpBrowseBtn.addEventListener('click', (e) => { e.stopPropagation(); xmpFileInput.click(); });
    }

    if (xmpUploadArea) {
        xmpUploadArea.addEventListener('click', (e) => {
            if (e.target.closest('#xmp-browse-btn')) return;
            xmpFileInput.click();
        });
        xmpUploadArea.addEventListener('dragover', (e) => { e.preventDefault(); xmpUploadArea.classList.add('drag-over'); });
        xmpUploadArea.addEventListener('dragleave', () => xmpUploadArea.classList.remove('drag-over'));
        xmpUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            xmpUploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.toLowerCase().endsWith('.xmp')) uploadXMPPreset(file);
                else alert('Please upload an .xmp file');
            }
        });
    }

    if (xmpFileInput) {
        xmpFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) { uploadXMPPreset(e.target.files[0]); e.target.value = ''; }
        });
    }
}

// ============================================
// Process Button State
// ============================================
function updateProcessButton() {
    const uploadCount = state.uploadedFiles.length;
    const hasUpload = uploadCount > 0;
    const hasRef = !!state.selectedReference;
    const hasXMP = state.xmpPreset !== null;
    const canProcess = hasUpload && (state.styleSource === 'xmp' ? hasXMP : hasRef);

    if (elements.processBtn) {
        elements.processBtn.disabled = !canProcess;
    }

    if (elements.btnText) {
        elements.btnText.textContent = uploadCount > 0
            ? `Apply style to ${uploadCount} photo${uploadCount > 1 ? 's' : ''}`
            : 'Apply style to my photos';
    }

    if (elements.actionHint) {
        if (canProcess) {
            elements.actionHint.textContent = '';
        } else if (!hasRef) {
            elements.actionHint.textContent = 'Upload a style photo to begin';
        } else if (!hasUpload) {
            elements.actionHint.textContent = 'Add your photos to get started';
        }
    }
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Upload dropzone
    if (elements.uploadDropzone) {
        elements.uploadDropzone.addEventListener('click', (e) => {
            // Ignore clicks on the toolbar buttons inside the preview area
            if (e.target.closest('.upload-toolbar') || e.target.closest('.uploaded-item')) return;
            elements.fileInput.click();
        });
        elements.uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.add('drag-over');
        });
        elements.uploadDropzone.addEventListener('dragleave', () => elements.uploadDropzone.classList.remove('drag-over'));
        elements.uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files);
        });
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFiles(e.target.files);
                e.target.value = '';
            }
        });
    }

    if (elements.btnClearAll) {
        elements.btnClearAll.addEventListener('click', (e) => {
            e.stopPropagation();
            clearAllUploads();
        });
    }

    // Reference upload
    if (elements.referenceUploadArea) {
        elements.referenceUploadArea.addEventListener('click', () => elements.referenceFileInput.click());
        elements.referenceUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.referenceUploadArea.classList.add('drag-over');
        });
        elements.referenceUploadArea.addEventListener('dragleave', () => elements.referenceUploadArea.classList.remove('drag-over'));
        elements.referenceUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.referenceUploadArea.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) uploadReferenceImage(file);
                else alert('Please upload an image file');
            }
        });
    }

    if (elements.referenceFileInput) {
        elements.referenceFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) { uploadReferenceImage(e.target.files[0]); e.target.value = ''; }
        });
    }

    // Method pills
    if (elements.methodPills) {
        elements.methodPills.addEventListener('click', (e) => {
            const pill = e.target.closest('.method-pill');
            if (pill) setSelectedMethod(pill.dataset.method);
        });
    }

    // Settings toggle
    if (elements.settingsToggle) {
        elements.settingsToggle.addEventListener('click', () => {
            const isOpen = elements.settingsPanel.style.display !== 'none';
            elements.settingsPanel.style.display = isOpen ? 'none' : 'block';
            elements.settingsToggle.classList.toggle('open', !isOpen);
        });
    }

    // NILUT toggle
    if (elements.nilutToggle) {
        elements.nilutToggle.addEventListener('click', () => {
            const isOpen = elements.nilutContent.style.display !== 'none';
            elements.nilutContent.style.display = isOpen ? 'none' : 'block';
            elements.nilutToggle.classList.toggle('open', !isOpen);
        });
    }

    // Sliders
    if (elements.colorStrength) {
        elements.colorStrength.addEventListener('input', (e) => {
            elements.colorValue.textContent = e.target.value + '%';
        });
    }
    if (elements.luminanceStrength) {
        elements.luminanceStrength.addEventListener('input', (e) => {
            elements.luminanceValue.textContent = e.target.value + '%';
        });
    }

    // Process button
    if (elements.processBtn) {
        elements.processBtn.addEventListener('click', processAllImages);
    }

    // NILUT buttons
    if (elements.trainAllNilutBtn) elements.trainAllNilutBtn.addEventListener('click', trainAllNilutModels);
    if (elements.refreshNilutBtn) elements.refreshNilutBtn.addEventListener('click', loadNilutStatus);
    if (elements.trainUniversalBtn) elements.trainUniversalBtn.addEventListener('click', trainUniversalNilutModel);

    // NILUT mode
    if (elements.nilutModeUniversal) {
        elements.nilutModeUniversal.addEventListener('change', () => {
            state.nilutMode = 'universal';
            updateNilutSectionVisibility();
            updateModelSelectorVisibility();
        });
    }
    if (elements.nilutModePerRef) {
        elements.nilutModePerRef.addEventListener('change', () => {
            state.nilutMode = 'per_reference';
            updateNilutSectionVisibility();
            updateModelSelectorVisibility();
        });
    }
}

// ============================================
// Processing — N photos × 1 reference
// ============================================
async function processAllImages() {
    if (state.styleSource === 'xmp') return processXMP();
    if (!state.selectedReference || state.uploadedFiles.length === 0) return;

    const ref = state.references.find(r => r.filename === state.selectedReference);
    if (!ref) {
        alert('Selected reference not found.');
        return;
    }

    // UI: show processing state
    elements.btnText.style.display = 'none';
    elements.btnLoading.style.display = 'inline-flex';
    elements.processBtn.disabled = true;
    elements.progressBarContainer.style.display = 'flex';
    setNavStatus('Processing...', true);

    const total = state.uploadedFiles.length;
    let current = 0;

    const refResult = {
        refUrl: ref.url,
        refName: ref.name,
        refFilename: ref.filename,
        targetResults: {} // keyed by uploaded filename
    };

    try {
        elements.progressText.textContent = `Processing ${total} photo${total > 1 ? 's' : ''} in parallel...`;
        elements.progressBarFill.style.width = `0%`;
        elements.progressLabel.textContent = `0 / ${total}`;

        const processOne = async (upload) => {
            const formData = new FormData();
            formData.append('target_filename', upload.filename);
            formData.append('reference_filename', ref.filename);
            formData.append('color_strength', elements.colorStrength.value / 100);
            formData.append('luminance_strength', elements.luminanceStrength.value / 100);
            formData.append('skin_protection', elements.skinProtection.checked);
            formData.append('neon_protection', elements.neonProtection.checked);
            formData.append('lip_protection', elements.lipProtection.checked);
            formData.append('methods', JSON.stringify(getSelectedMethods()));
            formData.append('nilut_mode', getNilutMode());
            formData.append('nilut_models', JSON.stringify(getSelectedNilutModels()));

            const response = await fetch('/api/process-all', { method: 'POST', body: formData });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }
            const data = await response.json();

            current++;
            elements.progressText.textContent = `Processing ${current}/${total}...`;
            elements.progressBarFill.style.width = `${(current / total) * 100}%`;
            elements.progressLabel.textContent = `${current} / ${total}`;

            return { upload, data };
        };

        const results = await Promise.all(state.uploadedFiles.map(processOne));
        for (const { upload, data } of results) {
            refResult.targetResults[upload.filename] = {
                targetUrl: upload.url,
                targetFilename: upload.filename,
                originalName: upload.originalName,
                methods: data.outputs || {},
                errors: data.errors || {}
            };
        }

        state.processedResults = [refResult];

        // Store and navigate
        sessionStorage.setItem('styleTransferResults', JSON.stringify(state.processedResults));
        sessionStorage.setItem('styleTransferSettings', JSON.stringify({
            colorStrength: elements.colorStrength.value,
            luminanceStrength: elements.luminanceStrength.value,
            skinProtection: elements.skinProtection.checked,
            neonProtection: elements.neonProtection.checked,
            lipProtection: elements.lipProtection.checked
        }));
        sessionStorage.setItem('selectedMethods', JSON.stringify(getSelectedMethods()));
        sessionStorage.setItem('nilutMode', getNilutMode());
        sessionStorage.setItem('nilutModels', JSON.stringify(getSelectedNilutModels()));
        window.location.href = '/results';

    } catch (error) {
        console.error('Processing failed:', error);
        alert('Processing failed: ' + error.message);
        resetProcessingUI();
    }
}

async function processXMP() {
    if (!state.xmpPreset || state.uploadedFiles.length === 0) return;

    elements.btnText.style.display = 'none';
    elements.btnLoading.style.display = 'inline-flex';
    elements.processBtn.disabled = true;
    elements.progressBarContainer.style.display = 'flex';
    setNavStatus('Processing...', true);

    const total = state.uploadedFiles.length;
    let current = 0;

    const refResult = {
        refUrl: null,
        refName: state.xmpPreset.presetName,
        refFilename: state.xmpPreset.filename,
        targetResults: {}
    };

    try {
        elements.progressText.textContent = `Applying preset to ${total} photo${total > 1 ? 's' : ''}...`;
        elements.progressBarFill.style.width = `0%`;
        elements.progressLabel.textContent = `0 / ${total}`;

        const processOne = async (upload) => {
            const formData = new FormData();
            formData.append('target_filename', upload.filename);
            formData.append('xmp_params', JSON.stringify(state.xmpPreset.params));

            const response = await fetch('/api/process-xmp', { method: 'POST', body: formData });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }
            const data = await response.json();
            current++;
            elements.progressText.textContent = `Processing ${current}/${total}...`;
            elements.progressBarFill.style.width = `${(current / total) * 100}%`;
            elements.progressLabel.textContent = `${current} / ${total}`;
            return { upload, data };
        };

        const results = await Promise.all(state.uploadedFiles.map(processOne));
        for (const { upload, data } of results) {
            refResult.targetResults[upload.filename] = {
                targetUrl: upload.url,
                targetFilename: upload.filename,
                originalName: upload.originalName,
                methods: { 'xmp_preset': { filename: data.output_filename, url: data.output_url, method_name: 'XMP Preset', time_ms: 0 } },
                errors: {}
            };
        }

        state.processedResults = [refResult];
        sessionStorage.setItem('styleTransferResults', JSON.stringify(state.processedResults));
        sessionStorage.setItem('styleTransferSettings', JSON.stringify({
            colorStrength: 100, luminanceStrength: 0,
            skinProtection: false, neonProtection: false, lipProtection: false
        }));
        sessionStorage.setItem('selectedMethods', JSON.stringify(['xmp_preset']));
        window.location.href = '/results';

    } catch (error) {
        console.error('XMP processing failed:', error);
        alert('Processing failed: ' + error.message);
        resetProcessingUI();
    }
}

function resetProcessingUI() {
    elements.btnText.style.display = 'inline';
    elements.btnLoading.style.display = 'none';
    elements.progressBarContainer.style.display = 'none';
    elements.progressBarFill.style.width = '0%';
    elements.processBtn.disabled = false;
    setNavStatus('Ready', false);
}

function setNavStatus(text, isProcessing) {
    if (elements.navStatus) {
        const dot = elements.navStatus.querySelector('.status-dot');
        const label = elements.navStatus.querySelector('span:last-child');
        if (dot) dot.classList.toggle('processing', isProcessing);
        if (label) label.textContent = text;
    }
}
