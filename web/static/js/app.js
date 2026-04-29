// ============================================
// LuminaFix Pro - Main Application JS
// Single image + multiple references workflow
// ============================================

const state = {
    references: [],
    selectedReferences: new Set(),
    uploadedFile: null, // Single file: {file, filename, url, originalName}
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
        previewImage: document.getElementById('preview-image'),
        btnReplace: document.getElementById('btn-replace'),
        imageInfo: document.getElementById('image-info'),
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

    loadReferences();
    loadNilutStatus();
    loadUniversalNilutStatus();
    setupEventListeners();
    setupXMPEventListeners();
    updateNilutSectionVisibility();
    updateModelSelectorVisibility();
});

// ============================================
// References
// ============================================
async function loadReferences() {
    try {
        const response = await fetch('/api/references');
        const data = await response.json();

        if (data.references.length === 0) {
            elements.referenceGrid.innerHTML = '<div class="loading-skeleton">No reference images found.</div>';
            state.selectedReferences.clear();
            updateSelectionCount();
            return;
        }

        state.references = data.references;
        state.categories = data.categories || ['All'];

        // Build category tabs
        buildCategoryTabs();

        // Clean up stale selections
        const existingFilenames = new Set(data.references.map(r => r.filename));
        state.selectedReferences.forEach(filename => {
            if (!existingFilenames.has(filename)) state.selectedReferences.delete(filename);
        });

        renderReferenceGrid();
        updateSelectionCount();
        updateProcessButton();
    } catch (error) {
        console.error('Failed to load references:', error);
        elements.referenceGrid.innerHTML = '<div class="loading-skeleton">Failed to load references</div>';
    }
}

function buildCategoryTabs() {
    const tabsContainer = document.getElementById('category-tabs');
    if (!tabsContainer) return;

    // Build category list — include "Custom" if any user refs exist
    const hasCustom = state.references.some(r => r.category === 'Custom');
    let realCats = state.categories.filter(c => c !== 'All');
    if (hasCustom && !realCats.includes('Custom')) realCats.push('Custom');
    tabsContainer.innerHTML = realCats.map(cat =>
        `<button class="cat-tab ${cat === state.activeCategory ? 'active' : ''}" data-category="${cat}">${cat}</button>`
    ).join('');

    tabsContainer.addEventListener('click', (e) => {
        const tab = e.target.closest('.cat-tab');
        if (!tab) return;
        // Toggle: click same category again to collapse
        if (state.activeCategory === tab.dataset.category) {
            state.activeCategory = null;
            tabsContainer.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
        } else {
            state.activeCategory = tab.dataset.category;
            tabsContainer.querySelectorAll('.cat-tab').forEach(t => t.classList.toggle('active', t === tab));
        }
        renderReferenceGrid();
    });
}

function renderReferenceGrid() {
    const wrapper = document.getElementById('ref-grid-wrapper');

    // If no category selected, hide the grid
    if (!state.activeCategory) {
        if (wrapper) wrapper.style.display = 'none';
        return;
    }

    if (wrapper) wrapper.style.display = 'block';

    const filtered = state.activeCategory === 'All'
        ? state.references
        : state.references.filter(ref => ref.category === state.activeCategory);

    if (filtered.length === 0) {
        elements.referenceGrid.innerHTML = '<div class="loading-skeleton">No references in this category.</div>';
        return;
    }

    elements.referenceGrid.innerHTML = filtered.map(ref => `
        <div class="reference-item ${ref.type === 'user' ? 'user-uploaded' : ''} ${state.selectedReferences.has(ref.filename) ? 'selected' : 'deselected'}"
             data-filename="${ref.filename}"
             data-type="${ref.type}"
             onclick="toggleReferenceSelection('${ref.filename}', event)">
            <div class="ref-selection-indicator">
                <svg class="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            </div>
            <img src="${ref.thumb_url || ref.url}" alt="${ref.name}" loading="lazy" decoding="async">
            <span class="ref-name">${ref.name}</span>
            ${ref.deletable ? `
                <button class="btn-delete-ref" onclick="deleteReference('${ref.filename}')" title="Delete">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                    </svg>
                </button>
            ` : ''}
            ${ref.type === 'user' ? '<span class="ref-badge">Custom</span>' : ''}
        </div>
    `).join('');
}

function toggleReferenceSelection(filename, event) {
    if (event && event.target.closest('.btn-delete-ref')) return;

    if (state.selectedReferences.has(filename)) {
        state.selectedReferences.delete(filename);
    } else {
        if (state.selectedReferences.size >= 25) {
            alert('Maximum 25 references can be selected.');
            return;
        }
        state.selectedReferences.add(filename);
    }

    const refItem = document.querySelector(`.reference-item[data-filename="${filename}"]`);
    if (refItem) {
        refItem.classList.toggle('selected', state.selectedReferences.has(filename));
        refItem.classList.toggle('deselected', !state.selectedReferences.has(filename));
    }

    updateSelectionCount();
    updateProcessButton();
}

function selectAllReferences() {
    const maxToSelect = Math.min(state.references.length, 25);
    state.references.slice(0, maxToSelect).forEach(ref => state.selectedReferences.add(ref.filename));

    document.querySelectorAll('.reference-item').forEach((item, i) => {
        if (i < maxToSelect) {
            item.classList.add('selected');
            item.classList.remove('deselected');
        }
    });

    if (state.references.length > 25) {
        alert('Selected first 25 references (maximum).');
    }

    updateSelectionCount();
    updateProcessButton();
}

function deselectAllReferences() {
    state.selectedReferences.clear();
    document.querySelectorAll('.reference-item').forEach(item => {
        item.classList.remove('selected');
        item.classList.add('deselected');
    });
    updateSelectionCount();
    updateProcessButton();
}

function updateSelectionCount() {
    if (elements.selectionCount) {
        const count = state.selectedReferences.size;
        elements.selectionCount.textContent = `${count} selected`;
    }
}

window.toggleReferenceSelection = toggleReferenceSelection;
window.selectAllReferences = selectAllReferences;
window.deselectAllReferences = deselectAllReferences;

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

        // Add the uploaded ref directly to state and auto-select
        const newRef = {
            name: data.name,
            filename: data.filename,
            url: data.url,
            type: 'user',
            category: 'Custom',
            deletable: true,
        };

        state.references.push(newRef);
        state.selectedReferences.add(data.filename);

        // Show the uploaded image in the custom ref grid
        renderCustomRefGrid();

        updateSelectionCount();
        updateProcessButton();

    } catch (error) {
        console.error('Reference upload failed:', error);
        alert('Failed to upload reference: ' + error.message);
    }
}

async function deleteReference(filename) {
    if (!confirm('Delete this reference image?')) return;
    try {
        const response = await fetch(`/api/references/${filename}`, { method: 'DELETE' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Delete failed');
        }
        await loadReferences();
        await loadNilutStatus();
    } catch (error) {
        console.error('Delete failed:', error);
        alert('Failed to delete reference: ' + error.message);
    }
}

window.deleteReference = deleteReference;

// Remove a custom reference uploaded in this session
window.removeCustomRef = function(filename) {
    state.references = state.references.filter(r => r.filename !== filename);
    state.selectedReferences.delete(filename);
    renderCustomRefGrid();
    updateSelectionCount();
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
        <div class="custom-ref-item ${state.selectedReferences.has(ref.filename) ? 'selected' : ''}" data-filename="${ref.filename}" onclick="toggleReferenceSelection('${ref.filename}', event)">
            <img src="${ref.url}" alt="${ref.name}">
            <button class="btn-delete-ref" onclick="event.stopPropagation(); removeCustomRef('${ref.filename}')" title="Remove">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
            <span class="ref-name">${ref.name}</span>
        </div>
    `).join('');
}

// ============================================
// Single Image Upload
// ============================================
async function handleFileUpload(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        state.uploadedFile = {
            file: file,
            filename: data.filename,
            url: data.url,
            originalName: data.original_name
        };

        // Show preview
        elements.heroUpload.style.display = 'none';
        elements.imagePreviewContainer.style.display = 'block';
        elements.previewImage.src = data.url;
        elements.imageInfo.textContent = `${data.original_name} — ${(file.size / 1024).toFixed(0)} KB`;

        updateProcessButton();
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Failed to upload image: ' + error.message);
    }
}

function clearUpload() {
    state.uploadedFile = null;
    elements.heroUpload.style.display = 'block';
    elements.imagePreviewContainer.style.display = 'none';
    elements.previewImage.src = '';
    elements.fileInput.value = '';
    updateProcessButton();
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
// Style Source
// ============================================
function setStyleSource(source) {
    state.styleSource = source === 'custom' ? 'reference' : source;

    // Update active states on all 3 source cards
    document.querySelectorAll('.source-card').forEach(card => {
        card.classList.toggle('active', card.dataset.source === source);
    });

    // Show/hide panels
    const refBlock = document.getElementById('reference-block');
    const xmpPanel = document.getElementById('xmp-source-panel');
    const customCard = document.getElementById('custom-ref-card');

    if (refBlock) refBlock.style.display = source === 'reference' ? 'block' : 'none';
    if (xmpPanel) xmpPanel.style.display = source === 'xmp' ? 'block' : 'none';
    if (customCard) customCard.style.display = source === 'custom' ? 'block' : 'none';

    updateProcessButton();
}
window.setStyleSource = setStyleSource;

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
    const hasUpload = state.uploadedFile !== null;
    const hasSelectedRefs = state.selectedReferences.size > 0;
    const hasXMP = state.xmpPreset !== null;
    const canProcess = hasUpload && (state.styleSource === 'xmp' ? hasXMP : hasSelectedRefs);

    if (elements.processBtn) {
        elements.processBtn.disabled = !canProcess;
    }

    if (elements.btnText) {
        if (state.styleSource === 'xmp') {
            elements.btnText.textContent = 'Apply Preset';
        } else {
            const count = state.selectedReferences.size;
            elements.btnText.textContent = count > 0 ? `Generate ${count} Version${count > 1 ? 's' : ''}` : 'Generate Versions';
        }
    }

    if (elements.actionHint) {
        if (canProcess) {
            elements.actionHint.textContent = '';
        } else if (!hasSelectedRefs && state.styleSource === 'reference') {
            elements.actionHint.textContent = 'Choose a style to begin';
        } else if (state.styleSource === 'xmp' && !hasXMP) {
            elements.actionHint.textContent = 'Upload an XMP preset file';
        } else if (!hasUpload) {
            elements.actionHint.textContent = 'Upload your photo';
        }
    }
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Upload dropzone
    if (elements.uploadDropzone) {
        elements.uploadDropzone.addEventListener('click', () => elements.fileInput.click());
        elements.uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.add('drag-over');
        });
        elements.uploadDropzone.addEventListener('dragleave', () => elements.uploadDropzone.classList.remove('drag-over'));
        elements.uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) handleFileUpload(e.dataTransfer.files[0]);
        });
    }

    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
        });
    }

    // Replace image
    if (elements.btnReplace) {
        elements.btnReplace.addEventListener('click', () => {
            clearUpload();
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

    // Source toggle
    if (elements.sourceRefBtn) elements.sourceRefBtn.addEventListener('click', () => setStyleSource('reference'));
    if (elements.sourceXmpBtn) elements.sourceXmpBtn.addEventListener('click', () => setStyleSource('xmp'));
    const customBtn = document.getElementById('source-custom-btn');
    if (customBtn) customBtn.addEventListener('click', () => setStyleSource('custom'));

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
// Processing
// ============================================
async function processAllImages() {
    if (state.styleSource === 'xmp') return processXMP();
    if (state.selectedReferences.size === 0 || !state.uploadedFile) return;

    // UI: show processing state
    elements.btnText.style.display = 'none';
    elements.btnLoading.style.display = 'inline-flex';
    elements.processBtn.disabled = true;
    elements.progressBarContainer.style.display = 'flex';
    setNavStatus('Processing...', true);

    state.processedResults = [];
    const selectedRefs = state.references.filter(ref => state.selectedReferences.has(ref.filename));
    const total = selectedRefs.length;
    let current = 0;

    const inputResult = {
        inputUrl: state.uploadedFile.url,
        originalName: state.uploadedFile.originalName,
        filename: state.uploadedFile.filename,
        referenceResults: {}
    };

    try {
        elements.progressText.textContent = `Processing ${total} references in parallel...`;
        elements.progressBarFill.style.width = `0%`;
        elements.progressLabel.textContent = `0 / ${total}`;

        const processRef = async (ref) => {
            const formData = new FormData();
            formData.append('target_filename', state.uploadedFile.filename);
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

            return { ref, data };
        };

        const results = await Promise.all(selectedRefs.map(processRef));
        for (const { ref, data } of results) {
            inputResult.referenceResults[ref.name] = {
                refName: ref.name,
                refUrl: ref.url,
                refFilename: ref.filename,
                methods: data.outputs || {},
                errors: data.errors || {}
            };
        }

        state.processedResults.push(inputResult);

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
    if (!state.xmpPreset || !state.uploadedFile) return;

    elements.btnText.style.display = 'none';
    elements.btnLoading.style.display = 'inline-flex';
    elements.processBtn.disabled = true;
    setNavStatus('Processing...', true);

    try {
        const formData = new FormData();
        formData.append('target_filename', state.uploadedFile.filename);
        formData.append('xmp_params', JSON.stringify(state.xmpPreset.params));

        const response = await fetch('/api/process-xmp', { method: 'POST', body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Processing failed');
        }

        const data = await response.json();
        const inputResult = {
            inputUrl: state.uploadedFile.url,
            originalName: state.uploadedFile.originalName,
            filename: state.uploadedFile.filename,
            referenceResults: {
                [state.xmpPreset.presetName]: {
                    refName: state.xmpPreset.presetName,
                    refUrl: null,
                    refFilename: state.xmpPreset.filename,
                    methods: { 'xmp_preset': { filename: data.output_filename, url: data.output_url, method_name: 'XMP Preset', time_ms: 0 } },
                    errors: {}
                }
            }
        };

        state.processedResults = [inputResult];
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
