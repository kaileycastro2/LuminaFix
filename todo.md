### Milestone 1
``
Build end-to-end reference → output pipeline. Support single image + batch processing (20–50 images). Deterministic results. Portrait-safe and neon-safe behavior. Runnable code with setup instructions.
``

### Milestone 2
``
Improve aesthetic quality over Milestone 1. Better tone curve and contrast behavior. Improved color separation and depth. Preserve skin tones and highlights. No visual regressions. Client must clearly prefer results over Milestone 1.
``

#### Technical Tasks (Milestone 2):

**Reinhard Improvements:**
- [ ] Add std scaling to Reinhard (full formula: `(pixel - target_mean) / target_std * ref_std + ref_mean`)
- [ ] Add local contrast transfer (not just global std)
- [ ] Implement histogram matching for tone curve transfer

**NILUT Improvements (in order):**
- [ ] Step 1: Fix pairing - use histogram matching instead of L-sorting (root cause fix)
- [ ] Step 2: If still not enough, add hybrid approach (Reinhard + NILUT refinement)
- [ ] Increase training samples (20K → 50K) and epochs (200 → 500)
- [ ] Bigger network (256 hidden, 4 layers)

**Protection:**
- [ ] Preserve skin tones during contrast adjustments
- [ ] Protect highlight/shadow detail from clipping

### Milestone 3
``
Add parameter abstraction (Temperature, Tint, Contrast placeholders). Generate Lightroom-compatible XMP presets. Export edited images + XMP as ZIP. Preview output must align conceptually with exported preset behavior.
``

### Milestone 4
``
Minimal but functional UI/UX (upload, progress, preview, download). Performance improvements where feasible. Bug fixes and basic QA. Clean codebase and documentation. Project handoff. The MVP will use an email-based magic link system. Payments may be collected via one-time payment methods.
``