// ─── KAI // LYRICS-SYNC: AUDIO INSPECTOR & AUDIT MODULE ───
import { state } from '../state.js';
import { player } from '../player.js';
import { analyzeAudio, extractLyricsFromMp3 } from '../api.js';

export function initInspectorModule() {
    const dropzone = document.getElementById('inspector-dropzone');
    const input = document.getElementById('inspector-input');
    const loadedPill = document.getElementById('inspector-loaded-pill');
    const filenameLabel = document.getElementById('inspector-filename');
    const btnUseCurrent = document.getElementById('btn-inspector-use-current');
    const btnRunAudit = document.getElementById('btn-run-audio-audit');
    const auditHud = document.getElementById('inspector-hud');
    const auditHudText = document.getElementById('inspector-hud-text');

    const dashboardGrid = document.getElementById('inspector-dashboard');

    let inspectFile = null;

    function setInspectFile(file) {
        if (!file || !file.name.toLowerCase().endsWith('.mp3')) {
            alert('Please select an .mp3 file for inspection.');
            return;
        }
        inspectFile = file;
        filenameLabel.textContent = file.name;
        dropzone.classList.add('hidden');
        loadedPill.classList.remove('hidden');
        btnRunAudit.disabled = false;
    }

    input.addEventListener('change', (e) => {
        if (e.target.files[0]) setInspectFile(e.target.files[0]);
    });

    dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) setInspectFile(e.dataTransfer.files[0]);
    });

    // "Use current audio" button
    state.on('currentAudio', (audio) => {
        if (audio && audio.file) {
            btnUseCurrent.classList.remove('hidden');
        }
    });

    btnUseCurrent.addEventListener('click', () => {
        const audio = state.get('currentAudio');
        if (audio && audio.file) {
            setInspectFile(audio.file);
        }
    });

    btnRunAudit.addEventListener('click', async () => {
        if (!inspectFile) return;

        btnRunAudit.disabled = true;
        auditHud.classList.remove('hidden', 'error', 'success');
        auditHud.classList.add('busy');
        auditHudText.textContent = 'Executing full audit: acoustic profiling, neural genres, AcoustID hash, AI watermarks & safety…';
        dashboardGrid.classList.add('hidden');

        try {
            // Run analyzeAudio and extractLyrics concurrently
            const [analysis, lyricsData] = await Promise.all([
                analyzeAudio(inspectFile, { includeSignal: true, includeFingerprint: true, includeAiProvenance: true, topKGenres: 5 }),
                extractLyricsFromMp3(inspectFile).catch(() => ({ content_rating: null }))
            ]);

            renderDashboard(analysis, lyricsData.content_rating);

            auditHud.classList.remove('busy');
            auditHud.classList.add('success');
            auditHudText.textContent = `✅ Full audit completed for ${inspectFile.name}!`;
            dashboardGrid.classList.remove('hidden');

        } catch (err) {
            auditHud.classList.remove('busy');
            auditHud.classList.add('error');
            auditHudText.textContent = `❌ Audit error: ${err.message}`;
            console.error(err);
        } finally {
            btnRunAudit.disabled = false;
        }
    });

    function renderDashboard(data, fallbackContentRating) {
        // ── Card 1: Copyright & Fingerprint ──
        const fpBox = document.getElementById('card-fingerprint-box');
        const fpStatus = document.getElementById('card-fingerprint-status');
        const fpDuration = document.getElementById('card-fingerprint-duration');
        const btnCopyFp = document.getElementById('btn-copy-fingerprint');

        const rawFp = data.copyright_fingerprint?.acoustid_fingerprint || data.fingerprint || null;
        const durSec = data.copyright_fingerprint?.duration_sec || data.duration_sec || null;

        if (rawFp) {
            fpBox.textContent = rawFp;
            fpStatus.textContent = 'ACOUSTID READY';
            fpStatus.className = 'card-badge card-badge--cyan';
        } else {
            fpBox.textContent = 'No Chromaprint generated (fpcalc not detected or audio too short).';
            fpStatus.textContent = 'UNAVAILABLE';
            fpStatus.className = 'card-badge card-badge--amber';
        }
        fpDuration.textContent = durSec ? `${durSec.toFixed(1)}s` : '--';

        btnCopyFp.onclick = () => {
            if (!rawFp) return;
            navigator.clipboard.writeText(rawFp).then(() => {
                const orig = btnCopyFp.textContent;
                btnCopyFp.textContent = 'COPIED!';
                setTimeout(() => btnCopyFp.textContent = orig, 1500);
            });
        };

        // ── Card 2: AI Provenance & Watermarks ──
        const aiVerdict = document.getElementById('card-ai-verdict');
        const aiVerdictBox = document.getElementById('card-ai-verdict-box');
        const aiConfidenceVal = document.getElementById('card-ai-confidence-val');
        const aiConfidenceBar = document.getElementById('card-ai-confidence-bar');
        const aiSunoStatus = document.getElementById('card-ai-suno');
        const aiUdioStatus = document.getElementById('card-ai-udio');
        const aiCutoff = document.getElementById('card-ai-cutoff');
        const aiReasoning = document.getElementById('card-ai-reasoning');

        const prov = data.ai_provenance || {};
        const isAi = prov.is_synthetic || prov.is_ai_generated || false;
        const confPct = Math.round((prov.confidence || 0) * 100);

        const aiCard = aiVerdictBox.closest('.audit-card');
        const aiCardBadge = aiCard?.querySelector('.card-badge');

        if (isAi) {
            aiVerdictBox.className = 'verdict-box verdict-box--purple';
            aiVerdict.textContent = '✨ AI SYNTHESIS VERIFIED';
            aiConfidenceBar.className = 'meter-fill meter-fill--purple';
            if (aiCard) aiCard.className = 'audit-card audit-card--purple';
            if (aiCardBadge) {
                aiCardBadge.className = 'card-badge card-badge--purple';
                aiCardBadge.textContent = prov.detected_source ? prov.detected_source.toUpperCase() : 'AI GENERATED';
            }
        } else {
            aiVerdictBox.className = 'verdict-box verdict-box--cyan';
            aiVerdict.textContent = '🌱 ORGANIC / NATURAL PROVENANCE';
            aiConfidenceBar.className = 'meter-fill meter-fill--cyan';
            if (aiCard) aiCard.className = 'audit-card audit-card--cyan';
            if (aiCardBadge) {
                aiCardBadge.className = 'card-badge card-badge--cyan';
                aiCardBadge.textContent = 'ORGANIC';
            }
        }

        aiConfidenceVal.textContent = `${confPct}%`;
        aiConfidenceBar.style.width = `${confPct}%`;

        const sunoHit = prov.detected_source === 'suno' || prov.signals?.metadata_marker || prov.suno_watermark_detected;
        const udioHit = prov.detected_source === 'udio' || prov.udio_artifacts_detected;

        aiSunoStatus.textContent = sunoHit ? 'DETECTED' : 'CLEAR';
        aiSunoStatus.className = `stat-val ${sunoHit ? 'accent-purple' : 'accent-cyan'}`;

        aiUdioStatus.textContent = udioHit ? 'DETECTED' : 'CLEAR';
        aiUdioStatus.className = `stat-val ${udioHit ? 'accent-purple' : 'accent-cyan'}`;

        if (prov.signals?.spectral_cutoff_khz) {
            aiCutoff.textContent = `${prov.signals.spectral_cutoff_khz} kHz`;
        } else if (prov.spectral_cutoff_hz) {
            aiCutoff.textContent = `${(prov.spectral_cutoff_hz / 1000).toFixed(1)} kHz`;
        } else {
            aiCutoff.textContent = 'Full Range (20+ kHz)';
        }

        if (prov.signals?.details && prov.signals.details.length) {
            aiReasoning.textContent = prov.signals.details.join(' • ');
        } else {
            aiReasoning.textContent = prov.reasoning || (isAi ? 'Synthetic artifacts or marker found' : 'Natural acoustic characteristics detected.');
        }

        // ── Card 3: Content Safety ──
        const safetyVerdictBox = document.getElementById('card-safety-verdict-box');
        const safetyVerdict = document.getElementById('card-safety-verdict');
        const flaggedWordsList = document.getElementById('card-flagged-words');
        const profanityScore = document.getElementById('card-score-profanity');
        const sexualScore = document.getElementById('card-score-sexual');

        const safety = data.content_rating || fallbackContentRating || {};
        const isExplicit = safety.is_explicit || (safety.rating && String(safety.rating).toLowerCase() === 'explicit');

        if (isExplicit) {
            safetyVerdictBox.className = 'verdict-box verdict-box--coral';
            safetyVerdict.textContent = '🔞 EXPLICIT CONTENT DETECTED';
        } else {
            safetyVerdictBox.className = 'verdict-box verdict-box--emerald';
            safetyVerdict.textContent = '🟢 CLEAN / ALL-AGES VERIFIED';
        }

        flaggedWordsList.innerHTML = '';
        const words = safety.matched_terms || safety.flagged_words || [];
        if (words.length) {
            words.forEach(w => {
                const tag = document.createElement('span');
                tag.className = 'tag-pill';
                tag.textContent = w;
                flaggedWordsList.appendChild(tag);
            });
        } else {
            flaggedWordsList.innerHTML = '<span style="color:var(--text-muted); font-size:0.7rem;">None detected</span>';
        }

        const breakdown = safety.category_breakdown || {};
        profanityScore.textContent = `${safety.matched_count || breakdown.profanity || 0} hits`;
        sexualScore.textContent = `${breakdown.sexual || 0} hits`;

        // ── Card 4: Musical DNA & Acoustic Profile ──
        const bpmVal = document.getElementById('card-bpm-val');
        const keyVal = document.getElementById('card-key-val');
        const energyVal = document.getElementById('card-energy-val');
        const energyBar = document.getElementById('card-energy-bar');
        const danceVal = document.getElementById('card-dance-val');
        const lufsVal = document.getElementById('card-lufs-val');
        const genresList = document.getElementById('card-genres-list');

        const rawBpm = data.bpm ?? data.tier_breakdown?.tier2_signal?.bpm;
        bpmVal.textContent = rawBpm ? Math.round(rawBpm) : '--';

        const rawKey = data.key || data.tier_breakdown?.tier2_signal?.key;
        keyVal.textContent = rawKey ? `KEY: ${rawKey}` : 'KEY: --';

        let energyPct = 50;
        let energyLabel = '--';
        const rawEnergy = data.energy ?? data.tier_breakdown?.tier2_signal?.energy;

        if (typeof rawEnergy === 'number') {
            energyPct = Math.round(rawEnergy * 100);
            energyLabel = `${energyPct}%`;
        } else if (typeof rawEnergy === 'string') {
            const lower = rawEnergy.toLowerCase();
            if (lower === 'high') { energyPct = 85; energyLabel = 'High (85%)'; }
            else if (lower === 'medium' || lower === 'mid') { energyPct = 55; energyLabel = 'Medium (55%)'; }
            else if (lower === 'low') { energyPct = 25; energyLabel = 'Low (25%)'; }
            else { energyPct = 50; energyLabel = rawEnergy; }
        }

        energyVal.textContent = energyLabel;
        energyBar.style.width = `${energyPct}%`;

        const dance = data.danceability ?? data.tier_breakdown?.tier2_signal?.danceability;
        danceVal.textContent = dance ? `${Math.round(dance * 100)}%` : '--';

        const lufs = data.loudness_lufs ?? data.tier_breakdown?.tier2_signal?.loudness_lufs;
        lufsVal.textContent = lufs ? `${lufs.toFixed(1)} LUFS` : '--';

        genresList.innerHTML = '';
        const neural = data.tier_breakdown?.tier3_neural?.top_genres || data.neural_genres || [];

        if (neural.length) {
            neural.slice(0, 4).forEach(g => {
                const genreName = g.genre || g.name || 'Genre';
                const score = g.score ?? g.probability ?? 0;
                const pct = Math.round(score * 100);
                const row = document.createElement('div');
                row.className = 'meter-row';
                row.innerHTML = `
                    <div class="meter-header">
                        <span>${genreName}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="meter-track">
                        <div class="meter-fill meter-fill--purple" style="width: ${pct}%;"></div>
                    </div>
                `;
                genresList.appendChild(row);
            });
        } else if (data.primary_genre) {
            genresList.innerHTML = `<div class="stat-row"><span class="stat-name">Genre</span><span class="stat-val accent-cyan">${data.primary_genre}</span></div>`;
        }
    }
}
