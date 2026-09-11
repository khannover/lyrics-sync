// ─── KAI // LYRICS-SYNC: MASTER APPLICATION ENTRY ───
import { state } from './state.js';
import { player, formatTime } from './player.js';
import { initTeleprompter } from './teleprompter.js';
import { initSyncModule } from './modules/sync.js';
import { initInspectorModule } from './modules/inspector.js';
import { initEditorModule } from './modules/editor.js';
import { initTelemetryModule } from './modules/telemetry.js';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Subsystems
    initTeleprompter();
    initSyncModule();
    initInspectorModule();
    initEditorModule();
    initTelemetryModule();

    // 2. Tab Navigation
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.deck-tab-panel');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active', 'hidden'));

            btn.classList.add('active');
            tabPanels.forEach(p => {
                if (p.id === `tab-${target}`) {
                    p.classList.add('active');
                } else {
                    p.classList.add('hidden');
                }
            });
            state.set('activeTab', target);
        });
    });

    // 3. Cyber Audio Player Controls
    const btnPlayToggle = document.getElementById('player-play-toggle');
    const timeCur = document.getElementById('player-cur-time');
    const timeTotal = document.getElementById('player-total-time');
    const scrubberBar = document.getElementById('player-scrubber-bar');
    const scrubberFill = document.getElementById('player-scrubber-fill');
    const speedChips = document.querySelectorAll('.speed-chip');

    btnPlayToggle.addEventListener('click', () => player.togglePlay());

    state.on('player:state', ({ isPlaying }) => {
        btnPlayToggle.textContent = isPlaying ? '❚❚' : '▶';
    });

    state.on('player:timeupdate', ({ currentTime, duration }) => {
        timeCur.textContent = formatTime(currentTime);
        timeTotal.textContent = formatTime(duration);
        if (duration > 0) {
            const pct = (currentTime / duration) * 100;
            scrubberFill.style.width = `${pct}%`;
        }
    });

    scrubberBar.addEventListener('click', (e) => {
        const rect = scrubberBar.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const dur = player.getDuration();
        if (dur > 0) {
            player.seek(ratio * dur);
        }
    });

    speedChips.forEach(chip => {
        chip.addEventListener('click', () => {
            speedChips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const rate = parseFloat(chip.dataset.speed);
            player.setPlaybackRate(rate);
        });
    });

    // Global spacebar to play/pause (when not typing in textarea or input)
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && e.target.tagName !== 'TEXTAREA' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            player.togglePlay();
        }
    });
});
