// ─── KAI // LYRICS-SYNC: TELEPROMPTER & LRC RENDERER ───
import { state } from './state.js';
import { player } from './player.js';

export function parseLrc(lrcText) {
    if (!lrcText) return [];
    const lines = lrcText.split(/\r?\n/);
    const map = [];
    const timeRe = /\[(\d+):(\d+(?:\.\d+)?)\]/g;

    for (const line of lines) {
        const text = line.replace(timeRe, '').trim();
        let match;
        timeRe.lastIndex = 0;
        while ((match = timeRe.exec(line)) !== null) {
            const mins = parseFloat(match[1]);
            const secs = parseFloat(match[2]);
            const time = mins * 60 + secs;
            map.push({ time, text });
        }
    }

    map.sort((a, b) => a.time - b.time);
    return map;
}

export function formatLrcTimestamp(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(2);
    return `[${m.toString().padStart(2, '0')}:${s.padStart(5, '0')}]`;
}

export function initTeleprompter() {
    const container = document.getElementById('stage-lyrics-wrap');
    if (!container) return;

    let syncMap = [];
    let activeIndex = -1;

    function renderLines() {
        if (!syncMap.length) {
            container.innerHTML = `
                <div class="stage-line active" style="font-size:1.1rem; color: var(--cyan);">// WAITING FOR AUDIO & LYRICS</div>
                <div class="stage-line" style="color:var(--text-muted); font-size: 0.85rem;">
                    Run Auto-Sync, load an LRC in the Studio, or drop an MP3 with embedded lyrics.
                </div>
            `;
            return;
        }

        container.innerHTML = '';
        syncMap.forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'stage-line';
            div.dataset.index = idx;
            div.dataset.time = item.time;
            div.textContent = item.text || '♪';
            div.addEventListener('click', () => {
                player.seek(item.time);
                player.play().catch(console.error);
            });
            container.appendChild(div);
        });
    }

    state.on('syncMap', (map) => {
        syncMap = map || [];
        activeIndex = -1;
        renderLines();
    });

    state.on('player:timeupdate', ({ currentTime }) => {
        if (!syncMap.length) return;

        // Find active line: last line with time <= currentTime
        let newActive = -1;
        for (let i = 0; i < syncMap.length; i++) {
            if (syncMap[i].time <= currentTime + 0.15) {
                newActive = i;
            } else {
                break;
            }
        }

        if (newActive !== activeIndex) {
            activeIndex = newActive;
            const lines = container.querySelectorAll('.stage-line');
            lines.forEach((line, idx) => {
                line.classList.remove('active', 'passed');
                if (idx === activeIndex) {
                    line.classList.add('active');
                    // Smooth auto scroll to active line
                    line.scrollIntoView({ behavior: 'smooth', block: 'center' });
                } else if (idx < activeIndex) {
                    line.classList.add('passed');
                }
            });
            state.update('player', p => ({ ...p, activeLineIndex: activeIndex }));
        }
    });

    renderLines();
}
