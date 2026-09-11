// ─── KAI // LYRICS-SYNC: MANUAL LRC STUDIO MODULE ───
import { state } from '../state.js';
import { player } from '../player.js';
import { parseLrc, formatLrcTimestamp } from '../teleprompter.js';
import { offerDownload } from '../api.js';

export function initEditorModule() {
    const rawTextarea = document.getElementById('editor-raw-textarea');
    const tableBody = document.getElementById('editor-table-body');
    const btnSwitchView = document.getElementById('btn-editor-switch-view');
    const tableView = document.getElementById('editor-table-view');
    const textView = document.getElementById('editor-text-view');

    const btnAddLine = document.getElementById('btn-editor-add-line');
    const btnNudgeAllMinus = document.getElementById('btn-editor-nudge-all-minus');
    const btnNudgeAllPlus = document.getElementById('btn-editor-nudge-all-plus');
    const btnExportLrc = document.getElementById('btn-editor-export');
    const btnSendToSync = document.getElementById('btn-editor-send-to-sync');

    let isTableView = true;
    let items = [];

    function updateFromSyncMap(map) {
        items = (map || []).map(x => ({ ...x }));
        render();
    }

    state.on('syncMap', updateFromSyncMap);

    btnSwitchView.addEventListener('click', () => {
        isTableView = !isTableView;
        btnSwitchView.textContent = isTableView ? 'SWITCH TO RAW LRC' : 'SWITCH TO TABLE';
        if (isTableView) {
            // Parse from raw
            items = parseLrc(rawTextarea.value);
            state.set('syncMap', items);
            tableView.classList.remove('hidden');
            textView.classList.add('hidden');
            renderTable();
        } else {
            // Build raw
            rawTextarea.value = buildRawLrc();
            tableView.classList.add('hidden');
            textView.classList.remove('hidden');
        }
    });

    function buildRawLrc() {
        return items.map(x => `${formatLrcTimestamp(x.time)} ${x.text}`).join('\n');
    }

    function render() {
        if (isTableView) renderTable();
        else rawTextarea.value = buildRawLrc();
    }

    function renderTable() {
        tableBody.innerHTML = '';
        if (!items.length) {
            tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding:20px; color:var(--text-muted);">No lyric lines loaded yet. Paste an LRC or run Auto-Sync.</td></tr>';
            return;
        }

        items.forEach((item, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="width: 32px; text-align:center;">
                    <button class="player-btn btn-play-line" style="width:22px; height:22px; font-size:10px;">▶</button>
                </td>
                <td style="width: 90px;">
                    <input type="text" class="input-time" value="${formatLrcTimestamp(item.time).replace(/[\[\]]/g, '')}" style="width:100%; background:transparent; border:1px solid var(--border-subtle); color:var(--cyan); font-family:var(--font-mono); font-size:0.75rem; padding:2px 4px; border-radius:3px;">
                </td>
                <td>
                    <input type="text" class="input-text" value="${item.text || ''}" style="width:100%; background:transparent; border:1px solid var(--border-subtle); color:var(--text-primary); font-family:var(--font-mono); font-size:0.75rem; padding:2px 4px; border-radius:3px;">
                </td>
                <td style="width: 130px; text-align:right;">
                    <button class="btn-cyber btn-cyber--ghost btn-cyber--sm btn-set-now" title="Set to current playhead">SET</button>
                    <button class="btn-cyber btn-cyber--ghost btn-cyber--sm btn-minus-100" title="-100ms">-0.1s</button>
                    <button class="btn-cyber btn-cyber--ghost btn-cyber--sm btn-plus-100" title="+100ms">+0.1s</button>
                    <button class="btn-cyber btn-cyber--coral btn-cyber--sm btn-del" title="Delete">✕</button>
                </td>
            `;

            tr.querySelector('.btn-play-line').onclick = () => {
                player.seek(item.time);
                player.play().catch(console.error);
            };

            tr.querySelector('.btn-set-now').onclick = () => {
                item.time = player.getCurrentTime();
                items.sort((a, b) => a.time - b.time);
                state.set('syncMap', items);
            };

            tr.querySelector('.btn-minus-100').onclick = () => {
                item.time = Math.max(0, item.time - 0.1);
                items.sort((a, b) => a.time - b.time);
                state.set('syncMap', items);
            };

            tr.querySelector('.btn-plus-100').onclick = () => {
                item.time += 0.1;
                items.sort((a, b) => a.time - b.time);
                state.set('syncMap', items);
            };

            tr.querySelector('.btn-del').onclick = () => {
                items.splice(idx, 1);
                state.set('syncMap', items);
            };

            const textInput = tr.querySelector('.input-text');
            textInput.onchange = () => {
                item.text = textInput.value;
                state.set('syncMap', items);
            };

            tableBody.appendChild(tr);
        });
    }

    btnAddLine.addEventListener('click', () => {
        const curTime = player.getCurrentTime();
        items.push({ time: curTime, text: 'New Lyric Line' });
        items.sort((a, b) => a.time - b.time);
        state.set('syncMap', items);
    });

    btnNudgeAllMinus.addEventListener('click', () => {
        items.forEach(x => x.time = Math.max(0, x.time - 0.2));
        state.set('syncMap', items);
    });

    btnNudgeAllPlus.addEventListener('click', () => {
        items.forEach(x => x.time += 0.2);
        state.set('syncMap', items);
    });

    btnExportLrc.addEventListener('click', () => {
        const lrc = buildRawLrc();
        const blob = new Blob([lrc], { type: 'text/plain;charset=utf-8' });
        offerDownload(blob, 'edited_lyrics.lrc');
    });

    btnSendToSync.addEventListener('click', () => {
        const lrc = buildRawLrc();
        const syncTextarea = document.getElementById('sync-lyrics-text');
        const sourceTextRadio = document.getElementById('source-text');
        if (syncTextarea && sourceTextRadio) {
            sourceTextRadio.click();
            syncTextarea.value = lrc;
            document.querySelector('.tab-btn[data-tab="sync"]')?.click();
        }
    });
}
