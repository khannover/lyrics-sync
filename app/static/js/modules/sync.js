// ─── KAI // LYRICS-SYNC: AUTO-SYNC WORKFLOW MODULE ───
import { state } from '../state.js';
import { player } from '../player.js';
import { parseLrc } from '../teleprompter.js';
import { syncLyrics, extractLyricsFromMp3, extractOrTranscribe, offerDownload } from '../api.js';

export function initSyncModule() {
    const audioDropzone = document.getElementById('sync-audio-dropzone');
    const audioInput = document.getElementById('sync-audio-input');
    const audioLoadedPill = document.getElementById('sync-audio-loaded');
    const audioFileName = document.getElementById('sync-audio-filename');
    const audioFileSize = document.getElementById('sync-audio-filesize');
    const btnInspectAudio = document.getElementById('btn-sync-inspect-shortcut');

    const sourceFileRadio = document.getElementById('source-file');
    const sourceTextRadio = document.getElementById('source-text');
    const lyricsFileWrap = document.getElementById('lyrics-file-wrap');
    const lyricsTextWrap = document.getElementById('lyrics-text-wrap');
    const lyricsFileInput = document.getElementById('sync-lyrics-file');
    const lyricsTextInput = document.getElementById('sync-lyrics-text');

    const btnExtractEmbedded = document.getElementById('btn-extract-embedded');
    const btnTranscribeWhisper = document.getElementById('btn-transcribe-whisper');

    const btnStartSync = document.getElementById('btn-start-sync');
    const hudStatus = document.getElementById('sync-hud-status');
    const hudText = document.getElementById('sync-hud-text');

    const resultsCard = document.getElementById('sync-results-card');
    const ratingBadge = document.getElementById('sync-rating-badge');
    const btnDownloadZip = document.getElementById('btn-download-sync-zip');
    const btnCopyLrc = document.getElementById('btn-copy-sync-lrc');

    let loadedAudioFile = null;
    let loadedLyricsFile = null;
    let lastZipBlob = null;
    let lastLrcText = '';

    function checkReady() {
        const hasAudio = !!loadedAudioFile;
        const lyricsMode = sourceTextRadio.checked ? 'text' : 'file';
        const hasLyrics = lyricsMode === 'text' ? !!lyricsTextInput.value.trim() : !!loadedLyricsFile;
        btnStartSync.disabled = !(hasAudio && hasLyrics);
    }

    function setHud(message, type = 'normal') {
        hudStatus.classList.remove('busy', 'error', 'success', 'hidden');
        if (type !== 'hidden') {
            hudStatus.classList.add(type);
            hudText.textContent = message;
        } else {
            hudStatus.classList.add('hidden');
        }
    }

    // ── Audio Upload ──
    function setAudioFile(file) {
        if (!file || !file.name.toLowerCase().endsWith('.mp3')) {
            alert('Please select an .mp3 audio file.');
            return;
        }
        loadedAudioFile = file;
        const url = URL.createObjectURL(file);
        audioFileName.textContent = file.name;
        audioFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        audioDropzone.classList.add('hidden');
        audioLoadedPill.classList.remove('hidden');

        state.set('currentAudio', { file, url, name: file.name, size: file.size });
        player.loadAudio(url, file.name);

        btnExtractEmbedded.disabled = false;
        btnTranscribeWhisper.disabled = false;
        checkReady();
    }

    audioInput.addEventListener('change', (e) => {
        if (e.target.files[0]) setAudioFile(e.target.files[0]);
    });

    audioDropzone.addEventListener('dragover', (e) => { e.preventDefault(); audioDropzone.classList.add('drag-over'); });
    audioDropzone.addEventListener('dragleave', () => audioDropzone.classList.remove('drag-over'));
    audioDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        audioDropzone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) setAudioFile(e.dataTransfer.files[0]);
    });

    btnInspectAudio?.addEventListener('click', () => {
        // Jump directly to Inspector tab
        document.querySelector('.tab-btn[data-tab="inspector"]')?.click();
    });

    // ── Lyrics Source Toggles ──
    sourceFileRadio.addEventListener('change', () => {
        lyricsFileWrap.classList.remove('hidden');
        lyricsTextWrap.classList.add('hidden');
        checkReady();
    });

    sourceTextRadio.addEventListener('change', () => {
        lyricsFileWrap.classList.add('hidden');
        lyricsTextWrap.classList.remove('hidden');
        checkReady();
    });

    lyricsFileInput.addEventListener('change', (e) => {
        loadedLyricsFile = e.target.files[0];
        checkReady();
    });

    lyricsTextInput.addEventListener('input', checkReady);

    // ── Extract Embedded Lyrics ──
    btnExtractEmbedded.addEventListener('click', async () => {
        if (!loadedAudioFile) return;
        setHud('Extracting embedded ID3 tags…', 'busy');
        btnExtractEmbedded.disabled = true;

        try {
            const data = await extractLyricsFromMp3(loadedAudioFile);
            const foundText = data.plain_lyrics || data.timed_lyrics;
            if (foundText) {
                sourceTextRadio.checked = true;
                sourceFileRadio.checked = false;
                lyricsFileWrap.classList.add('hidden');
                lyricsTextWrap.classList.remove('hidden');
                lyricsTextInput.value = foundText;
                setHud(`✅ Extracted lyrics from embedded tags!`, 'success');
                checkReady();
            } else {
                setHud('⚠️ No embedded lyrics found in this MP3.', 'normal');
            }
        } catch (err) {
            setHud(`❌ Extraction failed: ${err.message}`, 'error');
        } finally {
            btnExtractEmbedded.disabled = false;
        }
    });

    // ── Auto-Transcribe with Whisper ──
    btnTranscribeWhisper.addEventListener('click', async () => {
        if (!loadedAudioFile) return;
        setHud('Transcribing vocal track with Whisper… (this may take 20-40s)', 'busy');
        btnTranscribeWhisper.disabled = true;

        try {
            const data = await extractOrTranscribe(loadedAudioFile);
            const text = data.plain_lyrics || data.timed_lyrics;
            if (text) {
                sourceTextRadio.checked = true;
                lyricsFileWrap.classList.add('hidden');
                lyricsTextWrap.classList.remove('hidden');
                lyricsTextInput.value = text;
                setHud(`✅ Transcribed audio using Whisper (${data.source})!`, 'success');
                checkReady();
            } else {
                setHud('⚠️ Transcriber returned empty text.', 'normal');
            }
        } catch (err) {
            setHud(`❌ Transcription failed: ${err.message}`, 'error');
        } finally {
            btnTranscribeWhisper.disabled = false;
        }
    });

    // ── Run Alignment ──
    btnStartSync.addEventListener('click', async () => {
        if (!loadedAudioFile) return;

        const isTextMode = sourceTextRadio.checked;
        const lyricsPayload = isTextMode ? lyricsTextInput.value.trim() : loadedLyricsFile;
        const embedMode = document.querySelector('input[name="embed-mode"]:checked')?.value || 'overwrite';

        btnStartSync.disabled = true;
        setHud('Uploading audio & computing Whisper acoustic alignment…', 'busy');
        resultsCard.classList.add('hidden');

        try {
            const { zipBlob, mp3Blob, lrcText } = await syncLyrics(loadedAudioFile, lyricsPayload, embedMode);
            lastZipBlob = zipBlob;
            lastLrcText = lrcText;

            // Load synced MP3 into player
            const syncedAudioUrl = URL.createObjectURL(mp3Blob);
            player.loadAudio(syncedAudioUrl, loadedAudioFile.name.replace('.mp3', '_synced.mp3'));

            // Parse LRC and load into teleprompter
            const map = parseLrc(lrcText);
            state.set('syncMap', map);
            state.set('currentLrcText', lrcText);

            resultsCard.classList.remove('hidden');
            setHud(`✅ Alignment complete! ${map.length} timestamped lines generated.`, 'success');

        } catch (err) {
            setHud(`❌ Alignment error: ${err.message}`, 'error');
            console.error(err);
        } finally {
            btnStartSync.disabled = false;
            checkReady();
        }
    });

    btnDownloadZip.addEventListener('click', () => {
        if (!lastZipBlob || !loadedAudioFile) return;
        offerDownload(lastZipBlob, loadedAudioFile.name.replace('.mp3', '_synced.zip'));
    });

    btnCopyLrc.addEventListener('click', () => {
        if (!lastLrcText) return;
        navigator.clipboard.writeText(lastLrcText).then(() => {
            const orig = btnCopyLrc.textContent;
            btnCopyLrc.textContent = 'COPIED!';
            setTimeout(() => btnCopyLrc.textContent = orig, 1500);
        });
    });
}
