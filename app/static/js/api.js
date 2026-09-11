// ─── KAI // LYRICS-SYNC: API CLIENT ───

export async function getHealth() {
    const res = await fetch('/health');
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
}

export async function getQueue() {
    const res = await fetch('/queue');
    if (!res.ok) throw new Error(`Queue check failed: ${res.statusText}`);
    return await res.json();
}

export async function extractLyricsFromMp3(mp3File) {
    const form = new FormData();
    form.append('mp3', mp3File);
    const res = await fetch('/lyrics/from-mp3', { method: 'POST', body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Failed to extract embedded lyrics');
    }
    return await res.json();
}

export async function extractOrTranscribe(mp3File) {
    const form = new FormData();
    form.append('mp3', mp3File);
    const res = await fetch('/lyrics/extract', { method: 'POST', body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Failed to transcribe audio lyrics');
    }
    return await res.json();
}

export async function analyzeAudio(mp3File, options = {}) {
    const form = new FormData();
    form.append('mp3', mp3File);
    form.append('include_signal', options.includeSignal !== false);
    form.append('include_fingerprint', options.includeFingerprint !== false);
    form.append('include_ai_provenance', options.includeAiProvenance !== false);
    form.append('top_k_genres', options.topKGenres || 5);
    if (options.forceNeural) form.append('force_neural', 'true');

    const res = await fetch('/audio/analyze', { method: 'POST', body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Audio analysis failed');
    }
    return await res.json();
}

export async function syncLyrics(mp3File, lyricsPayload, embedMode = 'overwrite') {
    const form = new FormData();
    form.append('mp3', mp3File);

    let lyricsBlob;
    let filename = 'lyrics.txt';
    if (lyricsPayload instanceof File) {
        lyricsBlob = lyricsPayload;
        filename = lyricsPayload.name;
    } else if (lyricsPayload instanceof Blob) {
        lyricsBlob = lyricsPayload;
    } else {
        lyricsBlob = new Blob([String(lyricsPayload ?? '')], { type: 'text/plain;charset=utf-8' });
    }

    form.append('lyrics', lyricsBlob, filename);
    form.append('embed_mode', embedMode);

    const res = await fetch('/sync', { method: 'POST', body: form });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Lyrics sync failed');
    }

    const blob = await res.blob();
    const entries = await loadZip(blob);

    let mp3Blob = null;
    let lrcText = null;

    for (const [name, data] of entries) {
        if (name.endsWith('.mp3')) mp3Blob = data;
        if (name.endsWith('.lrc')) lrcText = await data.text();
    }

    if (!mp3Blob || !lrcText) {
        throw new Error('ZIP missing expected synced audio or LRC file');
    }

    return { zipBlob: blob, mp3Blob, lrcText };
}

export async function loadZip(blob) {
    const buf = await blob.arrayBuffer();
    const view = new DataView(buf);
    const entries = [];
    let offset = 0;

    while (offset < buf.byteLength - 4) {
        const sig = view.getUint32(offset, true);
        if (sig !== 0x04034b50) break; // Local file header signature

        const compMethod = view.getUint16(offset + 8, true);
        const compSize = view.getUint32(offset + 18, true);
        const nameLen = view.getUint16(offset + 26, true);
        const extraLen = view.getUint16(offset + 28, true);
        const nameBytes = new Uint8Array(buf, offset + 30, nameLen);
        const fileName = new TextDecoder().decode(nameBytes);
        const dataStart = offset + 30 + nameLen + extraLen;
        const fileData = buf.slice(dataStart, dataStart + compSize);

        if (compMethod === 0) {
            // Stored
            entries.push([fileName, new Blob([fileData])]);
        } else if (compMethod === 8) {
            // Deflated
            const ds = new DecompressionStream('deflate-raw');
            const writer = ds.writable.getWriter();
            writer.write(new Uint8Array(fileData));
            writer.close();
            const decompressed = await new Response(ds.readable).blob();
            entries.push([fileName, decompressed]);
        }
        offset = dataStart + compSize;
    }
    return entries;
}

export function offerDownload(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
}
