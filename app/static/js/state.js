// ─── KAI // LYRICS-SYNC: REACTIVE STATE STORE ───

class StateStore {
    constructor() {
        this.listeners = new Map();
        this.data = {
            currentAudio: null,       // { file, url, name, size, duration }
            currentLyricsText: '',    // string
            currentLrcText: '',       // string
            syncMap: [],              // [{ time: number, text: string }]
            embedMode: 'overwrite',   // 'overwrite' | 'sylt_only'
            activeTab: 'sync',        // 'sync' | 'inspector' | 'editor' | 'telemetry'
            syncResults: null,        // { zipBlob, mp3Blob, lrcText, contentRating }
            inspectorResults: null,   // { ...audio/analyze }
            player: {
                isPlaying: false,
                currentTime: 0,
                duration: 0,
                playbackRate: 1.0,
                activeLineIndex: -1
            }
        };
    }

    get(key) {
        return this.data[key];
    }

    set(key, value) {
        this.data[key] = value;
        this.emit(key, value);
    }

    update(key, fn) {
        const newVal = fn(this.data[key]);
        this.set(key, newVal);
    }

    on(event, handler) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(handler);
        return () => this.listeners.get(event)?.delete(handler);
    }

    emit(event, payload) {
        const handlers = this.listeners.get(event);
        if (handlers) {
            for (const h of handlers) {
                try { h(payload); } catch (err) { console.error(err); }
            }
        }
    }
}

export const state = new StateStore();
