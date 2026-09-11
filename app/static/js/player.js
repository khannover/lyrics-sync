// ─── KAI // LYRICS-SYNC: CYBER AUDIO CONTROLLER ───
import { state } from './state.js';

class AudioController {
    constructor() {
        this.audio = new Audio();
        this.audio.preload = 'auto';

        this.audio.addEventListener('timeupdate', () => {
            const cur = this.audio.currentTime;
            const dur = this.audio.duration || 0;
            state.update('player', p => ({ ...p, currentTime: cur, duration: dur }));
            state.emit('player:timeupdate', { currentTime: cur, duration: dur });
        });

        this.audio.addEventListener('play', () => {
            state.update('player', p => ({ ...p, isPlaying: true }));
            state.emit('player:state', { isPlaying: true });
        });

        this.audio.addEventListener('pause', () => {
            state.update('player', p => ({ ...p, isPlaying: false }));
            state.emit('player:state', { isPlaying: false });
        });

        this.audio.addEventListener('ended', () => {
            state.update('player', p => ({ ...p, isPlaying: false, currentTime: 0 }));
            state.emit('player:state', { isPlaying: false });
        });

        this.audio.addEventListener('loadedmetadata', () => {
            const dur = this.audio.duration || 0;
            state.update('player', p => ({ ...p, duration: dur }));
            state.emit('player:metadata', { duration: dur });
        });
    }

    loadAudio(blobUrl, name = 'Audio Track') {
        this.audio.src = blobUrl;
        this.audio.load();
    }

    play() {
        return this.audio.play();
    }

    pause() {
        this.audio.pause();
    }

    togglePlay() {
        if (this.audio.paused) {
            this.play().catch(console.error);
        } else {
            this.pause();
        }
    }

    seek(timeSeconds) {
        if (!isFinite(timeSeconds)) return;
        this.audio.currentTime = Math.max(0, Math.min(timeSeconds, this.audio.duration || timeSeconds));
    }

    nudge(deltaSeconds) {
        this.seek(this.audio.currentTime + deltaSeconds);
    }

    setPlaybackRate(rate) {
        this.audio.playbackRate = rate;
        state.update('player', p => ({ ...p, playbackRate: rate }));
        state.emit('player:rate', rate);
    }

    getCurrentTime() {
        return this.audio.currentTime;
    }

    getDuration() {
        return this.audio.duration || 0;
    }
}

export const player = new AudioController();

export function formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return '00:00';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
