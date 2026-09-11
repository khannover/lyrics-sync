// ─── KAI // LYRICS-SYNC: TELEMETRY & API CHEATSHEET MODULE ───
import { getHealth, getQueue } from '../api.js';

export function initTelemetryModule() {
    const pillStatus = document.getElementById('pill-service-status');
    const pillQueue = document.getElementById('pill-queue-status');
    const pillDisk = document.getElementById('pill-disk-status');

    const statUptime = document.getElementById('telemetry-stat-status');
    const statVersion = document.getElementById('telemetry-stat-version');
    const statSlots = document.getElementById('telemetry-stat-slots');
    const statWaiting = document.getElementById('telemetry-stat-waiting');
    const statCompleted = document.getElementById('telemetry-stat-completed');
    const statDiskFree = document.getElementById('telemetry-stat-disk-free');

    async function poll() {
        try {
            const [health, queue] = await Promise.all([
                getHealth().catch(() => null),
                getQueue().catch(() => null)
            ]);

            if (health) {
                pillStatus.className = 'telemetry-pill online';
                pillStatus.innerHTML = '<span class="telemetry-dot"></span> SERVICE ONLINE';
                if (statUptime) statUptime.textContent = health.status.toUpperCase();
                if (statVersion) statVersion.textContent = `v${health.version || '1.0.0'}`;

                if (health.disk && pillDisk) {
                    pillDisk.textContent = `DISK: ${health.disk.free_gb} GB FREE`;
                    if (statDiskFree) statDiskFree.textContent = `${health.disk.free_gb} GB / ${health.disk.total_gb} GB`;
                }

                if (health.async_jobs && statCompleted) {
                    statCompleted.textContent = health.async_jobs.completed || 0;
                }
            } else {
                pillStatus.className = 'telemetry-pill';
                pillStatus.innerHTML = '<span class="telemetry-dot" style="background:var(--coral)"></span> OFFLINE';
            }

            if (queue && pillQueue) {
                pillQueue.textContent = `QUEUE: ${queue.waiting_jobs} / ${queue.total_slots} SLOTS`;
                if (statSlots) statSlots.textContent = queue.total_slots;
                if (statWaiting) statWaiting.textContent = queue.waiting_jobs;
            }
        } catch (err) {
            console.error('Telemetry polling error:', err);
        }
    }

    // Initial poll + interval
    poll();
    setInterval(poll, 12000);
}
