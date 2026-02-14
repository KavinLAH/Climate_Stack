import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo, { NetInfoState } from '@react-native-community/netinfo';
import type { Sighting } from '../types';

const QUEUE_KEY = 'pestcast_offline_queue';

// ---------------------------------------------------------------------------
// Queue CRUD
// ---------------------------------------------------------------------------

/** Append a sighting to the tail of the offline queue. */
export async function enqueue(sighting: Sighting): Promise<void> {
    const queue = await getAll();
    queue.push(sighting);
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
    console.log(`[offlineQueue] Enqueued sighting ${sighting.id} (${queue.length} pending)`);
}

/** Remove and return the first sighting from the queue. */
export async function dequeue(): Promise<Sighting | undefined> {
    const queue = await getAll();
    const item = queue.shift();
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
    return item;
}

/** Return all queued sightings without mutating the queue. */
export async function getAll(): Promise<Sighting[]> {
    const raw = await AsyncStorage.getItem(QUEUE_KEY);
    if (!raw) return [];
    try {
        return JSON.parse(raw) as Sighting[];
    } catch {
        return [];
    }
}

/** Delete everything in the queue. */
export async function clear(): Promise<void> {
    await AsyncStorage.removeItem(QUEUE_KEY);
}

// ---------------------------------------------------------------------------
// Network-aware auto-drain
// ---------------------------------------------------------------------------

// We dynamically import `syncSighting` to avoid a circular dependency
// (syncService → offlineQueue → syncService).
let _syncFn: ((s: Sighting) => Promise<boolean>) | null = null;

async function getSyncFn() {
    if (!_syncFn) {
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { syncSighting } = require('./syncService') as {
            syncSighting: (s: Sighting) => Promise<boolean>;
        };
        _syncFn = syncSighting;
    }
    return _syncFn;
}

/**
 * Drain the offline queue by attempting to sync each pending sighting.
 * Successfully synced items are removed; failures stay in the queue.
 */
export async function drainQueue(): Promise<void> {
    const queue = await getAll();
    if (queue.length === 0) return;

    console.log(`[offlineQueue] Draining ${queue.length} queued sighting(s)…`);
    const syncSighting = await getSyncFn();

    const remaining: Sighting[] = [];

    for (const sighting of queue) {
        try {
            const ok = await syncSighting(sighting);
            if (!ok) remaining.push(sighting);
        } catch {
            remaining.push(sighting);
        }
    }

    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(remaining));
    console.log(
        `[offlineQueue] Drain complete — ${queue.length - remaining.length} synced, ${remaining.length} still pending`,
    );
}

/**
 * Subscribe to network state changes.  When connectivity is restored,
 * automatically drain the offline queue.
 *
 * @returns An unsubscribe function to tear down the listener.
 */
export function startNetworkListener(): () => void {
    let wasPreviouslyOffline = false;

    const unsubscribe = NetInfo.addEventListener((state: NetInfoState) => {
        const isOnline = state.isConnected && state.isInternetReachable !== false;

        if (isOnline && wasPreviouslyOffline) {
            console.log('[offlineQueue] Connection restored — draining queue');
            drainQueue();
        }

        wasPreviouslyOffline = !isOnline;
    });

    return unsubscribe;
}
