import React, { useCallback, useEffect, useState } from 'react';
import {
    StyleSheet,
    Text,
    View,
    TouchableOpacity,
    ActivityIndicator,
    SafeAreaView,
    ScrollView,
    Image,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import 'react-native-get-random-values';
import { v4 as uuidv4 } from 'uuid';

import CameraViewfinder from '../components/CameraViewfinder';
import { useClassifier } from '../hooks/useClassifier';
import { useLocation } from '../hooks/useLocation';
import { syncSighting } from '../services/syncService';
import { startNetworkListener, drainQueue } from '../services/offlineQueue';
import type { AppScreen, ClassificationResult, Sighting, Severity } from '../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const DEVICE_ID = 'pestcast-dev-001'; // In prod, use a unique device fingerprint.

type Props = {
    /** Optional callback to return to the map dashboard. */
    onClose?: () => void;
};

export default function PestCaptureScreen({ onClose }: Props) {
    // ---- State ----
    const [screen, setScreen] = useState<AppScreen>('camera');
    const [capturedUri, setCapturedUri] = useState<string | null>(null);
    const [results, setResults] = useState<ClassificationResult[]>([]);
    const [syncing, setSyncing] = useState(false);
    const [syncResult, setSyncResult] = useState<'success' | 'queued' | null>(null);

    // ---- Hooks ----
    const { classify, isModelReady } = useClassifier();
    const { getCurrentPosition } = useLocation();

    // ---- Boot: start offline-queue network listener ----
    useEffect(() => {
        const unsub = startNetworkListener();
        drainQueue();
        return unsub;
    }, []);

    // ---- Handlers ----

    /** Camera captured a photo → run classification. */
    const handleCapture = useCallback(
        async (uri: string) => {
            setCapturedUri(uri);
            setScreen('classifying');

            const preds = await classify(uri);
            setResults(preds);
            setScreen('results');
        },
        [classify],
    );

    /** User confirms the top result → grab GPS + sync to Supabase. */
    const handleConfirm = useCallback(
        async (pestType: string, confidence: number) => {
            setScreen('syncing');
            setSyncing(true);

            const coords = await getCurrentPosition();

            const sighting: Sighting = {
                id: uuidv4(),
                image_url: capturedUri ?? '',
                pest_type: pestType,
                confidence,
                lat: coords?.lat ?? 0,
                lng: coords?.lng ?? 0,
                severity: deriveSeverity(confidence),
                device_id: DEVICE_ID,
                created_at: new Date().toISOString(),
                synced: false,
            };

            const ok = await syncSighting(sighting);
            setSyncResult(ok ? 'success' : 'queued');
            setSyncing(false);
            setScreen('done');
        },
        [capturedUri, getCurrentPosition],
    );

    /** Reset to camera for a new capture. */
    const handleReset = () => {
        setCapturedUri(null);
        setResults([]);
        setSyncResult(null);
        setScreen('camera');
    };

    // ---- Render by screen ----

    if (screen === 'camera') {
        return (
            <View style={styles.full}>
                <StatusBar style="light" />

                {/* Back button to map */}
                {onClose && (
                    <TouchableOpacity style={styles.backBtn} onPress={onClose}>
                        <Text style={styles.backBtnText}>{'← Map'}</Text>
                    </TouchableOpacity>
                )}

                <CameraViewfinder onCapture={handleCapture} />

                {!isModelReady && (
                    <View style={styles.modelBanner}>
                        <Text style={styles.modelBannerText}>{'⏳ Loading classifier model…'}</Text>
                    </View>
                )}
            </View>
        );
    }

    if (screen === 'classifying') {
        return (
            <SafeAreaView style={styles.centerDark}>
                <StatusBar style="light" />
                <ActivityIndicator size="large" color="#22c55e" />
                <Text style={styles.statusText}>Identifying pest…</Text>
            </SafeAreaView>
        );
    }

    if (screen === 'results') {
        return (
            <SafeAreaView style={styles.containerDark}>
                <StatusBar style="light" />
                <ScrollView contentContainerStyle={styles.resultScroll}>
                    {capturedUri && (
                        <Image source={{ uri: capturedUri }} style={styles.previewImage} />
                    )}

                    <Text style={styles.heading}>Classification Results</Text>

                    {results.map((r, i) => (
                        <TouchableOpacity
                            key={r.label}
                            style={[styles.resultCard, i === 0 && styles.resultCardTop]}
                            onPress={() => handleConfirm(r.label, r.confidence)}
                            activeOpacity={0.7}
                        >
                            <View style={styles.resultRow}>
                                <Text style={styles.resultLabel}>
                                    {i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'} {r.label}
                                </Text>
                                <Text style={styles.resultConf}>{(r.confidence * 100).toFixed(1)}%</Text>
                            </View>
                            <View style={styles.confBar}>
                                <View
                                    style={[styles.confFill, { width: `${r.confidence * 100}%` }]}
                                />
                            </View>
                        </TouchableOpacity>
                    ))}

                    <Text style={styles.hint}>Tap a result to confirm and report sighting</Text>
                </ScrollView>
            </SafeAreaView>
        );
    }

    if (screen === 'syncing') {
        return (
            <SafeAreaView style={styles.centerDark}>
                <StatusBar style="light" />
                <ActivityIndicator size="large" color="#22c55e" />
                <Text style={styles.statusText}>
                    {syncing ? 'Syncing sighting…' : 'Done!'}
                </Text>
            </SafeAreaView>
        );
    }

    // screen === 'done'
    return (
        <SafeAreaView style={styles.centerDark}>
            <StatusBar style="light" />
            <Text style={styles.doneIcon}>
                {syncResult === 'success' ? '✅' : '📶'}
            </Text>
            <Text style={styles.doneTitle}>
                {syncResult === 'success' ? 'Sighting Reported!' : 'Saved Offline'}
            </Text>
            <Text style={styles.doneSubtitle}>
                {syncResult === 'success'
                    ? 'Your sighting has been synced to the PestCast network.'
                    : "No connection. It will sync automatically when you\u2019re back online."}
            </Text>

            <View style={styles.doneActions}>
                <TouchableOpacity style={styles.newBtn} onPress={handleReset}>
                    <Text style={styles.newBtnText}>New Capture</Text>
                </TouchableOpacity>
                {onClose && (
                    <TouchableOpacity style={styles.mapBtn} onPress={onClose}>
                        <Text style={styles.mapBtnText}>Back to Map</Text>
                    </TouchableOpacity>
                )}
            </View>
        </SafeAreaView>
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function deriveSeverity(confidence: number): Severity {
    if (confidence >= 0.9) return 'critical';
    if (confidence >= 0.7) return 'high';
    if (confidence >= 0.4) return 'medium';
    return 'low';
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const styles = StyleSheet.create({
    full: { flex: 1, backgroundColor: '#000' },

    centerDark: {
        flex: 1,
        backgroundColor: '#0f172a',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
    },
    containerDark: {
        flex: 1,
        backgroundColor: '#0f172a',
    },
    statusText: {
        color: '#94a3b8',
        marginTop: 16,
        fontSize: 16,
    },

    // Back button
    backBtn: {
        position: 'absolute',
        top: 56,
        left: 16,
        zIndex: 10,
        backgroundColor: 'rgba(0,0,0,0.6)',
        paddingHorizontal: 14,
        paddingVertical: 8,
        borderRadius: 10,
    },
    backBtnText: { color: '#fff', fontSize: 15, fontWeight: '600' },

    // Model loading banner
    modelBanner: {
        position: 'absolute',
        top: 60,
        left: 24,
        right: 24,
        backgroundColor: 'rgba(0,0,0,0.7)',
        borderRadius: 12,
        padding: 12,
        alignItems: 'center',
    },
    modelBannerText: { color: '#fbbf24', fontSize: 14 },

    // Results
    resultScroll: { padding: 24, paddingTop: 12 },
    previewImage: {
        width: '100%',
        height: 240,
        borderRadius: 16,
        marginBottom: 20,
    },
    heading: {
        color: '#f1f5f9',
        fontSize: 22,
        fontWeight: '700',
        marginBottom: 16,
    },
    resultCard: {
        backgroundColor: '#1e293b',
        borderRadius: 14,
        padding: 16,
        marginBottom: 12,
    },
    resultCardTop: {
        borderColor: '#22c55e',
        borderWidth: 2,
    },
    resultRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    resultLabel: { color: '#e2e8f0', fontSize: 16, fontWeight: '600' },
    resultConf: { color: '#22c55e', fontSize: 16, fontWeight: '700' },
    confBar: {
        height: 6,
        borderRadius: 3,
        backgroundColor: '#334155',
        overflow: 'hidden',
    },
    confFill: {
        height: 6,
        borderRadius: 3,
        backgroundColor: '#22c55e',
    },
    hint: {
        color: '#64748b',
        textAlign: 'center',
        marginTop: 16,
        fontSize: 13,
    },

    // Done
    doneIcon: { fontSize: 56, marginBottom: 16 },
    doneTitle: {
        color: '#f1f5f9',
        fontSize: 24,
        fontWeight: '700',
        marginBottom: 8,
    },
    doneSubtitle: {
        color: '#94a3b8',
        fontSize: 15,
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 32,
    },
    doneActions: {
        flexDirection: 'row',
        gap: 12,
    },
    newBtn: {
        backgroundColor: '#22c55e',
        paddingHorizontal: 28,
        paddingVertical: 14,
        borderRadius: 14,
    },
    newBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
    mapBtn: {
        backgroundColor: '#334155',
        paddingHorizontal: 28,
        paddingVertical: 14,
        borderRadius: 14,
    },
    mapBtnText: { color: '#e2e8f0', fontSize: 16, fontWeight: '600' },
});
