import React, { useRef, useState } from 'react';
import {
    StyleSheet,
    TouchableOpacity,
    View,
    Text,
    ActivityIndicator,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as FileSystem from 'expo-file-system/legacy';

interface Props {
    /** Called with the local‑cache URI of the captured photo. */
    onCapture: (uri: string) => void;
}

/**
 * Full-screen camera viewfinder with a shutter button.
 * Captures a JPEG, copies it to the local cache directory, and invokes
 * the `onCapture` callback with the cached file URI.
 */
export default function CameraViewfinder({ onCapture }: Props) {
    const cameraRef = useRef<CameraView>(null);
    const [permission, requestPermission] = useCameraPermissions();
    const [capturing, setCapturing] = useState(false);

    // ----- Permission not yet determined -----
    if (!permission) {
        return (
            <View style={styles.center}>
                <ActivityIndicator size="large" color="#22c55e" />
            </View>
        );
    }

    // ----- Permission denied -----
    if (!permission.granted) {
        return (
            <View style={styles.center}>
                <Text style={styles.permText}>Camera access is required to identify pests.</Text>
                <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
                    <Text style={styles.permBtnText}>Grant Permission</Text>
                </TouchableOpacity>
            </View>
        );
    }

    // ----- Capture handler -----
    const handleCapture = async () => {
        if (!cameraRef.current || capturing) return;
        setCapturing(true);

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.85,
                skipProcessing: false,
            });

            if (!photo?.uri) {
                console.warn('[CameraViewfinder] No URI returned from takePictureAsync');
                setCapturing(false);
                return;
            }

            // Move to a predictable cache directory so it persists across renders.
            const cacheDir = `${FileSystem.cacheDirectory}pestcast/`;
            await FileSystem.makeDirectoryAsync(cacheDir, { intermediates: true });

            const filename = `capture_${Date.now()}.jpg`;
            const cachedUri = `${cacheDir}${filename}`;
            await FileSystem.moveAsync({ from: photo.uri, to: cachedUri });

            console.log('[CameraViewfinder] Photo cached at', cachedUri);
            onCapture(cachedUri);
        } catch (err) {
            console.error('[CameraViewfinder] Capture error:', err);
        } finally {
            setCapturing(false);
        }
    };

    // ----- Render -----
    return (
        <View style={styles.container}>
            <CameraView ref={cameraRef} style={styles.camera} facing="back">
                {/* Crosshair overlay */}
                <View style={styles.overlay}>
                    <View style={styles.crosshair} />
                </View>

                {/* Shutter button */}
                <View style={styles.shutterRow}>
                    <TouchableOpacity
                        style={[styles.shutterBtn, capturing && styles.shutterBtnDisabled]}
                        onPress={handleCapture}
                        disabled={capturing}
                        activeOpacity={0.7}
                    >
                        {capturing ? (
                            <ActivityIndicator color="#fff" />
                        ) : (
                            <View style={styles.shutterInner} />
                        )}
                    </TouchableOpacity>
                </View>
            </CameraView>
        </View>
    );
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#000' },
    camera: { flex: 1 },

    // Centered permission prompt
    center: {
        flex: 1,
        backgroundColor: '#0f172a',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 32,
    },
    permText: { color: '#e2e8f0', fontSize: 16, textAlign: 'center', marginBottom: 20 },
    permBtn: {
        backgroundColor: '#22c55e',
        paddingHorizontal: 24,
        paddingVertical: 12,
        borderRadius: 12,
    },
    permBtnText: { color: '#fff', fontWeight: '700', fontSize: 16 },

    // Crosshair overlay
    overlay: {
        ...StyleSheet.absoluteFillObject,
        alignItems: 'center',
        justifyContent: 'center',
    },
    crosshair: {
        width: 200,
        height: 200,
        borderWidth: 2,
        borderColor: 'rgba(34,197,94,0.6)',
        borderRadius: 16,
    },

    // Shutter button
    shutterRow: {
        position: 'absolute',
        bottom: 48,
        left: 0,
        right: 0,
        alignItems: 'center',
    },
    shutterBtn: {
        width: 76,
        height: 76,
        borderRadius: 38,
        borderWidth: 4,
        borderColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    shutterBtnDisabled: { opacity: 0.5 },
    shutterInner: {
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: '#fff',
    },
});
