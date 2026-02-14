import React, { useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ActivityIndicator,
    SafeAreaView,
} from 'react-native';
import { supabase } from '../services/supabase';

interface SettingsScreenProps {
    onClose: () => void;
}

export default function SettingsScreen({ onClose }: SettingsScreenProps) {
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState<string | null>(null);

    const handleTestConnection = async () => {
        setLoading(true);
        setStatus(null);
        try {
            const { error } = await supabase.from('sightings').insert({
                species: 'TEST_PBT',
                confidence: 0.99,
                latitude: 38.5449,
                longitude: -121.7405,
                severity: 'low',
                created_at: new Date().toISOString(),
                timestamp: new Date().toISOString(),
            });

            if (error) {
                console.error('[Settings] Supabase Test Failed:', error);
                setStatus(`Error: ${error.message} (${error.code || 'No code'})`);
            } else {
                console.log('[Settings] Supabase Test Success!');
                setStatus('Success: Dummy row inserted!');
            }
        } catch (err: any) {
            console.error('[Settings] Unexpected Error:', err);
            setStatus(`Fatal: ${err.message}`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={onClose} style={styles.backButton}>
                    <Text style={styles.backText}>← Back</Text>
                </TouchableOpacity>
                <Text style={styles.title}>Settings & Sync</Text>
            </View>

            <View style={styles.content}>
                <Text style={styles.description}>
                    Verify the connection to Supabase and check if Role-Based Access Control (RLS) is configured correctly.
                </Text>

                <TouchableOpacity
                    style={styles.button}
                    onPress={handleTestConnection}
                    disabled={loading}
                >
                    {loading ? (
                        <ActivityIndicator color="#fff" />
                    ) : (
                        <Text style={styles.buttonText}>Test Supabase Connection</Text>
                    )}
                </TouchableOpacity>

                {status && (
                    <View style={[styles.statusBox, status.startsWith('Error') ? styles.errorBox : styles.successBox]}>
                        <Text style={styles.statusText}>{status}</Text>
                    </View>
                )}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0a0e17',
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 20,
        gap: 16,
    },
    backButton: {
        padding: 8,
    },
    backText: {
        color: '#6366f1',
        fontSize: 16,
        fontWeight: '600',
    },
    title: {
        color: '#fff',
        fontSize: 20,
        fontWeight: 'bold',
    },
    content: {
        padding: 20,
        gap: 20,
    },
    description: {
        color: '#94a3b8',
        fontSize: 14,
        lineHeight: 20,
    },
    button: {
        backgroundColor: '#22c55e',
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
    },
    buttonText: {
        color: '#fff',
        fontSize: 16,
        fontWeight: '600',
    },
    statusBox: {
        padding: 16,
        borderRadius: 12,
        borderWidth: 1,
    },
    successBox: {
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderColor: '#22c55e',
    },
    errorBox: {
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderColor: '#ef4444',
    },
    statusText: {
        color: '#fff',
        fontSize: 13,
        fontFamily: 'Courier',
    },
});
