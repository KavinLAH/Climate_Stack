import 'react-native-gesture-handler';
import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import MapDashboard from './src/screens/MapDashboard';
import PestCaptureScreen from './src/screens/PestCaptureScreen';
import SettingsScreen from './src/screens/SettingsScreen';

export default function App() {
  const [activeScreen, setActiveScreen] = useState<'map' | 'capture' | 'settings'>('map');

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar style="light" />
      {activeScreen === 'capture' && (
        <PestCaptureScreen onClose={() => setActiveScreen('map')} />
      )}
      {activeScreen === 'settings' && (
        <SettingsScreen onClose={() => setActiveScreen('map')} />
      )}
      {activeScreen === 'map' && (
        <>
          <MapDashboard />
          {/* Settings button */}
          <TouchableOpacity
            style={[fabStyles.fab, fabStyles.settingsFab]}
            onPress={() => setActiveScreen('settings')}
            activeOpacity={0.8}
          >
            <Text style={fabStyles.fabIcon}>⚙️</Text>
          </TouchableOpacity>

          {/* Floating camera button */}
          <TouchableOpacity
            style={fabStyles.fab}
            onPress={() => setActiveScreen('capture')}
            activeOpacity={0.8}
          >
            <Text style={fabStyles.fabIcon}>📷</Text>
          </TouchableOpacity>
        </>
      )}
    </GestureHandlerRootView>
  );
}

const fabStyles = StyleSheet.create({
  fab: {
    position: 'absolute',
    bottom: 36,
    right: 20,
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#22c55e',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    zIndex: 10,
  },
  settingsFab: {
    bottom: 112,
    backgroundColor: '#374151',
  },
  fabIcon: { fontSize: 28 },
});
