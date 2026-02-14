import { useCallback, useMemo } from 'react';
import { useTensorflowModel } from 'react-native-fast-tflite';
import type { ClassificationResult } from '../types';

// ---------------------------------------------------------------------------
// Label map — must match the order of the model's output classes.
// Replace with the real labels from your trained model's `labels.json`.
// ---------------------------------------------------------------------------
const LABELS: string[] = [
    'Fall Armyworm',
    'Aphid',
    'Whitefly',
    'Corn Borer',
    'Stink Bug',
    'Locust',
    'Mealybug',
    'Thrips',
    'Japanese Beetle',
    'Colorado Potato Beetle',
    'Cabbage Looper',
    'Tomato Hornworm',
    'Spider Mite',
    'Flea Beetle',
    'Cutworm',
];

/**
 * Hook that loads a TFLite model from the asset bundle and exposes a
 * `classify` function that runs inference on a captured image.
 *
 * ⚠️  `react-native-fast-tflite` requires a native build — it will not work
 * inside Expo Go.  Use `npx expo prebuild` + `npx expo run:ios`.
 */
export function useClassifier() {
    // ⚠️  Mocking the model to bypass the missing assets/model.tflite crash.
    // const model = useTensorflowModel(require('../../assets/model.tflite'));
    // const isModelReady = model.state === 'loaded';
    const isModelReady = true;

    /**
     * Mock classification that bypasses actual TFLite inference.
     */
    const classify = useCallback(
        async (_imageUri: string): Promise<ClassificationResult[]> => {
            console.log('[useClassifier] Mocking inference for 1.5s...');

            // Simulate processing time
            await new Promise((resolve) => setTimeout(resolve, 1500));

            return [
                { label: 'Fall Armyworm', confidence: 0.98 },
                { label: 'Corn Earworm', confidence: 0.01 },
            ];
        },
        [],
    );

    return { classify, isModelReady };
}
