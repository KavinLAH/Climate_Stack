/**
 * Minimal type declarations for react-native-fast-tflite.
 *
 * This package ships native code without bundled .d.ts files.
 * These types cover the subset of the API used in PestCast.
 */
declare module 'react-native-fast-tflite' {
    export interface TensorflowModel {
        /** Run inference with a single input tensor and return a single output. */
        runForSingleOutput(input: ArrayBuffer[]): Float32Array;
    }

    export interface UseTensorflowModelResult {
        state: 'loading' | 'loaded' | 'error';
        model: TensorflowModel | undefined;
    }

    /**
     * React hook that loads a TFLite model from a bundled asset.
     * @param source - `require('path/to/model.tflite')`
     */
    export function useTensorflowModel(source: number): UseTensorflowModelResult;
}
