/**
 * @johnhenry/backend-webgpu — WebGPU implementation of the
 * @johnhenry/tensor-backend contract (browsers via navigator.gpu, Node/Bun
 * via the `webgpu` package's Dawn build, Deno natively).
 */
import { WebGpuBackend, type WebGpuBackendOptions } from "./backend.ts";
import { requestAdapter, requestDevice, summarizeAdapter } from "./device.ts";

export { WebGpuBackend, WebGpuTensor, type WebGpuBackendOptions } from "./backend.ts";
export { getGpu, requestAdapter, type AdapterSummary } from "./device.ts";
export { GEMM_DEFAULT, type GemmConfig } from "./kernels.ts";

export interface CreateWebGpuBackendOptions extends WebGpuBackendOptions {
  /** Use this device instead of requesting one (the backend won't destroy it). */
  device?: GPUDevice;
  /** Store/compute f16 natively when the adapter has `shader-f16` (default true). */
  preferF16?: boolean;
  powerPreference?: GPUPowerPreference;
  /** Also request "timestamp-query" (when available) so `rt.startProfiling()` works. */
  profiling?: boolean;
}

/** Creates a WebGPU backend; rejects when no adapter is available. */
export async function createWebGpuBackend(opts: CreateWebGpuBackendOptions = {}): Promise<WebGpuBackend> {
  const preferF16 = opts.preferF16 ?? true;
  if (opts.device) {
    const device = opts.device;
    const f16 = preferF16 && device.features.has("shader-f16");
    return new WebGpuBackend(device, summarizeAdapter(null, device), { ...opts, f16, ownsDevice: false });
  }
  const adapter = await requestAdapter(opts.powerPreference);
  if (!adapter) throw new Error("webgpu: no GPU adapter available");
  const device = await requestDevice(adapter, preferF16, opts.profiling ? ["timestamp-query"] : []);
  const f16 = device.features.has("shader-f16");
  return new WebGpuBackend(device, summarizeAdapter(adapter, device), { ...opts, f16, ownsDevice: true });
}

/** True when a WebGPU adapter can be obtained in this runtime (for skipping tests). */
export async function isWebGpuAvailable(): Promise<boolean> {
  return (await requestAdapter()) !== null;
}
