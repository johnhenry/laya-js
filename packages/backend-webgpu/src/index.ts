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
  /**
   * Use subgroup matrices (Metal simdgroup_matrix) for large Linears when the
   * adapter offers `chromium-experimental-subgroup-matrix` with f32 8×8×8
   * and subgroup size 32. In Node/Bun this creates Dawn with
   * `allow_unsafe_apis`; browsers need their own flag (Chrome:
   * --enable-unsafe-webgpu). Default true.
   */
  subgroupMatrix?: boolean;
}

const SGMAT = "chromium-experimental-subgroup-matrix" as GPUFeatureName;

/** f32 8×8×8 subgroup-matrix config and a fixed subgroup size of 32. */
function sgMatrixUsable(info: unknown, device: GPUDevice): boolean {
  if (!device.features.has(SGMAT)) return false;
  const i = info as { subgroupMatrixConfigs?: { componentType: string; resultComponentType: string; M: number; N: number; K: number }[]; subgroupMinSize?: number; subgroupMaxSize?: number } | undefined;
  if (!i || i.subgroupMinSize !== 32 || i.subgroupMaxSize !== 32) return false;
  return (i.subgroupMatrixConfigs ?? []).some((c) => c.componentType === "f32" && c.resultComponentType === "f32" && c.M === 8 && c.N === 8 && c.K === 8);
}

/** Creates a WebGPU backend; rejects when no adapter is available. */
export async function createWebGpuBackend(opts: CreateWebGpuBackendOptions = {}): Promise<WebGpuBackend> {
  const preferF16 = opts.preferF16 ?? true;
  if (opts.device) {
    const device = opts.device;
    const f16 = preferF16 && device.features.has("shader-f16");
    const subgroupMatrix = (opts.subgroupMatrix ?? true) && sgMatrixUsable((device as { adapterInfo?: unknown }).adapterInfo, device);
    return new WebGpuBackend(device, summarizeAdapter(null, device), { ...opts, f16, ownsDevice: false, subgroupMatrix });
  }
  const wantSg = opts.subgroupMatrix ?? true;
  const adapter = await requestAdapter(opts.powerPreference, wantSg);
  if (!adapter) throw new Error("webgpu: no GPU adapter available");
  const extra: GPUFeatureName[] = opts.profiling ? ["timestamp-query"] : [];
  if (wantSg) extra.push(SGMAT);
  const device = await requestDevice(adapter, preferF16, extra);
  const f16 = device.features.has("shader-f16");
  const subgroupMatrix = wantSg && sgMatrixUsable(adapter.info, device);
  return new WebGpuBackend(device, summarizeAdapter(adapter, device), { ...opts, f16, ownsDevice: true, subgroupMatrix });
}

/** True when a WebGPU adapter can be obtained in this runtime (for skipping tests). */
export async function isWebGpuAvailable(): Promise<boolean> {
  return (await requestAdapter(undefined, true)) !== null;
}
