/**
 * @johnhenry/backend-webgpu — WebGPU implementation of the
 * @johnhenry/tensor-backend contract (browsers via navigator.gpu, Node/Bun
 * via the `webgpu` package's Dawn build, Deno natively).
 */
import { WebGpuBackend, type WebGpuBackendOptions } from "./backend.ts";
import { requestAdapter, requestDevice, summarizeAdapter } from "./device.ts";

export { WebGpuBackend, WebGpuTensor, type GemmChoice, type WebGpuBackendOptions } from "./backend.ts";
export {
  Runtime,
  Storage,
  type BindingSpec,
  type CompiledKernel,
  type KernelSource,
  type ParamSpec,
  type ParamType,
  type RuntimeStats,
} from "./runtime.ts";
export { getGpu, requestAdapter, type AdapterSummary } from "./device.ts";
export { GEMM_DEFAULT, GEMM_V020, QUANT_GEMM_DEFAULT, QUANT_GEMM_NAVIGATOR, type GemmConfig, type QmvGemmConfig, type QuantGemmConfig, type SgGemmConfig, type SkinnyGemmConfig } from "./kernels.ts";

export interface CreateWebGpuBackendOptions extends WebGpuBackendOptions {
  /** Use this device instead of requesting one (the backend won't destroy it). */
  device?: GPUDevice;
  /**
   * With `device`: the adapter it was requested from. Subgroup-matrix
   * detection needs the adapter's `subgroupMatrixConfigs`, which Dawn does
   * not mirror onto `device.adapterInfo`; without it a device you pass in
   * never uses subgroup matrices.
   */
  adapter?: GPUAdapter;
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

/** The subgroup size when the device has `subgroups` and the adapter reports one fixed size, else 0. */
function fixedSubgroupSize(info: unknown, device: GPUDevice): number {
  if (!device.features.has("subgroups" as GPUFeatureName)) return 0;
  const i = info as { subgroupMinSize?: number; subgroupMaxSize?: number } | undefined;
  return i?.subgroupMinSize && i.subgroupMinSize === i.subgroupMaxSize ? i.subgroupMinSize : 0;
}

/** Creates a WebGPU backend; rejects when no adapter is available. */
export async function createWebGpuBackend(opts: CreateWebGpuBackendOptions = {}): Promise<WebGpuBackend> {
  const preferF16 = opts.preferF16 ?? true;
  if (opts.device) {
    const device = opts.device;
    const f16 = preferF16 && device.features.has("shader-f16");
    const info = opts.adapter?.info ?? (device as { adapterInfo?: unknown }).adapterInfo;
    const subgroupMatrix = (opts.subgroupMatrix ?? true) && sgMatrixUsable(info, device);
    return new WebGpuBackend(device, summarizeAdapter(opts.adapter ?? null, device), { ...opts, f16, ownsDevice: false, subgroupMatrix, subgroupSize: fixedSubgroupSize(info, device) });
  }
  const wantSg = opts.subgroupMatrix ?? true;
  const adapter = await requestAdapter(opts.powerPreference, wantSg);
  if (!adapter) throw new Error("webgpu: no GPU adapter available");
  const extra: GPUFeatureName[] = opts.profiling ? ["timestamp-query"] : [];
  if (wantSg) extra.push(SGMAT);
  extra.push("subgroups" as GPUFeatureName);
  const device = await requestDevice(adapter, preferF16, extra);
  const f16 = device.features.has("shader-f16");
  const subgroupMatrix = wantSg && sgMatrixUsable(adapter.info, device);
  return new WebGpuBackend(device, summarizeAdapter(adapter, device), { ...opts, f16, ownsDevice: true, subgroupMatrix, subgroupSize: fixedSubgroupSize(adapter.info, device) });
}

/** True when a WebGPU adapter can be obtained in this runtime (for skipping tests). */
export async function isWebGpuAvailable(): Promise<boolean> {
  return (await requestAdapter(undefined, true)) !== null;
}
