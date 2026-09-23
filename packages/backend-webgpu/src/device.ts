/// <reference types="@webgpu/types" />
/**
 * Adapter/device acquisition for browsers (navigator.gpu), Deno
 * (navigator.gpu) and Node/Bun (the `webgpu` package: Dawn).
 */

export interface AdapterSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** "navigator.gpu" or "webgpu (Dawn)". */
  source: string;
  features: string[];
  limits: Record<string, number>;
}

let nodeGpu: Promise<GPU | null> | undefined;

/** Returns a `GPU` object for this runtime, or null when WebGPU is unavailable. */
export function getGpu(): Promise<GPU | null> {
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  if (nav?.gpu) return Promise.resolve(nav.gpu);
  return (nodeGpu ??= (async () => {
    try {
      // Variable specifier: keeps browser bundlers from resolving the Node addon.
      const spec = "webgpu";
      const mod = (await import(/* @vite-ignore */ spec)) as { create(o: string[]): GPU; globals: object };
      const g = globalThis as Record<string, unknown>;
      if (g.GPUBufferUsage === undefined) Object.assign(g, mod.globals);
      return mod.create([]);
    } catch {
      return null;
    }
  })());
}

function gpuSource(): string {
  const nav = (globalThis as { navigator?: { gpu?: GPU } }).navigator;
  return nav?.gpu ? "navigator.gpu" : "webgpu (Dawn)";
}

export async function requestAdapter(powerPreference: GPUPowerPreference = "high-performance"): Promise<GPUAdapter | null> {
  const gpu = await getGpu();
  if (!gpu) return null;
  try {
    return await gpu.requestAdapter({ powerPreference });
  } catch {
    return null;
  }
}

const WANTED_LIMITS = [
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBuffersPerShaderStage",
] as const;

export async function requestDevice(adapter: GPUAdapter, preferF16: boolean, extra: GPUFeatureName[] = []): Promise<GPUDevice> {
  const requiredFeatures: GPUFeatureName[] = extra.filter((f) => adapter.features.has(f));
  if (preferF16 && adapter.features.has("shader-f16")) requiredFeatures.push("shader-f16");
  const requiredLimits: Record<string, number> = {};
  for (const k of WANTED_LIMITS) {
    const v = (adapter.limits as unknown as Record<string, number>)[k];
    if (typeof v === "number") requiredLimits[k] = v;
  }
  return adapter.requestDevice({ requiredFeatures, requiredLimits });
}

export function summarizeAdapter(adapter: GPUAdapter | null, device: GPUDevice): AdapterSummary {
  const info = (adapter?.info ?? (device as { adapterInfo?: GPUAdapterInfo }).adapterInfo) as GPUAdapterInfo | undefined;
  const limits: Record<string, number> = {};
  for (const k of WANTED_LIMITS) {
    const v = (device.limits as unknown as Record<string, number>)[k];
    if (typeof v === "number") limits[k] = v;
  }
  return {
    vendor: info?.vendor ?? "",
    architecture: info?.architecture ?? "",
    device: info?.device ?? "",
    description: info?.description ?? "",
    source: gpuSource(),
    features: [...device.features].map(String).sort(),
    limits,
  };
}
