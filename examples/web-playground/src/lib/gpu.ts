/** WebGPU capability detection with human-readable failure reasons. */
export interface GpuCapability {
  ok: boolean;
  /** `shader-f16` available (true f16 storage + math). */
  f16: boolean;
  reason?: string;
  adapter?: string;
}

export async function detectWebGpu(): Promise<GpuCapability> {
  if (!globalThis.isSecureContext) {
    return { ok: false, f16: false, reason: "WebGPU needs a secure context: open this page over https:// or http://localhost." };
  }
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) {
    return {
      ok: false,
      f16: false,
      reason:
        "This browser does not expose WebGPU (navigator.gpu). Use Chrome/Edge 113+ (f16 needs 120+), Safari 26+, or Firefox 141+ with WebGPU enabled.",
    };
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (e) {
    return { ok: false, f16: false, reason: `requestAdapter() failed: ${(e as Error).message}` };
  }
  if (!adapter) {
    return { ok: false, f16: false, reason: "WebGPU is present but no GPU adapter is available (blocklisted GPU, disabled hardware acceleration, or a headless browser without a GPU)." };
  }
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const name = info ? [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") : "";
  return { ok: true, f16: adapter.features.has("shader-f16"), adapter: name || "WebGPU adapter" };
}
