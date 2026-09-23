/// <reference types="@webgpu/types" />
/**
 * GPU runtime: size-class buffer pool, pipeline cache, uniform arena and
 * dispatch batching (one compute pass per submit; WebGPU orders dispatches
 * inside a pass, so no manual barriers are needed).
 */

// Numeric flags so we don't depend on GPUBufferUsage globals being installed.
export const USAGE_STORAGE = 0x80;
export const USAGE_COPY_SRC = 0x04;
export const USAGE_COPY_DST = 0x08;
export const USAGE_MAP_READ = 0x01;
export const USAGE_UNIFORM = 0x40;
const MAP_READ = 0x01;
const STAGE_COMPUTE = 0x4;

export type ParamType = "u32" | "i32" | "f32" | "vec8";
export type ParamSpec = readonly (readonly [string, ParamType])[];
export interface BindingSpec {
  name: string;
  /** WGSL element type, e.g. "f32", "vec4<f16>". */
  elem: string;
  access: "read" | "read_write";
}

export interface KernelSource {
  key: string;
  bindings: readonly BindingSpec[];
  params: ParamSpec;
  /** WGSL body (entry point `main`); header (enable, struct, bindings) is generated. */
  body: string;
  f16: boolean;
  /** Extra WGSL `enable` extensions (e.g. "chromium_experimental_subgroup_matrix"). */
  enables?: readonly string[];
  /** Extra global directives, e.g. "diagnostic(off, ...)". */
  directives?: readonly string[];
}

export interface CompiledKernel {
  key: string;
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  params: ParamSpec;
  /** Precomputed uniform layout: [name, type, byte offset]. */
  fields: (readonly [string, ParamType, number])[];
  paramBytes: number;
  nBindings: number;
}

/** Size classes: 256 B minimum, then quarter-power-of-two steps (≤25% waste). */
export function sizeClass(bytes: number): number {
  if (bytes <= 256) return 256;
  const p = 2 ** Math.floor(Math.log2(bytes));
  if (p === bytes) return p;
  const step = p / 4;
  return Math.ceil(bytes / step) * step;
}

export class Storage {
  refs = 1;
  readonly buffer: GPUBuffer;
  /** Allocated (size-class) bytes. */
  readonly bytes: number;
  constructor(buffer: GPUBuffer, bytes: number) {
    this.buffer = buffer;
    this.bytes = bytes;
  }
}

interface PooledBuffer {
  buffer: GPUBuffer;
  releasedEpoch: number;
}

export interface RuntimeStats {
  liveBytes: number;
  pooledBytes: number;
  buffersCreated: number;
  dispatches: number;
  submits: number;
  pipelines: number;
}

export class Runtime {
  readonly device: GPUDevice;
  private pool = new Map<number, PooledBuffer[]>();
  private stagingPool = new Map<number, GPUBuffer[]>();
  private pipelines = new Map<string, CompiledKernel>();
  private layouts = new Map<string, GPUBindGroupLayout>();
  private encoder: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private pendingDispatches = 0;
  private epoch = 0;
  private uniformChunks: { buffer: GPUBuffer; data: Uint8Array; view: DataView; used: number }[] = [];
  /** Bind groups by (layout, buffers, uniform chunk); the uniform offset is dynamic. */
  private bindGroups = new Map<string, GPUBindGroup>();
  private bufferIds = new WeakMap<GPUBuffer, number>();
  private nextBufferId = 1;
  private layoutIds = new Map<GPUBindGroupLayout, number>();
  private uniformIdx = 0;
  private deferredDestroy: GPUBuffer[] = [];
  private firstError: string | null = null;
  readonly uniformAlign: number;
  stats: RuntimeStats = { liveBytes: 0, pooledBytes: 0, buffersCreated: 0, dispatches: 0, submits: 0, pipelines: 0 };

  readonly maxBatch: number;
  readonly maxPooledBytes: number;
  /**
   * Sleep (setTimeout) for most of the expected GPU time before awaiting a
   * readback. Dawn-node resolves mapAsync by polling, which keeps a CPU core
   * busy for the whole wait (100% under Bun); on thermally limited machines
   * that power comes out of the GPU's budget.
   */
  sleepWhileWaiting = false;
  /** Dispatches enqueued since the last readback, and the last observed wait (ms) per count. */
  private sinceRead = 0;
  private waitMs = new Map<number, number>();
  /** The in-progress pre-read sleep: concurrent reads wait on it too before polling. */
  private sleeping: Promise<void> | null = null;

  constructor(device: GPUDevice, maxBatch: number, maxPooledBytes: number) {
    this.device = device;
    this.maxBatch = maxBatch;
    this.maxPooledBytes = maxPooledBytes;
    this.uniformAlign = Math.max(256, device.limits.minUniformBufferOffsetAlignment ?? 256);
    device.addEventListener?.("uncapturederror", (ev: Event) => {
      const e = (ev as GPUUncapturedErrorEvent).error;
      this.firstError ??= e?.message ?? String(e);
    });
  }

  /** Throws the first uncaptured device error, if any. */
  checkErrors(): void {
    if (this.firstError) {
      const m = this.firstError;
      this.firstError = null;
      throw new Error(`WebGPU error: ${m}`);
    }
  }

  // ---- buffers -------------------------------------------------------------

  /** Allocates a storage buffer; `forWrite` reports whether queue.writeBuffer is safe without a flush. */
  acquire(bytes: number): { buffer: GPUBuffer; bytes: number; writeHazard: boolean } {
    const cls = sizeClass(Math.max(4, bytes));
    const list = this.pool.get(cls);
    const hit = list?.pop();
    if (hit) {
      this.stats.pooledBytes -= cls;
      this.stats.liveBytes += cls;
      return { buffer: hit.buffer, bytes: cls, writeHazard: hit.releasedEpoch === this.epoch && this.pendingDispatches > 0 };
    }
    const buffer = this.device.createBuffer({ size: cls, usage: USAGE_STORAGE | USAGE_COPY_SRC | USAGE_COPY_DST });
    this.stats.buffersCreated++;
    this.stats.liveBytes += cls;
    return { buffer, bytes: cls, writeHazard: false };
  }

  release(buffer: GPUBuffer, bytes: number): void {
    this.stats.liveBytes -= bytes;
    if (this.stats.pooledBytes + bytes > this.maxPooledBytes) {
      // May still be referenced by the pending encoder: destroy after submit.
      if (this.pendingDispatches > 0) this.deferredDestroy.push(buffer);
      else buffer.destroy();
      return;
    }
    let list = this.pool.get(bytes);
    if (!list) this.pool.set(bytes, (list = []));
    list.push({ buffer, releasedEpoch: this.epoch });
    this.stats.pooledBytes += bytes;
  }

  /** Destroys all pooled (idle) buffers. */
  trim(): void {
    this.flush();
    for (const list of this.pool.values()) for (const b of list) b.buffer.destroy();
    this.pool.clear();
    this.bindGroups.clear();
    this.stats.pooledBytes = 0;
    for (const list of this.stagingPool.values()) for (const b of list) b.destroy();
    this.stagingPool.clear();
  }

  write(buffer: GPUBuffer, hazard: boolean, data: ArrayBufferView): void {
    if (hazard) this.flush();
    let view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (view.byteLength % 4) {
      const padded = new Uint8Array((view.byteLength + 3) & ~3);
      padded.set(view);
      view = padded;
    }
    // Pass the ArrayBuffer with an explicit byte offset: the Bun/Dawn binding
    // ignores a TypedArray view's byteOffset and would upload the wrong bytes.
    this.device.queue.writeBuffer(buffer, 0, view.buffer as ArrayBuffer, view.byteOffset, view.byteLength);
  }

  // ---- pipelines -----------------------------------------------------------

  kernel(k: () => KernelSource, key: string): CompiledKernel {
    const hit = this.pipelines.get(key);
    if (hit) return hit;
    const src = k();
    const { code, paramBytes } = assemble(src);
    const sig = src.bindings.map((b) => (b.access === "read" ? "r" : "w")).join("") + (src.params.length ? "u" : "");
    let layout = this.layouts.get(sig);
    if (!layout) {
      const entries: GPUBindGroupLayoutEntry[] = src.bindings.map((b, i) => ({
        binding: i,
        visibility: STAGE_COMPUTE,
        buffer: { type: b.access === "read" ? "read-only-storage" : "storage" },
      }));
      if (src.params.length) entries.push({ binding: src.bindings.length, visibility: STAGE_COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } });
      layout = this.device.createBindGroupLayout({ entries });
      this.layouts.set(sig, layout);
      this.layoutIds.set(layout, this.layoutIds.size + 1);
    }
    const module = this.device.createShaderModule({ code, label: key });
    const pipeline = this.device.createComputePipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "main" },
      label: key,
    });
    const { offsets } = paramLayout(src.params);
    const fields = orderedParams(src.params).map(([n, t], i) => [n, t, offsets[i]!] as const);
    const compiled: CompiledKernel = { key, pipeline, layout, params: src.params, fields, paramBytes, nBindings: src.bindings.length };
    this.pipelines.set(key, compiled);
    this.stats.pipelines++;
    return compiled;
  }

  // ---- dispatch --------------------------------------------------------------

  private bufferId(b: GPUBuffer): number {
    let id = this.bufferIds.get(b);
    if (id === undefined) this.bufferIds.set(b, (id = this.nextBufferId++));
    return id;
  }

  private uniform(bytes: number): { buffer: GPUBuffer; offset: number; data: DataView } {
    const size = 1 << 16;
    let chunk = this.uniformChunks[this.uniformIdx];
    if (chunk && chunk.used + bytes > size) chunk = this.uniformChunks[++this.uniformIdx];
    if (!chunk) {
      const data = new Uint8Array(size);
      chunk = {
        buffer: this.device.createBuffer({ size, usage: USAGE_UNIFORM | USAGE_COPY_DST }),
        data,
        view: new DataView(data.buffer),
        used: 0,
      };
      this.uniformChunks[this.uniformIdx] = chunk;
    }
    const offset = chunk.used;
    chunk.used = offset + Math.ceil(bytes / this.uniformAlign) * this.uniformAlign;
    return { buffer: chunk.buffer, offset, data: chunk.view };
  }

  dispatch(
    k: CompiledKernel,
    buffers: readonly GPUBuffer[],
    params: Record<string, number | readonly number[]>,
    groups: readonly [number, number?, number?],
  ): void {
    if (buffers.length !== k.nBindings) throw new Error(`dispatch: ${buffers.length} buffers for ${k.nBindings} bindings`);
    let key = String(this.layoutIds.get(k.layout));
    for (const buf of buffers) key += "," + this.bufferId(buf);
    let offsets: number[] | undefined;
    let ubuf: GPUBuffer | undefined;
    if (k.params.length) {
      const u = this.uniform(k.paramBytes);
      packParams(u.data, u.offset, k.fields, params);
      ubuf = u.buffer;
      offsets = [u.offset];
      key += ":" + this.bufferId(u.buffer) + ":" + k.paramBytes;
    }
    let bindGroup = this.bindGroups.get(key);
    if (!bindGroup) {
      const entries: GPUBindGroupEntry[] = buffers.map((buffer, i) => ({ binding: i, resource: { buffer } }));
      if (ubuf) entries.push({ binding: buffers.length, resource: { buffer: ubuf, offset: 0, size: k.paramBytes } });
      bindGroup = this.device.createBindGroup({ layout: k.layout, entries });
      // Keys embed buffer ids that are never reused, so stale entries only cost memory.
      if (this.bindGroups.size >= 4096) this.bindGroups.clear();
      this.bindGroups.set(key, bindGroup);
    }
    const prof = this.profiler;
    if (prof) {
      // Profiling: one pass per dispatch, bracketed by timestamps.
      this.encoder ??= this.device.createCommandEncoder();
      const i = prof.keys.length;
      const pass = this.encoder.beginComputePass({ timestampWrites: { querySet: prof.querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      pass.setPipeline(k.pipeline);
      pass.setBindGroup(0, bindGroup, offsets);
      pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1);
      pass.end();
      prof.keys.push(k.key);
    } else {
      if (!this.pass) {
        this.encoder ??= this.device.createCommandEncoder();
        this.pass = this.encoder.beginComputePass();
      }
      this.pass.setPipeline(k.pipeline);
      this.pass.setBindGroup(0, bindGroup, offsets);
      this.pass.dispatchWorkgroups(groups[0], groups[1] ?? 1, groups[2] ?? 1);
    }
    this.pendingDispatches++;
    this.sinceRead++;
    this.stats.dispatches++;
    if (this.pendingDispatches >= this.maxBatch) this.flush();
  }

  get hasPending(): boolean {
    return this.pendingDispatches > 0;
  }

  /** Ends the current pass and submits it (plus optional extra commands). */
  flush(extra?: (enc: GPUCommandEncoder) => void): void {
    if (!this.pendingDispatches && !extra) return;
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    this.pass?.end();
    const prof = this.profiler;
    if (prof && prof.keys.length) {
      const n = prof.keys.length;
      const staging = this.device.createBuffer({ size: n * 16, usage: USAGE_MAP_READ | USAGE_COPY_DST });
      this.encoder.resolveQuerySet(prof.querySet, 0, 2 * n, prof.resolve, 0);
      this.encoder.copyBufferToBuffer(prof.resolve, 0, staging, 0, n * 16);
      prof.pending.push({ staging, keys: prof.keys });
      prof.keys = [];
    }
    extra?.(this.encoder);
    for (let i = 0; i <= this.uniformIdx && i < this.uniformChunks.length; i++) {
      const c = this.uniformChunks[i]!;
      if (c.used) this.device.queue.writeBuffer(c.buffer, 0, c.data as Uint8Array<ArrayBuffer>, 0, c.used);
      c.used = 0;
    }
    this.uniformIdx = 0;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.pass = null;
    this.pendingDispatches = 0;
    this.epoch++;
    this.stats.submits++;
    for (const b of this.deferredDestroy) b.destroy();
    this.deferredDestroy.length = 0;
  }

  /** Copies `bytes` from `src` at `srcOffset` to host (flushes first). */
  async readBytes(src: GPUBuffer, srcOffset: number, bytes: number): Promise<ArrayBuffer> {
    const size = sizeClass(Math.max(4, (bytes + 3) & ~3));
    const staging =
      this.stagingPool.get(size)?.pop() ?? this.device.createBuffer({ size, usage: USAGE_MAP_READ | USAGE_COPY_DST });
    const copyBytes = (bytes + 3) & ~3;
    const t0 = performance.now();
    const work = this.sinceRead;
    this.sinceRead = 0;
    this.flush((enc) => enc.copyBufferToBuffer(src, srcOffset, staging, 0, copyBytes));
    const est = work ? this.waitMs.get(work) : undefined;
    // Sleep ~80% of the last wait for the same amount of work; poll only for the rest.
    // Self-correcting: an overestimate shrinks by 20% per read.
    if (this.sleepWhileWaiting && est !== undefined && est > 3) {
      const p = new Promise<void>((r) => setTimeout(r, est * 0.8 - (performance.now() - t0)));
      this.sleeping = p;
      await p;
      if (this.sleeping === p) this.sleeping = null;
    } else if (this.sleeping) await this.sleeping;
    await staging.mapAsync(MAP_READ, 0, copyBytes);
    if (work) {
      if (this.waitMs.size > 256) this.waitMs.clear();
      this.waitMs.set(work, performance.now() - t0);
    }
    const out = staging.getMappedRange(0, copyBytes).slice(0, bytes);
    staging.unmap();
    let list = this.stagingPool.get(size);
    if (!list) this.stagingPool.set(size, (list = []));
    list.push(staging);
    this.checkErrors();
    return out;
  }

  // ---- profiling (needs the "timestamp-query" feature) --------------------

  private profiler: {
    querySet: GPUQuerySet;
    resolve: GPUBuffer;
    keys: string[];
    pending: { staging: GPUBuffer; keys: string[] }[];
    totals: Map<string, { ns: number; count: number }>;
  } | null = null;

  /** Starts per-dispatch GPU timing (each dispatch gets its own timestamped pass). */
  startProfiling(): void {
    if (!this.device.features.has("timestamp-query")) throw new Error("webgpu: profiling needs the timestamp-query feature");
    this.flush();
    this.profiler ??= {
      querySet: this.device.createQuerySet({ type: "timestamp", count: 2 * this.maxBatch }),
      resolve: this.device.createBuffer({ size: 16 * this.maxBatch, usage: 0x200 /* QUERY_RESOLVE */ | USAGE_COPY_SRC }),
      keys: [],
      pending: [],
      totals: new Map(),
    };
  }

  /** Stops profiling and returns GPU time per kernel key (sorted, descending). */
  async stopProfiling(): Promise<{ kernel: string; ms: number; count: number }[]> {
    const prof = this.profiler;
    if (!prof) return [];
    this.flush();
    this.profiler = null;
    for (const { staging, keys } of prof.pending) {
      await staging.mapAsync(MAP_READ);
      const ts = new BigUint64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      keys.forEach((key, i) => {
        const t = prof.totals.get(key) ?? { ns: 0, count: 0 };
        t.ns += Number(ts[2 * i + 1]! - ts[2 * i]!);
        t.count++;
        prof.totals.set(key, t);
      });
    }
    prof.querySet.destroy();
    prof.resolve.destroy();
    return [...prof.totals].map(([kernel, t]) => ({ kernel, ms: t.ns / 1e6, count: t.count })).sort((a, b) => b.ms - a.ms);
  }

  async onIdle(): Promise<void> {
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
  }

  destroyAll(): void {
    this.flush();
    this.trim();
    for (const c of this.uniformChunks) c.buffer.destroy();
    this.uniformChunks = [];
    this.pipelines.clear();
  }
}

function paramLayout(params: ParamSpec): { offsets: number[]; bytes: number } {
  const offsets: number[] = [];
  let off = 0;
  // vec8 (array<vec4<u32>, 2>) fields first: 16-byte aligned.
  for (const [, t] of params) if (t === "vec8") (offsets.push(off), (off += 32));
  for (const [, t] of params) if (t !== "vec8") (offsets.push(off), (off += 4));
  return { offsets, bytes: Math.max(16, Math.ceil(off / 16) * 16) };
}

function orderedParams(params: ParamSpec): (readonly [string, ParamType])[] {
  return [...params.filter((p) => p[1] === "vec8"), ...params.filter((p) => p[1] !== "vec8")];
}

function packParams(dv: DataView, base: number, fields: CompiledKernel["fields"], values: Record<string, number | readonly number[]>): void {
  for (const [name, t, fo] of fields) {
    const o = base + fo;
    const v = values[name];
    if (v === undefined) throw new Error(`missing kernel param ${name}`);
    if (t === "vec8") {
      const arr = v as readonly number[];
      for (let j = 0; j < 8; j++) dv.setUint32(o + 4 * j, arr[j] ?? 0, true);
    } else if (t === "f32") dv.setFloat32(o, v as number, true);
    else if (t === "i32") dv.setInt32(o, v as number, true);
    else dv.setUint32(o, v as number, true);
  }
}

function assemble(src: KernelSource): { code: string; paramBytes: number } {
  const lines: string[] = [];
  if (src.f16) lines.push("enable f16;");
  for (const e of src.enables ?? []) lines.push(`enable ${e};`);
  for (const d of src.directives ?? []) lines.push(`${d};`);
  const ordered = orderedParams(src.params);
  const { bytes } = paramLayout(src.params);
  if (ordered.length) {
    lines.push("struct Params {");
    for (const [n, t] of ordered) lines.push(`  ${n}: ${t === "vec8" ? "array<vec4<u32>, 2>" : t},`);
    lines.push("}");
  }
  src.bindings.forEach((b, i) => lines.push(`@group(0) @binding(${i}) var<storage, ${b.access}> ${b.name}: array<${b.elem}>;`));
  if (ordered.length) lines.push(`@group(0) @binding(${src.bindings.length}) var<uniform> P: Params;`);
  lines.push(src.body);
  return { code: lines.join("\n"), paramBytes: bytes };
}
