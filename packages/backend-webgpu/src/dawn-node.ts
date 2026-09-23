/// <reference types="@webgpu/types" />
/**
 * Node/Bun GPU loader: the `webgpu` package (Dawn). Selected through the
 * package.json `#dawn` import; browser builds get dawn-browser.ts instead,
 * so bundlers never see the `webgpu` specifier.
 */
export async function loadDawn(flags: readonly string[]): Promise<GPU | null> {
  try {
    const mod = (await import("webgpu")) as { create(o: string[]): GPU; globals: object };
    const g = globalThis as Record<string, unknown>;
    if (g.GPUBufferUsage === undefined) Object.assign(g, mod.globals);
    return mod.create([...flags]);
  } catch {
    return null;
  }
}
