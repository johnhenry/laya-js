/// <reference types="@webgpu/types" />
/** Browser stand-in for dawn-node.ts: browsers use navigator.gpu only. */
export async function loadDawn(_flags: readonly string[]): Promise<GPU | null> {
  return null;
}
