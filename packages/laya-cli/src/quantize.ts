/**
 * `laya quantize`: writes a q8/q4 copy of a Laya checkpoint (see
 * @johnhenry/laya's quant.ts for the format) as a complete checkpoint
 * directory that `load()` reads like any other.
 */
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { openSafetensors } from "@johnhenry/math-plus-safetensors";
import { quantizeSafetensors, readCheckpoint, type QuantBits, type QuantizeReport } from "@johnhenry/laya";

export interface QuantizeCheckpointOptions {
  bits: QuantBits;
  /** Output directory (created; an existing model.safetensors there is refused unless `force`). */
  out: string;
  groupSize?: number | "row";
  /** Quantize the token embedding (default true). */
  embeddings?: boolean;
  exclude?: readonly RegExp[];
  q8?: readonly RegExp[];
  refine?: boolean;
  force?: boolean;
  revision?: string;
  token?: string;
  subfolder?: string;
  offline?: boolean;
}

/** Files copied verbatim next to the quantized weights (LICENSE/NOTICE when present). */
const COPY = ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] as const;
const COPY_IF_PRESENT = ["LICENSE", "NOTICE"] as const;

/**
 * Copies contents only: the HF cache stores read-only files, and copyFile
 * (clonefile on macOS) would carry that mode over and break `--force`.
 */
const copy = async (from: string, to: string) => writeFile(to, await readFile(from));

const exists = async (p: string) => {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
};

export interface QuantizeCheckpointResult extends QuantizeReport {
  source: string;
  out: string;
}

/** Quantizes `model` (Hub repo id or local checkpoint directory) into `opts.out`. */
export async function quantizeCheckpoint(model: string, opts: QuantizeCheckpointOptions): Promise<QuantizeCheckpointResult> {
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(model)) throw new Error("laya quantize needs a Hub repo id or a local checkpoint directory, not a URL");
  const ckpt = await readCheckpoint(model, {
    ...(opts.revision !== undefined ? { revision: opts.revision } : {}),
    ...(opts.token !== undefined ? { token: opts.token } : {}),
    ...(opts.subfolder !== undefined ? { subfolder: opts.subfolder } : {}),
    ...(opts.offline !== undefined ? { offline: opts.offline } : {}),
  });
  const src = ckpt.location;
  const out = resolve(opts.out);
  if (out === resolve(src)) throw new Error("laya quantize: --out must differ from the source checkpoint directory");
  if (!opts.force && (await exists(join(out, "model.safetensors")))) throw new Error(`laya quantize: ${join(out, "model.safetensors")} exists (use --force to overwrite)`);

  const file = await openSafetensors(join(src, "model.safetensors"));
  let result: Awaited<ReturnType<typeof quantizeSafetensors>>;
  try {
    result = await quantizeSafetensors(file, {
      bits: opts.bits,
      ...(opts.groupSize !== undefined ? { groupSize: opts.groupSize } : {}),
      ...(opts.embeddings !== undefined ? { embeddings: opts.embeddings } : {}),
      ...(opts.exclude ? { exclude: opts.exclude } : {}),
      ...(opts.q8 ? { q8: opts.q8 } : {}),
      ...(opts.refine !== undefined ? { refine: opts.refine } : {}),
      metadata: { laya_quant_source: model },
    });
  } finally {
    await file.close();
  }
  await mkdir(out, { recursive: true });
  for (const f of COPY) {
    await mkdir(dirname(join(out, f)), { recursive: true });
    await copy(join(src, f), join(out, f));
  }
  for (const f of COPY_IF_PRESENT) if (await exists(join(src, f))) await copy(join(src, f), join(out, f));
  await writeFile(join(out, "model.safetensors"), result.bytes);
  await writeFile(join(out, "README.md"), readme(model, result.report));
  return { ...result.report, source: src, out };
}

const mb = (n: number) => `${(n / 1e6).toFixed(1)} MB`;

function readme(model: string, r: QuantizeReport): string {
  const scheme = r.scheme === "q8" ? "8-bit symmetric" : "4-bit affine";
  const group = r.groupSize === "row" ? "one group per row" : `groups of ${r.groupSize} along the input dimension`;
  return `---
license: apache-2.0
tags:
- laya
- modernbert
- quantized
- laya-js
---

# ${model} — ${r.scheme} (laya-js)

A **${r.scheme}** (${scheme}, ${group}) copy of [\`${model}\`](https://huggingface.co/${model}),
made with \`laya quantize\` from [laya-js](https://github.com/johnhenry/laya-js).
It is derived from the Apache-2.0 Laya checkpoint (see LICENSE and NOTICE,
copied unchanged from the source); the weights were only re-encoded, not retrained.

- \`model.safetensors\`: ${mb(r.bytesOut)} (source ${mb(r.bytesIn)}); ${r.quantized.length} tensors
  quantized, ${r.kept.length} kept in their original dtype. The file is standard safetensors
  with \`__metadata__.laya_quant = "${r.scheme}"\`; see the laya-js format notes
  ([docs/QUANTIZATION.md](https://github.com/johnhenry/laya-js/blob/main/docs/QUANTIZATION.md)).
- Configs and tokenizer are copied unchanged.

## Usage

\`\`\`js
import { load } from "@johnhenry/laya";
const agent = await load("<this directory, Hub repo or base URL>");
\`\`\`

\`@johnhenry/laya\` ≥ 0.2 dequantizes the weights to float16/float32 while
loading, so GPU memory and speed are those of the float checkpoint; only the
download is smaller. The Python \`laya-mlx\` runtime cannot read this file.
`;
}
