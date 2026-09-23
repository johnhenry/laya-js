/** The published Laya MLX checkpoints (fp16 safetensors on the Hugging Face Hub). */
export interface Checkpoint {
  repo: string;
  label: string;
  encoder: string;
  approxBytes: number;
  blurb: string;
}

export const CHECKPOINTS: readonly Checkpoint[] = [
  {
    repo: "aac6fef/laya-multilingual-mlx",
    label: "Multilingual",
    encoder: "mmBERT-base",
    approxBytes: 644_000_000,
    blurb: "Smallest download; 1,800+ languages.",
  },
  {
    repo: "aac6fef/laya-mlx",
    label: "English",
    encoder: "ModernBERT-large",
    approxBytes: 843_000_000,
    blurb: "English original checkpoint.",
  },
  {
    repo: "aac6fef/laya-typed-decisions-mlx",
    label: "Typed decisions",
    encoder: "ModernBERT-large",
    approxBytes: 843_000_000,
    blurb: "Fine-tuned for typed choice / score / yes-no decisions.",
  },
];

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}
