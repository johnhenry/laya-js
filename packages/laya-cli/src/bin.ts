/** `laya` executable body (see bin/laya.js). */
import { main } from "./cli.ts";

const code = await main(process.argv.slice(2));
// Native backends (Dawn, MLX) may keep handles alive after dispose: exit explicitly.
process.exit(code);
