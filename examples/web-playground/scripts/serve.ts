/**
 * Tiny static file server for the browser examples (Node or Bun).
 *   node scripts/serve.ts [dir=dist] [--port 5173]
 * Serves on localhost (a secure context, so WebGPU and the Cache API work).
 */
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";

const args = process.argv.slice(2);
const port = Number(args.includes("--port") ? args[args.indexOf("--port") + 1] : process.env.PORT ?? 5173);
const dir = resolve(args.find((a, i) => !a.startsWith("--") && args[i - 1] !== "--port") ?? "dist");

const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8",
};

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  let path = normalize(join(dir, decodeURIComponent(url.pathname)));
  if (path !== dir && !path.startsWith(dir + sep)) {
    res.writeHead(403).end("Forbidden");
    return;
  }
  try {
    if (statSync(path).isDirectory()) path = join(path, "index.html");
    const size = statSync(path).size;
    res.writeHead(200, {
      "Content-Type": TYPES[extname(path)] ?? "application/octet-stream",
      "Content-Length": size,
      "Cache-Control": "no-cache",
    });
    if (req.method === "HEAD") res.end();
    else createReadStream(path).pipe(res);
  } catch {
    res.writeHead(404, { "Content-Type": "text/plain" }).end("Not found");
  }
});
server.listen(port, "localhost", () => console.log(`Serving ${dir} at http://localhost:${port}/`));
