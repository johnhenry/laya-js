/** Raw-mode keyboard for the live demo (cli.py `Keyboard`): buffered, non-blocking reads. */
export class Keyboard {
  #buffer = "";
  #active = false;
  readonly #onData = (chunk: Buffer | string) => {
    this.#buffer += chunk.toString();
  };

  constructor() {
    const stdin = process.stdin;
    if (!stdin.isTTY) return;
    stdin.setRawMode(true);
    stdin.on("data", this.#onData);
    stdin.resume();
    this.#active = true;
  }

  /** Everything typed since the last call ("" when nothing). */
  read(): string {
    const out = this.#buffer;
    this.#buffer = "";
    return out;
  }

  close(): void {
    if (!this.#active) return;
    const stdin = process.stdin;
    stdin.off("data", this.#onData);
    stdin.setRawMode(false);
    stdin.pause();
    this.#active = false;
  }
}
