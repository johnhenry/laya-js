/**
 * One-key state builders for the single-text question sets. Each uses the
 * field name that the matching `*Questions()` instructions refer to, so callers
 * do not have to hard-code it.
 */

/** State for `guardQuestions()`: the text is the `prompt`. */
export function guardState(text: string): { prompt: string } {
  return { prompt: text };
}

/** State for `moderationQuestions()`: the text is the `post`. */
export function moderationState(text: string): { post: string } {
  return { post: text };
}

/** State for `routerQuestions()`: the text is the `request`. */
export function routerState(text: string): { request: string } {
  return { request: text };
}

/** State for `triageQuestions()`: the text is the `message`. */
export function triageState(text: string): { message: string } {
  return { message: text };
}
