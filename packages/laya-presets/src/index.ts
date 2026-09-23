/**
 * @johnhenry/laya-presets — ready-made Laya question sets and email cleaning
 * (port of laya-mlx `presets.py` and `email.py`).
 */
export {
  EMAIL_CATEGORIES,
  emailQuestions,
  guardQuestions,
  moderationQuestions,
  routerQuestions,
  triageQuestions,
} from "./presets.ts";
export { cleanEmailBody, emailState, type EmailStateOptions } from "./email.ts";
