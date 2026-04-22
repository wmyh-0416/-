# SFT-CoT Train Log

- Started: `2026-04-22T13:53:31.458221`
- Finished: `2026-04-22T14:04:56.105212`
- Output dir: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_cot_qwen25_3b`
- Model source: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`
- Precision: `bf16`
- Using 4-bit: `True`
- Train samples: `367`
- Val samples: `52`
- Resume checkpoint: `None`
- Best checkpoint: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_cot_qwen25_3b/checkpoint-80`
- Best metric: `0.696850061416626`
- Global step: `92`
- Trainable params: `restored_from_saved_trainer_state`

## Sample Checks

- id: `simple-0408`
  roles: `['user', 'assistant']`
  user: `What is the primary function of a ship's propeller?`
  assistant: `<think>A ship’s propeller is the part that converts the engine’s rotational power into thrust in the water. When the propeller blades spin, they push water backward. By Newton’s...`
- id: `complex-0232`
  roles: `['user', 'assistant']`
  user: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  assistant: `<think>Apply the rewrite rules repeatedly to the leftmost reducible adjacent pair. Starting program: B# #A A# #B #A B# #B A# #A A# B# #A #A A# #B A# #B #B B# #A 1) B# #A → #A B#...`
- id: `complex-0045`
  roles: `['user', 'assistant']`
  user: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  assistant: `<think>Apply the rewrite rules repeatedly to adjacent facing pairs. Starting program: #A #A B# #A B# #A #B B# A# #A #B #A #A #A #A #A #B #B #A B# After reducing all possible fac...`

## Final Metrics

- best_global_step: `80`
- best_metric: `0.696850061416626`
- epoch: `2.0`
- eval_entropy: `0.6420983007320991`
- eval_loss: `0.696850061416626`
- eval_mean_token_accuracy: `0.8113589951625237`
- eval_num_tokens: `159888.0`
- eval_runtime: `10.2387`
- eval_samples_per_second: `5.079`
- eval_steps_per_second: `5.079`
- step: `92`
- total_flos: `2690634271555584.0`
- train_loss: `0.7777048725148906`
- train_runtime: `670.1961`
- train_samples_per_second: `1.095`
- train_steps_per_second: `0.137`

## Validation Preview

- id: `complex-0054`
  gold: `B → A → B → A → B → A → B → A → B → A → C`
  pred: `A → B → A → B → A → B → A → B → A → C`
- id: `complex-0472`
  gold: `C → A → A`
  pred: `C → A → C → A`
- id: `simple-0398`
  gold: `450`
  pred: `450 books`
- id: `simple-0418`
  gold: `A constitution is a foundational document that outlines the framework of a government, the powers and limitations of ...`
  pred: `The purpose of a constitution is to provide the foundational legal framework and guiding principles for how a governm...`
- id: `complex-0204`
  gold: `1`
  pred: `1`
