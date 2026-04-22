# DPO-Adaptive Train Log

- Started: `2026-04-22T16:10:30.371512`
- Finished: `2026-04-22T16:19:35.104054`
- Output dir: `/home/ym3447/学霸学渣SFT+DPO/outputs/dpo_adaptive_qwen25_3b`
- Model source: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`
- Precision: `bf16`
- Using 4-bit: `True`
- Init adapter path: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_cot_qwen25_3b/checkpoint-80`
- Train samples: `367`
- Val samples: `52`
- Resume checkpoint: `None`
- Best checkpoint: `/home/ym3447/学霸学渣SFT+DPO/outputs/dpo_adaptive_qwen25_3b/checkpoint-40`
- Best metric: `0.030634721741080284`
- Global step: `46`
- Policy params: `trainable_params=29,933,568 total_params=3,115,872,256 trainable_pct=0.9607`
- Reference params: `trainable_params=0 total_params=3,115,872,256 trainable_pct=0.0000`

## Sample Checks

- id: `simple-0408`
  difficulty: `simple` subset: `overthinkingbench` type: `objective`
  prompt: `What is the primary function of a ship's propeller?`
  chosen: `To convert engine power into thrust that moves the ship forward.`
  rejected: `<think>A ship’s propeller is the part that converts the engine’s rotational power into thrust in the water. When the propeller blades spin, they push water backward. By Newton’s...`
- id: `complex-0232`
  difficulty: `complex` subset: `underthinkingbench` type: `objective`
  prompt: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  chosen: `<think>Apply the rewrite rules repeatedly to the leftmost reducible adjacent pair. Starting program: B# #A A# #B #A B# #B A# #A A# B# #A #A A# #B A# #B #B B# #A 1) B# #A → #A B#...`
  rejected: `<answer>#A #B</answer>`
- id: `complex-0045`
  difficulty: `complex` subset: `underthinkingbench` type: `objective`
  prompt: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  chosen: `<think>Apply the rewrite rules repeatedly to adjacent facing pairs. Starting program: #A #A B# #A B# #A #B B# A# #A #B #A #A #A #A #A #B #B #A B# After reducing all possible fac...`
  rejected: `<answer>#A #A #A #A #A #A #A #A #A #A #B #A B#</answer>`

## Final Metrics

- epoch: `1.0`
- eval_logits/chosen: `-1.9147331714630127`
- eval_logits/rejected: `-1.7405043840408325`
- eval_logps/chosen: `-62.04738998413086`
- eval_logps/rejected: `-79.75251770019531`
- eval_loss: `0.030634721741080284`
- eval_rewards/accuracies: `1.0`
- eval_rewards/chosen: `1.063378095626831`
- eval_rewards/margins: `3.6731343269348145`
- eval_rewards/rejected: `-2.6097564697265625`
- eval_runtime: `21.498`
- eval_samples_per_second: `2.419`
- eval_steps_per_second: `2.419`
- total_flos: `0.0`
- train_loss: `0.16448333681277608`
- train_runtime: `443.027`
- train_samples_per_second: `0.828`
- train_steps_per_second: `0.104`

## Validation Preview

- id: `complex-0054` difficulty: `complex`
  gold: `B → A → B → A → B → A → B → A → B → A → C`
  pred: `A → B → B → B → B → B → B → B → B`
- id: `complex-0472` difficulty: `complex`
  gold: `C → A → A`
  pred: `C → C → C`
- id: `simple-0398` difficulty: `simple`
  gold: `450`
  pred: `450 books`
- id: `simple-0418` difficulty: `simple`
  gold: `A constitution is a foundational document that outlines the framework of a government, the powers and limitations of ...`
  pred: `In essence, a constitution sets out the rules that govern how a society is organized and operates. It establishes the...`
- id: `complex-0204` difficulty: `complex`
  gold: `1`
  pred: `1`
