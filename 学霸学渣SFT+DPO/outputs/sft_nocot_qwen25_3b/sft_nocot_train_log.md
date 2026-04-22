# SFT-NOCOT Train Log

- Started: `2026-04-22T15:46:34.758164`
- Finished: `2026-04-22T15:58:39.857380`
- Output dir: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_nocot_qwen25_3b`
- Model source: `/home/ym3447/学霸学渣SFT+DPO/models/Qwen2.5-3B-Instruct`
- Precision: `bf16`
- Using 4-bit: `True`
- Train samples: `367`
- Val samples: `52`
- Resume checkpoint: `None`
- Best checkpoint: `/home/ym3447/学霸学渣SFT+DPO/outputs/sft_nocot_qwen25_3b/checkpoint-80`
- Best metric: `0.5695514678955078`
- Global step: `92`
- Trainable params: `trainable_params=29,933,568 total_params=3,115,872,256 trainable_pct=0.9607`

## Sample Checks

- id: `simple-0408`
  roles: `['user', 'assistant']`
  user: `What is the primary function of a ship's propeller?`
  assistant: `To convert engine power into thrust that moves the ship forward.`
- id: `complex-0232`
  roles: `['user', 'assistant']`
  user: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  assistant: `<answer>#A #B</answer>`
- id: `complex-0045`
  roles: `['user', 'assistant']`
  user: `A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we must rewrite neighbor toke...`
  assistant: `<answer>#A #A #A #A #A #A #A #A #A #A #B #A B#</answer>`

## Final Metrics

- epoch: `2.0`
- eval_entropy: `0.5336259081959724`
- eval_loss: `0.5695514678955078`
- eval_mean_token_accuracy: `0.8509357422590256`
- eval_num_tokens: `80798.0`
- eval_runtime: `9.7542`
- eval_samples_per_second: `5.331`
- eval_steps_per_second: `5.331`
- total_flos: `1359688456126464.0`
- train_loss: `0.6873093286286229`
- train_runtime: `669.3631`
- train_samples_per_second: `1.097`
- train_steps_per_second: `0.137`

## Validation Preview

- id: `complex-0054`
  gold: `B → A → B → A → B → A → B → A → B → A → C`
  pred: `<answer>B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B → B...`
- id: `complex-0472`
  gold: `C → A → A`
  pred: `C → B → C → B`
- id: `simple-0398`
  gold: `450`
  pred: `450 books`
- id: `simple-0418`
  gold: `A constitution is a foundational document that outlines the framework of a government, the powers and limitations of ...`
  pred: `It establishes the framework for government and protects individual rights.`
- id: `complex-0204`
  gold: `1`
  pred: `2`
