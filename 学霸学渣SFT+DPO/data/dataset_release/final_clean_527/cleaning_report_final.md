# Cleaning Report Final

## Summary

- Input total: 928
- Final retained total: 527
- Objective retained: 446
- Open-ended retained: 81
- Simple retained: 335
- Complex retained: 192

## Output Counts

- `master_clean_dataset_final.jsonl`: 527
- `sft_cot_clean_final.jsonl`: 527
- `sft_nocot_clean_final.jsonl`: 527
- `dpo_adaptive_clean_final.jsonl`: 527

## Deletion Stats

- `cot_empty`: 0
- `cot_refusal`: 0
- `cot_fake_reasoning`: 95
- `cot_offtopic`: 0
- `cot_wrong_answer`: 213
- `nocot_empty`: 0
- `nocot_refusal`: 0
- `nocot_has_multistep_reasoning`: 12
- `nocot_offtopic`: 8
- `nocot_wrong_answer`: 258
- `pair_incomplete`: 401

## Primary Rejection Reasons

- `cot_fake_reasoning`: 95
- `cot_wrong_answer`: 213
- `nocot_has_multistep_reasoning`: 7
- `nocot_offtopic`: 6
- `nocot_wrong_answer`: 80

## Kept Samples

### simple-0306

- difficulty: `simple`
- question_type: `objective`
- question: What is the minimum number of bits required for an ADC to achieve a resolution of 0.1%?
- gold_answer: 10
- cot_response: <think>For an ADC, resolution is approximately 1 / 2^N. To achieve 0.1% = 0.001, we need 2^N ≥ 1000, so N ≈ 9.97. Round up to 10 bits.</think>
<answer>10 bits</answer>
- nocot_response: 10 bits

### simple-0356

- difficulty: `simple`
- question_type: `objective`
- question: What is the term for a tax on imports?
Options:
A. Embargo
B. Tariff
C. Quota
D. Subsidy
- gold_answer: Tariff
- cot_response: <think>A tax on imports is called a tariff. An embargo is a ban, a quota is a quantity limit, and a subsidy is government financial support.</think>
<answer>B. Tariff</answer>
- nocot_response: Tariff

### complex-0113

- difficulty: `complex`
- question_type: `objective`
- question: Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. If the answer is negative, reply as a negative value (ex., -0x3), not the two's-compliment form. Reply only with the final hexidecimal value.
((((0x5481 + 0x88b2) << 0x3) * ((0x2b5f >> 0x1) - (0x65fc >> 0x3))) >> 0x2)
- gold_answer: 0xf71efa0
- cot_response: <think>Compute step by step:
0x5481 + 0x88b2 = 0xdd33
0xdd33 << 0x3 = 0x6e998

0x2b5f >> 0x1 = 0x15af
0x65fc >> 0x3 = 0x0cbf
0x15af - 0x0cbf = 0x08f0

0x6e998 * 0x08f0 = 0x0f71efa00
0x0f71efa00 >> 0x2 = 0x0f71efa0</think>
<answer>0xf71efa0</answer>
- nocot_response: <answer>0xf71efa0</answer>

### simple-0112

- difficulty: `simple`
- question_type: `open_ended`
- question: What is the purpose of doping in semiconductor materials?
- gold_answer: Doping involves introducing impurities into a semiconductor material to alter its electrical properties. The purpose of doping is to create regions with different electrical properties, such as p-type or n-type regions, which are used to control the flow of electrical current in electronic devices.
- cot_response: <think>Doping is used to deliberately add small amounts of impurities to a pure semiconductor so its electrical conductivity can be controlled. By changing the number and type of charge carriers, doping makes it possible to create n-type or p-type material, which is essential for making devices like diodes, transistors, and integrated circuits.</think>
<answer>The purpose of doping in semiconductor materials is to control and increase their electrical conductivity by adding impurities that change the number of charge carriers.</answer>
- nocot_response: To control electrical conductivity by adding impurities that increase charge carriers.

### simple-0497

- difficulty: `simple`
- question_type: `objective`
- question: If a map scale is 1:1000, how many meters on the ground is represented by 1 cm on the map?
- gold_answer: 10
- cot_response: <think>A scale of 1:1000 means 1 unit on the map equals 1000 of the same units on the ground. So 1 cm on the map equals 1000 cm on the ground, which is 10 meters.</think>
<answer>10 meters</answer>
- nocot_response: 10 meters

## Deleted Samples

### simple-0025

- difficulty: `simple`
- question_type: `objective`
- cot_reason: `cot_wrong_answer`
- nocot_reason: `None`
- question: What is the term for a forest that is managed for timber production?
- gold_answer: Plantation

### complex-0426

- difficulty: `complex`
- question_type: `objective`
- cot_reason: `cot_wrong_answer`
- nocot_reason: `nocot_wrong_answer`
- question: The following question is a propositional logic reasoning question.

In the question we provide a list of premises. The task is to infer a correct conclusion from the premise.

FORMAT INSTRUCTIONS:
- Return the conclusion logic statement, as your final answer.
- Use the following notation to denote symbols
    - OR = ∨
    - AND = ∧
    - IMPLIES = →
    - IFF = ↔
    - NOT = ¬

Here is the question:
Given:
1. P
.2. Q
.3. (P ∨ P)
.What can we conclude from the above statements?
- gold_answer: None

### complex-0319

- difficulty: `complex`
- question_type: `objective`
- cot_reason: `cot_wrong_answer`
- nocot_reason: `None`
- question: I have a Go problem for you. Black moves next - can you capture some of the white stones?

```
   A B C D E F G H I J K L M N O
15 . . . . . . . . . . X . . . .
14 . . . O . . . . . . . . . . .
13 . . . . . . . . . . O . . X .
12 . . . . O . . . . . O . . X X
11 . . . . . . X . . . . . X O O
10 . . . . . . . . O . . . X O .
 9 . . . . . . . . . . . . . X .
 8 . . . . . X X . . . . . . . .
 7 . . . . . . O . . . . . . . .
 6 . . . . . . . . . X . . . . .
 5 . . . . . . X . . . . . . . .
 4 . . . . . . . . . . . . . . .
 3 X O . . . . . . . O . . . . .
 2 . . . . . . . . O . . . . . .
 1 . . . . . . . . . . . . . . .
```

X - Black
O - White

Specify your move in coordinates (e.g. 'C4' for column C, row 4)
- gold_answer: O10

### simple-0018

- difficulty: `simple`
- question_type: `objective`
- cot_reason: `cot_wrong_answer`
- nocot_reason: `nocot_wrong_answer`
- question: A control system is marginally stable if its gain margin is?
- gold_answer: 0

### complex-0381

- difficulty: `complex`
- question_type: `objective`
- cot_reason: `cot_wrong_answer`
- nocot_reason: `nocot_wrong_answer`
- question: Navigate from 'F' (start) to '*' (goal):

```
VVVVVVVVVVVVVVVVVVVV
VJVJJJJJJVJJJJJJJJJV
VJVVJJJJJJJJVJVJVVJV
VJVVJJJJVVJVJJJJJJJV
VJVJJV*VVJVJJJVJJJJV
VVJJJJVVVVJJVJJVVJJV
VJJJJJJVVJJVJVVJJVFV
VJJJVVVVVVVJJJVJJJJV
VJVJVJJJJVJJJJJJVJJV
VJJVJJJJJVVJJJJJJVVV
VJVJVVJVJJJJJJVJJJJV
VJVVJJVJJJVJJVJJJVJV
VJVJVJJJJVJVJVJJJJJV
VJJJJJJJJVJVJVJVJVJV
VJJJJVVVJJJJJJJVJVJV
VJVJVVJJJVJVJJJJVVJV
VJVJJJJJJJJVJJVJJJVV
VJJJVVVJJJJJJJJJJVVV
VJJJJJJVJJJVJJVVVJVV
VVVVVVVVVVVVVVVVVVVV
```
Legend: 'V' = Wall, 'J' = Passage

What is the minimum number of steps to reach the goal?
Give only the number of steps as your final answer, no other text or formatting.
- gold_answer: 20

