---
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'
date: 2025-11-04T12:06:46+00:00
draft: false
description: 'Paper-reading notes: DeepSeek-R1 - Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'
ShowWordCount: true
ShowReadingTime: false
tag: "Notes"
---

Reinforcement Learning

# Introduction

Recent LLMs are rapidly advancing toward AGI. **Post-training** has emerged as an important component of the full training pipeline, which enhances accuracy on reasoning tasks, align with social values, and adapt to user preferences, all while requiring relatively minimal computational resources against **pre-training**. In the context of reasoning capabilities, OpenAI’s o1 series models were the first to introduce **inference-time scaling** by increasing the length of the CoT reasoning process. This approach has achieved significant improvements in various reasoning tasks, such as mathematics, coding, and scientific reasoning. 

However, the challenge of effective **test-time scaling (efficient reasoning at inference)** remains an open question for the research community. Several prior works have explored various approaches, including process-based reward models, RL, and search algorithms such as MCTS and Beam Search. However, none of these methods has achieved general reasoning performance comparable to OpenAI’s o1 series models.

This paper proposes a **RL-only-based approach DeepSeek-R1-Zero** which directly applied RL to the base model without relying on supervised fine-tuing (SFT) as a preliminary step. The network use **DeepSeek-V3-Base** as the foundation and apply **GRPO** (a reinforcement learning framework). After thousands of RL steps, **DeepSeek-R1-Zero** exhibits super performance on reasoning benchmarks, showing strong reasoning performance, and matching OpenAI o1-0912.

However, **DeepSeek-R1-Zero** suffers from **poor readability** and **language mixing**. To address these issues and further enhance reasoning performance, the DeepSeek introduce
**DeepSeek-R1**, which incorporates a small amount of **cold-start data** and a **multi-stage training
pipeline**. 

<aside>

- **DeepSeek-R1-Zero** → trained **only with RL**, no SFT (“pure RL from base model”).
- **DeepSeek-R1** → adds extra **cold-start SFT** + **synthetic data generation** + **another RL phase**.
</aside>

Training pipeline:

1. Collect thousands of **cold-start data** to **fine-tune** **DeepSeek-V3-Base** model.
2. Perform reasoning-oriented **RL** like DeepSeek-R1-Zero.
3. Near convergence in the RL process, Create new **SFT(supervised fine-tuning) data** through **rejection sampling** on the RL checkpoint, combined with supervised data from DeepSeek-V3 in domains such as writing, factual QA, and self-cognition, and then retrain the DeepSeek-V3-Base model.
4. After fine-tuning with the new data, the checkpoint undergoes an additional RL process, taking into account prompts from all scenarios.

After these steps, the obtained checkpoint referred to as **DeepSeek-R1**, which achieves performance on par with OpenAI-o1-1217.

Finally, they perform **distillation** of DeepSeek-R1 into smaller models (based on **Qwen2.5-32B**). Even after removing RL, these distilled models retain reasoning skills, showing that **large-model reasoning discoveries can be transferred**. Notably, the **14B distilled model** beats all open-source baselines on reasoning benchmarks.

<aside>

## Contributions

**Post-Training: Large-Scale Reinforcement Learning on the Base Model**

- **DeepSeek-R1-Zero**: RL applied directly to base model **without SFT** → achieves self-verification, reflection, long CoT reasoning.
- **DeepSeek-R1**: Pipeline with **2 RL + 2 SFT stages** → improves reasoning, alignment, and non-reasoning abilities.

**Distillation: Smaller Models Can Be Powerful Too**

- Reasoning patterns learned by large models can be **distilled** into smaller ones.
- The open-source **DeepSeek-R1-Distill** family (1.5B – 70B parameters) performs **exceptionally well**, matching or surpassing strong baselines like **OpenAI o1-mini**.
    - **DeepSeek-R1-Distill-Qwen-7B:** 55.5% AIME 2024.
    - **DeepSeek-R1-Distill-Qwen-32B:** 72.6% AIME 2024, 94.3% MATH-500, 57.2% LiveCodeBench.

## **Summary of Evaluation Results**

- **Reasoning Tasks:**
    - DeepSeek-R1 reaches 79.8 % Pass@1 on AIME 2024 (≈ OpenAI o1-1217).
    - On MATH-500 → 97.3 %, Codeforces → 2 029 Elo (> 96 % human).
    - Performs slightly below DeepSeek-V3 in engineering tasks.
- **Knowledge Tasks:**
    
    On MMLU (90.8 %), MMLU-Pro (84 %), GPQA Diamond (71.5 %), DeepSeek-R1 beats DeepSeek-V3 and is close to OpenAI o1-1217.
    
- **Other Abilities:**
    - Excels in writing, summarization, code generation, and instruction following.
    - Achieves 87.6 % length-controlled win-rate (AlpacaEval 2.0) and 92.3 % win-rate (ArenaHard).
    - Especially strong on **long-context understanding** and **non-exam queries**.
</aside>

# Approach

Previous work has heavily relied on large amounts of supervised data to enhance model
performance. The DeepSeek team demonstrate that reasoning capabilities can be significantly
improved through **large-scale reinforcement learning (RL)**, even without using supervised
fine-tuning (SFT) as a cold start. Furthermore, performance can be further enhanced with
the inclusion of a small amount of cold-start data. The following sections introduce: (1)
DeepSeek-R1-Zero, which applies RL directly to the base model without any SFT data, and
(2) DeepSeek-R1, which applies RL starting from a checkpoint fine-tuned with thousands of
long Chain-of-Thought (CoT) examples. 3) Distill the reasoning capability from DeepSeek-R1 to
small dense models.

## DeepSeek-R1-Zero

<aside>

DeepSeek-R1-Zero: **Reinforcement Learning on the Base Model**

</aside>

**Proximal Policy Optimization (PPO)** is the standard RL method used in LLM fine-tuning (e.g., RLHF). It involves three models: a **policy model** that generates responses, a **reward(value) model** that scores them, and a **critic model** that estimates a baseline (the expected reward) to compare with the reward. The policy and critic are trained together, using the difference between the actual reward and the baseline to encourage better-than-average outputs while maintaining training stability. But the **reward model is fixed** which only provides the scores. However, PPO requires a large critic network (often the same size as the policy model), which doubles computational cost.

To address this, **Group Relative Policy Optimization (GRPO)** simplifies the process by removing the critic and computing the baseline directly from the **average reward of a group of sampled responses**, making reinforcement learning more efficient for large LLMs like DeepSeek-R1-Zero.

### **Reinforcement Learning Algorithm**

**Goal:** Train the policy model $\pi_\theta$ to generate above-average answers while keeping training stable and close to a reference model.

**Objective Function:**

<div class="math">
$$
J_{GRPO}(\theta)
= E[{q \sim P(Q), \{o_i\}^G_{i=1} \sim \pi_{\theta_{\text{old}}}(O|q)}] \\
\frac{1}{G} \sum_{i=1}^{G}
\Big(
\min\Big(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,
\text{clip}\Big(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\Big)A_i
\Big) - \beta D_{KL}(\pi_\theta||\pi_{\text{ref}})
\Big)
$$
<div>

- $q∼P(Q)$: sample a **question/prompt** from the dataset.
- $ \{o_i\}^G_{i=1} \sim \pi_{\theta_{\text{old}}}(O|q)$: use the old model to generate **G different answers** to the same question.
- $\frac{1}{G} \sum_{i=1}^{G}$: averages the result over the G samples (the group).
- $\frac{\pi_\theta}{\pi_{\theta_{old}}}$: probability ratio showing how the model’s belief changes.
- $A_i$: **advantage**, computed within the group, measures how much better each answer is than group average ($baseline=mean(r_1​,r_2​,…,r_G​)$):

$$
A_i = \frac{r_i - {mean}(r_1,\dots,r_G)}{{std}(r_1,\dots,r_G)}
$$

- $r_i$: reward for response $o_i$.
- **clip(·)**: limits updates to ensure stability (same as PPO).
- $D_{KL}(\pi_\theta||\pi_{ref})$: KL penalty keeping the model close to a reference policy (e.g., base model).

$$
D_{KL}(\pi_\theta || \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} -\log\frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

- $\varepsilon, \beta$: hyperparameters controlling clipping and KL weight.

Where do the **rewards $r_i$** come from? → From **Reward Modeling**, which defines how to score each generated answer. 

$$
\boxed{\text{Total computations} \propto O \times G}
$$

- $O$ = number of **prompts/questions** sampled in a batch
- $G$ = number of **responses per prompt**

### Reward Modeling

To train DeepSeek-R1-Zero, a **rule-based reward system** was adopted that mainly consists of two types of rewards:

$$
r_i=r_i(accuracy)+r_i(format)
$$

- **Accuracy rewards**: The accuracy reward model evaluates whether the response is correct.
For example, in the case of math problems with deterministic results, the model is required
to provide the final answer in a specified format (e.g., within a box), enabling reliable
rule-based verification of correctness. If the answer matches the correct one, reward = 1; else 0 (or scaled).
- **Format rewards**: In addition to the accuracy reward model, we employ a format reward
model that enforces the model to put its thinking process between ‘<think>’ and ‘</think>’
tags. Reward = 1 if the output format is correct, else 0.

### Training Template

To train DeepSeek-R1-Zero, begining by designing a straightforward template that guides
the base model to follow to the specified instructions. This following template requires DeepSeek-R1-Zero to **first produce a reasoning process, followed by the final answer. prompt** will be replaced with the specific reasoning question during training.
Limit the constraints to this structural format, avoiding any content-specific biases to ensure that the model’s natural progression can be observed accurately during the RL process.

![image.png](43d9d8bb-ab88-4db1-98e7-84f74c71f242.png)

Although DeepSeek-R1-Zero exhibits strong reasoning capabilities and autonomously develops unexpected and powerful reasoning behaviors, it faces several issues. For instance, DeepSeek R1-Zero struggles with challenges like **poor readability**, and **language mixing**(the base model DeepSeek-V3 is multilingual, and RL doesn’t penalize mixing languages). To make reasoning processes more readable and share them with the open community, we explore **DeepSeek-R1**, a method that utilizes RL with human-friendly cold-start data.

> **Good readability (_after_SFT):**
> 

<aside>

<think>
To compute the sum of the first 5 even numbers:
2 + 4 + 6 + 8 + 10 = 30.
</think>
<answer> 30 </answer>

</aside>

> **Poor readability (DeepSeek-R1-Zero):**
> 

<aside>

<think> sum=2+4+6+8+10=>=30?? yes right 30 correct result final output=30 </think>
<answer>30</answer>

</aside>

## DeepSeek-R1

<aside>

DeepSeek-R1: **Reinforcement Learning with Cold Start**

</aside>

Inspired by the promising results of DeepSeek-R1-Zero, two natural questions arise: 1) Can reasoning performance be further improved or convergence accelerated by incorporating **a small amount of high-quality data as a cold start**? 2) How can we train a user-friendly model that not only produces clear and coherent Chains of Thought (CoT) but also demonstrates **strong general capabilities**? To address these questions, we design a pipeline to train DeepSeek-R1. The pipeline consists of four stages, outlined as follows.

### Cold Start

Construct and collect a small amount of **long CoT data** to fine-tune the model as the initial RL actor. The data can include reasoning examples in the **prompt** (few-shot) or detailed reasoning in the **response**, ensuring the model learns to generate readable, step-by-step solutions. To collect such data, there are several approaches: using few-shot prompting with a long CoT as an example, directly prompting models to generate detailed answers with reflection and verification, gathering DeepSeek-R1-Zero outputs in a readable format, and refining the results through post-processing by human annotators. 
In this work, they collect thousands of **cold-start data** to fine-tune the DeepSeek-V3-Base as
the starting point for RL. Compared to DeepSeek-R1-Zero, the advantages of cold start data are readability of responses and better performance by **designing a readable pattern** that includes a summary at the end of each response.

### Reasoning-oriented Reinforcement Learning

After fine-tuning DeepSeek-V3-Base on cold-start data, the same GRPO-based RL process as DeepSeek-R1-Zero is applied to **enhancing the model’s reasoning capabilities** in math, logic, science, and coding tasks. However, RL caused language mixing in Chain-of-Thought reasoning, so they introduced a **language-consistency reward**, computed as the ratio of target-language words in the reasoning text. The final reward combines **accuracy** and **language consistency**: $r_{final} = r_{accuracy} + r_{lang}$Although this slightly reduces task performance, it produces more readable, human-preferred outputs. The model is trained until convergence, resulting in the final **DeepSeek-R1** model.

### Rejection Sampling and Supervised Fine-Tuning

After reasoning-oriented RL training is finished, they use the **RL-trained checkpoint** to **generate new supervised fine-tuning (SFT) data** for the next round, aiming to improve model’s capabilities in writing, role-playing, and other general-purpose tasks. They produce **two types of data**: For reasoning data, they **sample many responses** (like 10–20 per question) from the **RL checkpoint**. then they **perform rejection sampling**. For each prompt, they sample multiple responses and **retain only the correct ones**. For non-reasoning data, they add general-purpose data reuse or regenerate these using **DeepSeek-V3’s SFT data**.

### Reinforcement Learning for all Scenarios

To further align the model with human preferences, they implement a secondary reinforcement
learning stage aimed at improving the model’s **helpfulness and harmlessness** while simultaneously refining its reasoning capabilities. 

For reasoning data, they follow to the methodology outlined in DeepSeek-R1-Zero, which utilizes **rule-based rewards** to guide the learning process. For general data, they resort to reward models to capture **human preferences $r_i^{(preference)}∈[0,1]$**. They build upon the DeepSeek-V3 pipeline and adopt a similar distribution of preference pairs and training prompts. **For helpfulness**, they focus exclusively on the final summary, ensuring that the assessment emphasizes the utility and relevance of the response to the user while minimizing interference with the underlying reasoning process. **For harmlessness**, they evaluate the entire response of the model, including both the reasoning process and the summary, to identify and mitigate any potential risks, biases, or harmful content that may arise during the generation process. Ultimately, the integration of reward signals and diverse data distributions enables to train a model that excels in **reasoning while prioritizing helpfulness and harmlessness.**

## Distillation: Empower Small Models with Reasoning Capability

For distilled models, the team apply **only SFT** and do not include an RL stage, even though
incorporating RL could substantially boost model performance.

# Conclusion

This paper shares how to enhancing model reasoning abilities through reinforcement learning. DeepSeek-R1-Zero represents a pure RL approach without relying on cold-start data, achieving strong performance across various tasks. DeepSeek-R1 is more powerful, leveraging cold-start data alongside iterative RL fine-tuning. Ultimately, DeepSeek-R1 achievesperformance comparable to OpenAI-o1-1217 on a range of tasks.
They further explore distillation the reasoning capability to small dense models. They use DeepSeek-R1 as the teacher model to generate 800K training samples, and fine-tune several small dense models. The results are promising: DeepSeek-R1-Distill-Qwen-1.5B outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks with 28.9% on AIME and 83.9% on MATH. Other dense models also achieve impressive results, significantly outperforming other instruction-tuned models based on the same underlying checkpoints.

In the future, there are some research directions across the following directions for DeepSeek-R1.

- **General Capability**: Currently, the capabilities of DeepSeek-R1 fall short of DeepSeek-V3 in tasks such as function calling, multi-turn, complex role-playing, and JSON output. Moving forward, we plan to explore how long CoT can be leveraged to enhance tasks in these fields.
- **Language Mixing**: DeepSeek-R1 is currently optimized for Chinese and English, which may result in language mixing issues when handling queries in other languages. For instance, DeepSeek-R1 might use English for reasoning and responses, even if the query is in a language other than English or Chinese. We aim to address this limitation in future updates.
- **Prompting Engineering**: When evaluating DeepSeek-R1, we observe that it is sensitive to prompts. Few-shot prompting consistently degrades its performance. Therefore, we recommend users directly describe the problem and specify the output format using a zero-shot setting for optimal results.
- **Software Engineering Tasks**: Due to the long evaluation times, which impact the efficiency of the RL process, large-scale RL has not been applied extensively in software engineering tasks. As a result, DeepSeek-R1 has not demonstrated a huge improvement over DeepSeek-V3 on software engineering benchmarks. Future versions will address this by implementing rejection sampling on software engineering data or incorporating asynchronous evaluations during the RL process to improve efficiency.