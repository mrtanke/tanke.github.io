---
title: 'A Bridging Model for Parallel Computation'
date: 2025-10-29T12:30:04+00:00
draft: false
description: 'Paper-reading notes: A Bridging Model for Parallel Computation'
ShowWordCount: true
ShowReadingTime: false
---

## Von Neumann model
The **von Neumann model** was an *“efficient bridge between software and hardware”* because:

- **Hardware designers** could build chips to execute it efficiently.
- **Software developers** could write high-level programs that compile into this model.

Thus, the **von Neumann model** is the connecting bridge that enables programs from
the diverse and chaotic world of software to run efficientby on machines from the diverse and chaotic world of hardware.

## Bulk-synchronous parallel (BSP) model

it is a viable candidate for the role of **bridging model**.

Valiant wants parallel simulations to be **almost as fast as ideal ones**.

- The extra cost should be only a **small constant factor**, not growing with processor count.
- He tries to **avoid efficiency loss that scales with log(p)**.
- The model should work well for **any number of processors**, from a few to millions.

---

## Features of BSP model 
A major feature of the **BSP model** is that it provides this option with optimal efficiency (i.e., within constant factors) provided the programmer writes programs with sufficient parallel slackness.

BSP can automatically manage communication and memory efficiently **if** the program exposes

**enough parallel work**. If there’s enough parallelism (many tasks per processor), the model achieves **near-optimal performance** with **little manual tuning**.