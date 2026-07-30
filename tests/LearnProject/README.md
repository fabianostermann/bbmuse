# Stochastic multi-agent system for test purposes

## Main concept

AgentA looks at the clock and, depending on the phase, has 4 / 2 / 1 / 3 allowed options — then picks uniformly among them. AgentB copies A with an offset (no choice) and separately rolls a 3-sided die. AgentD rolls a 3-sided die when A is even, otherwise has no choice. AgentC flips a coin when a certain sum is even, otherwise is forced.

## What cloning learns

Not "what did AgentA do," but "what die was AgentA rolling." A neural net that perfectly understands AgentA still can't predict its output — it can only predict the distribution. So its loss stops at the die's unpredictability and cannot go lower.

## Why AgentD exists

Look at the picture. AgentC can see AgentB's output. AgentD cannot see AgentB's, and B can't see D's — they're blind to each other, but the reward pays them only when their numbers match.

That's the difference the 2×2 tests:

- Chain (A and C): C watches A and reacts. Train them one at a time and it works, because reacting to something you can see is easy. -> "iterated best response" (IBR) / "Heterogeneous-Agent PPO" (HAPPO) w/o importance correction
- Parallel (B and D): neither can watch the other. They have to agree in advance on a convention, using only the shared reward as feedback. Training one at a time may fail here, because each keeps adjusting to a partner that has already moved on. -> "Independent PPO" (IPPO) if no shared critic else "Multi-Agent PPO" (MAPPO, which builds on CTDE)

## The formal setting

Cooperative, common (team) reward, partial observability per agent → a Dec-POMDP. Heterogeneous agents, no parameter sharing (your action spaces differ per module).

One nuance worth stating explicitly in the paper: standard Dec-POMDP assumes simultaneous actions, but your blackboard executes in topological order, so downstream agents observe upstream actions within the timestep. That's a sequential-move stochastic game, and it's the structural property that makes your chain cell easy and your parallel cell hard. It's also exactly the property HAPPO and MAT construct deliberately — you get it from the architecture.

# Details of diagnostic scenario for bblearn

Clock (exogenous, NOT armed) -> AgentA -> AgentB -> AgentC, plus one reward.
No semantics; every rule is "compute a support, sample uniformly from it",
so every entropy floor is known analytically (and simulation-verified).

## Run

    bblearn arm AgentA AgentB AgentC        # never arm Clock (see Clock.py)
    bblearn listen --quit-after <T>
    bblearn clone AgentA    # then AgentB, AgentC
    bblearn sculpt AgentC
    bblearn apply AgentC --clone <id>       # or --sculpt <id>

There is no time.sleep anywhere — a run should produce thousands of
timesteps per second. Keep episodes long enough that each of the ~12-24
distinct states is visited many times (floor estimation needs group sizes
well above 1; a few thousand timesteps is plenty).

## Expected numbers (the failure ladder)

| check                      | expected                | if wrong, suspect            |
|----------------------------|-------------------------|------------------------------|
| A: per-group floors        | ln4, ln2, 0, ln3        | floor estimator grouping     |
| A: loss__RepA converges to | 0.7945                  | core pack/spec/CE plumbing   |
| B: shift-segment CE        | ~ 0                     | input wiring to the model    |
| B: loss__RepB converges to | 1.0986 (= 0+ln3, joint) | /len(nvec) convention drift  |
| C: loss__RepC converges to | 0.3369 (= 35/72 * ln2)  | multi-input conditioning     |
| D: loss__RepD converges to | 0.5036 (?)              | ?                            |
| all: kl__<rep>             | ~ 0 after convergence   | (whatever the row above says)|
| lesion RepA before B reads | shift segment collapses | info not flowing on the edge |


Read the ladder top-down: the first failing row localizes the bug.
0.3466 instead of 0.3369 for C specifically means the estimator or model
is not conditioning on the full (RepA, RepB) input.

## Sculpting (reward: prefer RepC == 0)

Qualitative signature to look for:
  - in "even" states the policy shifts probability mass toward 0,
  - in "odd" states it resists the reward (deterministic anchor).
The quantitative closed-form check pi*(0) = e^(r/beta)/(e^(r/beta)+1)
holds only with UNNORMALIZED advantages — compute_advantages currently
standardizes returns, so treat that check as future work (see AgentC.py).
The reward is also designed to trigger the constant-reward/std~0 edge case
in compute_advantages if the policy saturates (see prefer_c_zero.py).

## Parallel variant and the 2x2 (sequential vs joint)

AgentD (RepA -> RepD) is a PARALLEL SIBLING of AgentB: both read RepA,
neither observes the other. Adding it changes NO existing floor; its own
cloning floor is (11/24)*ln3 = 0.5035.

Two coordination rewards define the 2x2 (x {sequential, joint} learning):

| cell     | reward file              | structure                | prediction              |
|----------|--------------------------|--------------------------|-------------------------|
| chain    | chain_coordination.py    | C observes A             | sequential ~= joint     |
| parallel | parallel_coordination.py | B, D mutually unobserved | methods may differ      |

Analytic anchors for reading the results:

    chain:    baseline 0.4201 -> C-only 2/3 -> full 1.0   (KL_A = 0.4479)
    parallel: baseline 1/3    ->            -> full 1.0   (KL_B = ln3 = 1.0986,
                                                           KL_D = 0.5035)
    parallel optimum: shared convention f(a); f(1)=1, f(3)=0 forced by the
    odd-a anchor -> exactly 9 minimal-KL conventions. Log the learned f(a)
    per agent to see WHICH convention gets locked, not just the reward.

REWARDS DIR IS GLOBAL: every *.py in rewards/ is loaded and averaged into
one advantage signal during any sculpt run. Keep exactly ONE reward file
active per experiment (move the rest to e.g. rewards/disabled/).
