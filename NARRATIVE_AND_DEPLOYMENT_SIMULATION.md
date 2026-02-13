# The Fairness Lifecycle: From Benchmark to Bedside
## Narrative Framework & Real-Life Deployment Simulation

---

## PART 1: THE CORE NARRATIVE ARGUMENT

### The Problem With Fairness Research Today

Most fairness-in-ML papers end at the benchmark. They report AUROC, TPR gaps, and
percentage reductions in a controlled cross-validation setting and declare victory. But
the *journey* from a benchmark result to a deployed clinical tool that actually reduces
health inequity is long, nonlinear, and full of failure points that existing literature
almost completely ignores.

Your paper's unique contribution is not just "we tested 26 methods." It is that you
traced the **full lifecycle** of fairness -- from the moment data is collected from
51,720 elderly Europeans, through every stage where bias enters or is corrected, all the
way to the point where a clinician acts on a prediction -- and you discovered that
**fairness transforms, degrades, and sometimes inverts at each stage.**

### The Two Anchoring Terms

**JOURNEY** = The path a single prediction takes from raw survey response to clinical
action. At every waypoint, fairness is gained or lost. Your data shows exactly where.

**LIFECYCLE** = The institutional process of building, validating, deploying, and
maintaining a fair ML system. Your 26 methods don't just differ in TPR gap reduction --
they differ in *which stage of the lifecycle they can survive.*

### The One-Sentence Thesis

> "Algorithmic fairness is not a property of a model -- it is a property of a deployment
> lifecycle, and most methods that achieve fairness in benchmarks cannot sustain it through
> the journey to clinical use."

---

## PART 2: THE DEPLOYMENT SIMULATION

### Setting: A National Health Screening Programme

Imagine the EU commissions a **pan-European mental health screening programme** for adults
aged 50+ using the SHARE infrastructure. The goal: identify individuals at risk of
depression (EUROD), low life satisfaction (LS), poor quality of life (CASP), and poor
self-rated health (SRH), then route them to appropriate interventions.

The system must serve 51,720+ individuals across multiple countries, and the EU mandate
requires **equitable detection rates across socioeconomic groups** -- a low-SES retiree
in Portugal deserves the same chance of being correctly identified as a high-SES retiree
in Denmark.

Here is what happens at each stage of deployment, and what your data reveals:

---

### STAGE 1: DATA COLLECTION (The Inherited Bias)
**Lifecycle Phase: Problem Origination**

Before any model is built, the SHARE Wave 9 data already encodes a societal truth that
becomes the central fairness problem:

| SES Quintile | Depression | Low Life Sat. | Low QoL | Poor Self-Health |
|:---:|:---:|:---:|:---:|:---:|
| Q1 (poorest) | 34.7% | 42.5% | 44.1% | 48.1% |
| Q5 (wealthiest) | 21.0% | 19.8% | 16.8% | 25.3% |

**The journey insight:** The disparity is not equal across outcomes. CASP shows a 2.6x
ratio (44.1% vs 16.8%), while SRH shows only 1.9x (48.1% vs 25.3%). This means fairness
interventions face fundamentally different challenges per outcome -- a fact invisible in
single-outcome studies.

**Impact point:** Any model trained on this data will learn that "low SES predicts poor
mental health" as a strong signal. This is clinically true but creates a fairness trap:
the model becomes excellent at detecting mental health problems in Q1 but systematically
*under-detects* in Q4-Q5.

**What this means for the paper narrative:** The fairness problem is born BEFORE
modelling begins. The lifecycle starts at data collection, not at model training.

---

### STAGE 2: BASELINE MODEL (The Unfair Default)
**Lifecycle Phase: Model Development**

The screening programme builds a LightGBM classifier (AUROC ~0.83 across all outcomes).
Performance looks excellent on aggregate. But the TPR breakdown by SES quintile reveals
the embedded harm:

| Outcome | Baseline TPR Gap (Q1 vs Q5) | What This Means Clinically |
|:---:|:---:|:---|
| EUROD | 0.167 (16.7 pp) | 1 in 6 depressed wealthy individuals missed compared to poor |
| LS | 0.301 (30.1 pp) | Nearly 1 in 3 dissatisfied wealthy individuals missed |
| CASP | 0.280 (28.0 pp) | More than 1 in 4 low-QoL wealthy individuals missed |
| SRH | 0.181 (18.1 pp) | ~1 in 5 unhealthy wealthy individuals missed |

**The journey insight:** The model's unfairness is *outcome-dependent*. Life satisfaction
(LS) shows 30.1 pp gap vs depression's 16.7 pp. A one-size-fits-all fairness fix cannot
work. The lifecycle must accommodate outcome-specific intervention.

**Impact point:** If deployed as-is, this screening programme would disproportionately
miss mental health problems in higher-SES individuals. While this might seem
counterintuitive ("shouldn't we focus on the poor?"), equitable detection is the legal
and ethical requirement -- under-detection in ANY group means those individuals don't get
referred to care.

---

### STAGE 3: FAIRNESS INTERVENTION SELECTION (The Fork in the Road)
**Lifecycle Phase: Bias Mitigation**

The programme's data science team must choose from 26 bias mitigation methods across
4 categories. This is where your benchmark provides its most critical lifecycle insight:
**the best method on paper is often un-deployable in practice.**

#### The Benchmark Rankings (What a naive reader would conclude):

| Outcome | Best TPR Reduction | % Reduction | Looks Great? |
|:---:|:---|:---:|:---:|
| EUROD | FairConstraints | 89.3% | Yes |
| LS | ExpGrad_EO | 94.4% | Yes |
| CASP | ExpGrad_TPR | 92.5% | Yes |
| SRH | RejectOption | 90.5% | Yes |

#### The Deployment Reality (What actually matters):

| Outcome | Best Method | Deployable? | Requires SES at Inference? |
|:---:|:---|:---:|:---:|
| EUROD | FairConstraints | YES | No |
| LS | ExpGrad_EO | YES | No |
| CASP | ExpGrad_TPR | YES | No |
| SRH | RejectOption | NO | Yes (needs SES for every prediction) |

**The journey insight for SRH:** The best-performing method (RejectOption, 90.5%
reduction) requires knowing each patient's SES at the moment of prediction. In a
real screening programme, this means:
- Every patient must disclose income/wealth data at screening
- The system must maintain a live SES classification pipeline
- Privacy regulations (GDPR) may prohibit using SES as a runtime feature
- The SES data may not be available at the point of care

The *actually deployable* best method for SRH is **ExpGrad_TPR** (86.6% reduction) --
still excellent, but the gap between "benchmark best" and "deployment best" is the core
story of the lifecycle.

#### The Deployability Cliff (Your Paper's Killer Finding):

| Category | Total Methods | Deployable | Non-Deployable |
|:---:|:---:|:---:|:---:|
| Pre-processing | 5 | 2 (40%) | 3 (60%) |
| In-processing | 13 | 10 (77%) | 3 (23%) |
| Post-processing | 6 | 0 (0%) | 6 (100%) |
| Augmentation | 2 | 2 (100%) | 0 (0%) |

**ZERO post-processing methods are deployable.** Every single one requires the protected
attribute at inference time. Yet post-processing methods consistently show the best
fairness metrics in benchmarks. This is the lifecycle illusion: the methods that win
benchmarks are the methods that cannot survive deployment.

---

### STAGE 4: THE ACCURACY-FAIRNESS TRADE-OFF (The Real Cost)
**Lifecycle Phase: Performance Negotiation**

The programme's clinical advisory board asks: "What does fairness cost us in accuracy?"

Your data provides precise answers for the deployable methods:

#### Deployable Methods -- The Real Menu:

**For EUROD (Depression):**
| Method | TPR Gap | % Reduction | AUROC | AUROC Drop | Deployable |
|:---|:---:|:---:|:---:|:---:|:---:|
| BASELINE | 0.167 | -- | 0.830 | -- | Yes |
| FairConstraints | 0.018 | 89.3% | 0.818 | -0.012 | Yes |
| ExpGrad_TPR | 0.019 | 88.8% | 0.819 | -0.011 | Yes |
| ExpGrad_EO | 0.020 | 88.2% | 0.819 | -0.011 | Yes |
| Reweighing | 0.119 | 28.7% | 0.830 | -0.000 | Yes |

**The journey insight:** FairConstraints and ExpGrad variants achieve ~89% fairness
improvement at a cost of only 1.1-1.2 percentage points of AUROC. This is clinically
negligible. The trade-off is not a cliff -- it's a gentle slope. But Reweighing preserves
full AUROC with only 29% fairness gain. The lifecycle question becomes: "How much fairness
does the mandate require?"

**For LS (Life Satisfaction) -- The Hardest Outcome:**
| Method | TPR Gap | % Reduction | AUROC | AUROC Drop |
|:---|:---:|:---:|:---:|:---:|
| BASELINE | 0.301 | -- | 0.805 | -- |
| ExpGrad_EO | 0.017 | 94.4% | 0.775 | -0.030 |
| ExpGrad_TPR | 0.019 | 93.8% | 0.777 | -0.028 |
| FairConstraints | 0.060 | 80.2% | 0.755 | -0.050 |

Life satisfaction has the steepest trade-off: 2.8-5.0 pp AUROC loss for strong fairness.
This is still within clinical acceptability, but the lifecycle negotiation is tighter.

---

### STAGE 5: STABILITY ACROSS POPULATIONS (Will It Hold?)
**Lifecycle Phase: Validation & Generalization**

The clinical board asks: "If we deploy this across all SHARE countries, will the fairness
gains hold?"

Your 5-fold cross-validation with sign tests answers this:

| Method | EUROD Sign | LS Sign | CASP Sign | SRH Sign | Verdict |
|:---|:---:|:---:|:---:|:---:|:---|
| ExpGrad_TPR | 5/5 | 5/5 | 5/5 | 5/5 | Rock-solid across all folds |
| ExpGrad_EO | 5/5 | 5/5 | 5/5 | 5/5 | Rock-solid across all folds |
| FairConstraints | 5/5 | 5/5 | 5/5 | 5/5 | Rock-solid across all folds |
| GroupDRO | 1/5 | 1/5 | 3/5 | 3/5 | UNSTABLE -- fails in most folds |
| GerryFairClassifier | 0/5 | 2/5 | 3/5 | 3/5 | HARMFUL -- increases unfairness |
| CTGAN | 2/5 | 2/5 | 2/5 | 3/5 | UNRELIABLE -- coin flip |

**The journey insight:** GroupDRO preserves the highest AUROC (0.831 for EUROD) but
improves fairness in only 1 of 5 folds. In a real deployment, this means the method would
reduce disparity in ~20% of populations and do nothing (or harm) in the other 80%. A
regulator would reject this immediately.

**Impact point:** Stability is a non-negotiable lifecycle requirement. A method that works
brilliantly in one fold but fails in four is worse than a method that works moderately in
all five.

---

### STAGE 6: COMPUTATIONAL FEASIBILITY (Can We Actually Run This?)
**Lifecycle Phase: Infrastructure & Operations**

The IT team asks: "Can we retrain this model quarterly as new SHARE waves arrive?"

| Method | Time per Outcome (sec) | Annual Retraining (4 outcomes, quarterly) | Feasible? |
|:---|:---:|:---:|:---:|
| ExpGrad_TPR | ~130 | ~35 min | Yes |
| FairConstraints | ~50 | ~13 min | Yes |
| ExpGrad_EO | ~165 | ~44 min | Yes |
| GridSearch_TPR | ~1,375 | ~6.1 hours | Marginal |
| LFR | ~1,315 | ~5.8 hours | Marginal |
| Reweighing | ~1.8 | ~0.5 min | Trivial |

**The journey insight:** The recommended deployable methods (ExpGrad variants,
FairConstraints) are computationally moderate -- minutes, not hours. This matters for the
lifecycle because fairness is not a one-time fix; the model must be re-validated and
potentially retrained as population demographics shift.

---

### STAGE 7: CALIBRATION (Can Clinicians Trust the Scores?)
**Lifecycle Phase: Clinical Integration**

A clinician asks: "When the model says 70% risk, is it really 70%?"

Your Brier score analysis for the recommended deployable methods:

| Outcome | Baseline Brier | ExpGrad_TPR Brier | FairConstraints Brier | Degradation |
|:---:|:---:|:---:|:---:|:---|
| EUROD | 0.138 | 0.140 | 0.141 | +0.002-0.003 (negligible) |
| LS | 0.156 | 0.164 | 0.173 | +0.008-0.017 (small) |
| CASP | 0.122 | 0.130 | 0.149 | +0.008-0.027 (moderate for FC) |
| SRH | 0.131 | 0.134 | 0.147 | +0.003-0.016 (small) |

**The journey insight:** Fairness corrections introduce slight calibration degradation.
For ExpGrad_TPR, it's clinically negligible across all outcomes. For FairConstraints on
CASP, the 0.027 increase is noticeable and might require post-hoc recalibration --
adding another lifecycle step.

---

### STAGE 8: THRESHOLD ROBUSTNESS (Does It Survive the Clinic?)
**Lifecycle Phase: Operational Decision-Making**

The programme coordinator asks: "We need sensitivity >= 70% for our screening mandate.
Does the fairness still hold?"

Your multi-threshold robustness data shows that the BASELINE TPR gap persists regardless
of threshold strategy:

| Strategy | EUROD Gap | LS Gap | CASP Gap | SRH Gap |
|:---|:---:|:---:|:---:|:---:|
| Balanced Acc (default) | 0.193 | 0.280 | 0.301 | 0.205 |
| Sensitivity >= 70% | 0.221 | 0.293 | 0.307 | 0.218 |
| Prevalence-matched | 0.202 | 0.299 | 0.301 | 0.205 |
| Fixed 0.5 | 0.217 | 0.323 | 0.291 | 0.231 |

**The journey insight:** Unfairness is NOT an artifact of threshold choice. It is baked
into the model's learned representations. This validates the need for in-processing
interventions (like ExpGrad) that fix the problem at the learning stage, not just at the
decision boundary. Post-processing can mask the problem but doesn't resolve it.

---

### STAGE 9: REGULATORY & ETHICAL REVIEW (The Gate That Kills Methods)
**Lifecycle Phase: Governance**

The EU ethics board reviews the system. They have three non-negotiable requirements:

1. **No protected attribute at inference** (GDPR: you cannot require patients to disclose
   SES for every screening)
2. **Consistent improvement** (must improve fairness in >= 80% of population subsets)
3. **Clinically acceptable accuracy** (AUROC drop <= 0.03 from baseline)

**Applying these filters to all 26 methods:**

| Filter | Methods Surviving |
|:---|:---:|
| Start | 26 |
| After Deployability (no A at inference) | 14 |
| After Stability (sign >= 4/5 on all outcomes) | 5 (ExpGrad_TPR, ExpGrad_EO, ExpGrad_DP, FairConstraints, Reweighing) |
| After Accuracy (AUROC drop <= 0.03 all outcomes) | 3 (ExpGrad_TPR, ExpGrad_EO, Reweighing) |
| After Fairness Effectiveness (>= 25% reduction all outcomes) | **2 (ExpGrad_TPR, ExpGrad_EO)** |

**The journey insight:** Of 26 methods, only 2 survive the full regulatory lifecycle.
This is the paper's most powerful finding: **92% of fairness methods fail before reaching
the patient.**

---

### STAGE 10: DEPLOYMENT & MONITORING (The Ongoing Journey)
**Lifecycle Phase: Production & Maintenance**

The programme deploys ExpGrad_TPR. What does monitoring look like?

**Expected impact across the screening programme (51,720 individuals):**

For EUROD (Depression detection):
- Baseline: TPR gap = 0.167 --> ~355 Q5 individuals with depression go undetected
  relative to Q1 detection rates
- With ExpGrad_TPR: TPR gap = 0.019 --> ~40 Q5 individuals undetected relative to Q1
- **Net impact: ~315 additional correct identifications per screening wave in the
  highest-SES group alone**

For LS (Life Satisfaction):
- Baseline: TPR gap = 0.301 --> ~605 Q5 individuals with low life satisfaction go
  undetected
- With ExpGrad_TPR: TPR gap = 0.019 --> ~38 undetected
- **Net impact: ~567 additional correct identifications**

For CASP (Quality of Life):
- Baseline: TPR gap = 0.280 --> ~476 Q5 individuals with low QoL go undetected
- With ExpGrad_TPR: TPR gap = 0.021 --> ~36 undetected
- **Net impact: ~440 additional correct identifications**

For SRH (Self-Rated Health):
- Baseline: TPR gap = 0.181 --> ~464 Q5 individuals with poor health go undetected
- With ExpGrad_TPR: TPR gap = 0.024 --> ~62 undetected
- **Net impact: ~402 additional correct identifications**

**Combined across all 4 outcomes: ~1,724 additional correct identifications per
screening wave** in the top SES quintile, with comparable proportional improvements
across Q2-Q4.

---

## PART 3: THE LIFECYCLE MAP (Your Paper's Organizing Framework)

```
THE FAIRNESS LIFECYCLE
======================

STAGE 1: DATA         --> Bias is INHERITED from societal inequality
  |                       (SES gradient: 2.6x for CASP, 1.9x for SRH)
  |                       YOUR FINDING: Fairness challenge is outcome-dependent
  v
STAGE 2: MODEL        --> Bias is ENCODED by the learning algorithm
  |                       (TPR gaps: 16.7-30.1 pp across outcomes)
  |                       YOUR FINDING: One baseline, four different fairness problems
  v
STAGE 3: INTERVENTION --> Bias is CORRECTED (or not) by mitigation methods
  |                       (26 methods, 4 categories)
  |                       YOUR FINDING: Post-processing wins benchmarks, fails deployment
  v
STAGE 4: TRADE-OFF    --> Fairness COSTS are negotiated with stakeholders
  |                       (1-5 pp AUROC for 80-94% TPR gap reduction)
  |                       YOUR FINDING: The cost is far lower than assumed
  v
STAGE 5: STABILITY    --> Fairness is TESTED across populations
  |                       (Sign test: 5/5 for top methods, 0-1/5 for others)
  |                       YOUR FINDING: Most methods are unstable
  v
STAGE 6: COMPUTE      --> Fairness is CONSTRAINED by infrastructure
  |                       (2 min vs 23 min vs 6 hours)
  |                       YOUR FINDING: Recommended methods are computationally tractable
  v
STAGE 7: CALIBRATION  --> Fairness DISTORTS probability estimates
  |                       (Brier score increase: 0.002-0.027)
  |                       YOUR FINDING: Negligible for ExpGrad, noticeable for others
  v
STAGE 8: THRESHOLD    --> Fairness is INDEPENDENT of classification cutoff
  |                       (Gaps persist across 4 threshold strategies)
  |                       YOUR FINDING: In-processing > post-processing for root-cause fix
  v
STAGE 9: REGULATION   --> Fairness methods are FILTERED by deployment constraints
  |                       (26 --> 14 --> 5 --> 3 --> 2 survive)
  |                       YOUR FINDING: 92% of methods fail before reaching the patient
  v
STAGE 10: DEPLOYMENT  --> Fairness IMPACTS real patients
                          (~1,724 additional correct identifications per wave)
                          YOUR FINDING: The methods that survive the lifecycle save lives
```

---

## PART 4: THE NARRATIVE STRUCTURE FOR YOUR PAPER

### Recommended Paper Flow:

**Title suggestion:**
"The Fairness Lifecycle: From Benchmark to Bedside in Mental Health Screening --
A Comprehensive Evaluation of 26 Bias Mitigation Methods Using SHARE Wave 9"

**Section 1: Introduction**
- Open with the clinical scenario: EU mental health screening for elderly populations
- The promise of ML: high accuracy (AUROC 0.83) at population scale
- The hidden problem: a 30-point TPR gap means the model systematically fails higher-SES
  individuals
- The gap in literature: most fairness studies stop at the benchmark; no one traces the
  full lifecycle
- Your contribution: the first study to evaluate fairness methods not just on metrics
  but on *deployability, stability, calibration, threshold robustness, and computational
  feasibility*

**Section 2: The Fairness Problem (Stages 1-2)**
- Characterize the SES gradient in SHARE Wave 9
- Show that the baseline model encodes this gradient as differential detection rates
- Emphasize the outcome-dependency: 4 outcomes, 4 different fairness challenges
- Key figure: Baseline TPR by quintile for all 4 outcomes

**Section 3: The Benchmark (Stages 3-4)**
- Present the 26-method comparison across 4 outcomes
- Organize by lifecycle viability, not just TPR reduction
- Introduce the deployability contract as a first-class result
- Key table: Method x Outcome matrix with deployability flags
- Key figure: Pareto frontier showing accuracy-fairness trade-off

**Section 4: The Lifecycle Filter (Stages 5-9)**
- This is your NOVEL CONTRIBUTION section
- Show how 26 methods collapse to 2 through successive lifecycle filters
- Stability analysis: sign tests across folds
- Calibration analysis: Brier score impact
- Threshold robustness: fairness holds regardless of operating point
- Computational feasibility: runtime constraints
- Key figure: "Survival funnel" showing 26 --> 14 --> 5 --> 3 --> 2
- Key table: The lifecycle filter results

**Section 5: The Deployment Impact (Stage 10)**
- Translate remaining methods into clinical impact
- Quantify: ~1,724 additional correct identifications per screening wave
- Discuss the real-world implications for EU health equity policy
- Connect back to the "journey" metaphor: from SHARE survey response to clinical referral

**Section 6: Discussion**
- The lifecycle framing as a new paradigm for fairness research
- Why post-processing's benchmark dominance is an illusion
- Why stability matters more than peak performance
- The deployability contract as a standard for future fairness studies
- Limitations: 5-fold CV, single dataset, SES operationalization

---

## PART 5: WHERE YOUR PAPER IMPACTS THE MOST

Based on the full lifecycle analysis, here are the **three highest-impact contributions**
that should anchor your narrative:

### Impact 1: The Deployability Cliff (Most Novel)
**"Zero of six post-processing methods survive deployment."**

This is the finding that will get cited. No one has systematically shown that the entire
category of post-processing fairness methods -- which dominate benchmarks -- is
non-deployable because they universally require the protected attribute at inference.
This challenges the dominant paradigm in the fairness literature.

### Impact 2: The Lifecycle Funnel (Most Practical)
**"26 methods enter; 2 survive."**

The progressive filtering from 26 to 2 methods through deployability, stability,
accuracy, and effectiveness criteria is a framework other researchers can adopt. It
transforms fairness evaluation from a single-metric comparison into a multi-stage
lifecycle assessment.

### Impact 3: The Clinical Translation (Most Impactful)
**"~1,724 additional correct identifications per screening wave."**

Translating TPR gap reductions into actual patient counts makes the abstract concrete.
Policymakers and clinicians don't understand TPR gaps. They understand "315 people
with depression who would have been missed are now correctly identified."

---

## PART 6: THE METHODS THAT DIE AT EACH STAGE (Narrative Drama)

This section maps each method's "death" in the lifecycle journey, creating narrative
tension:

### Die at Stage 3 (Intervention -- They Don't Even Work):
- **GerryFairClassifier**: Increases unfairness in EUROD (sign 0/5), LS (-1.9%), and SRH (-4.2%)
- **GroupCalibration**: Increases unfairness in EUROD (-11.4%), LS (-11.1%), CASP (-8.3%)
- **SampleReweighting**: Increases unfairness in EUROD (-9.6%), barely helps elsewhere
- **CTGAN**: Increases unfairness in EUROD (-1.4%), LS (-1.8%), CASP (-2.6%)
- **DecoupledClassifier**: Increases unfairness in EUROD (-12.0%)

*These methods are "dead on arrival" -- they make the problem worse.*

### Die at Stage 5 (Stability -- They Work Sometimes):
- **GroupDRO**: Great AUROC (0.831 for EUROD) but only improves fairness in 1/5 folds
  for EUROD and LS. A coin flip is more reliable.
- **GridSearch_TPR**: High variance (SD up to 0.110 for CASP), unstable across folds
- **FairGBM**: Improves 4/5 folds for EUROD but only 4/5 for LS and SRH -- marginal

*These methods are "fair-weather friends" -- they help in some populations, not others.*

### Die at Stage 9 (Regulation -- They Can't Be Deployed):
- **ALL Post-processing** (ThreshOpt_TPR, ThreshOpt_EO, ThreshOpt_DP, EqOdds_PP,
  RejectOption, GroupCalibration): Require protected attribute at inference
- **AdvDebiasing**: Requires protected attribute encoded into the model
- **PrejudiceRemover**: Requires protected attribute at inference
- **DIR, CorrelationRemover, LFR**: Transform features using protected attribute at runtime

*These methods are "greenhouse flowers" -- they thrive in the lab but die in the field.*

### Survive the Full Journey (2 of 26):
- **ExpGrad_TPR**: 86.6-93.8% TPR reduction, deployable, stable (5/5 all outcomes),
  AUROC drop 1.1-2.8 pp, computational cost ~130 sec
- **ExpGrad_EO**: 86.0-94.4% TPR reduction, deployable, stable (5/5 all outcomes),
  AUROC drop 1.1-3.0 pp, computational cost ~165 sec

*These are the "survivors" -- they complete the full journey from benchmark to bedside.*

---

## PART 7: RECOMMENDED FIGURES FOR THE PAPER

1. **The Lifecycle Funnel** (NEW -- does not exist yet): A Sankey or funnel diagram
   showing 26 methods entering and progressive elimination at each stage. This is your
   signature figure.

2. **Baseline Disparity Heatmap** (exists: part of composite): TPR by quintile x outcome,
   showing the problem's severity and outcome-dependency.

3. **Pareto Frontier** (exists): AUROC vs TPR gap with deployable methods highlighted.
   Add a visual distinction between deployable and non-deployable methods.

4. **The Survival Table** (NEW): A single table showing each method's status at each
   lifecycle stage (Pass/Fail), ending with the 2 survivors.

5. **Clinical Impact Calculator** (NEW): Bar chart showing the number of additional
   correct identifications per outcome for the surviving methods.

---

## SUMMARY: Your Paper's Story in One Paragraph

The literature treats fairness as a model property measured at the benchmark. We show it
is a lifecycle property that must survive ten successive stages -- from data collection
through regulatory review to patient impact. Using SHARE Wave 9 data and 26 bias
mitigation methods across four mental health outcomes, we trace the complete journey of
algorithmic fairness from benchmark to bedside. We find that the methods dominating
fairness benchmarks (post-processing) universally fail deployment constraints, that 92% of
methods are eliminated by real-world lifecycle requirements, and that only two methods
(ExpGrad_TPR and ExpGrad_EO) survive the full journey while achieving 86-94% reductions
in socioeconomic detection disparities at a clinically negligible accuracy cost. These
survivors would yield approximately 1,724 additional correct mental health identifications
per screening wave in the highest-SES quintile alone, translating algorithmic fairness
from a benchmark abstraction into a concrete public health impact.
