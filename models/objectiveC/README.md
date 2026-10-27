# Objective C: federated learning across hospitals

Trains the disease, treatment, atypical-case and outbreak models across hospitals H001–H010 with
[Flower](https://flower.ai), so every hospital benefits from all hospitals' data without any patient row
leaving its hospital. Nothing in `models/objectiveA|B|E` is modified: their code is loaded as it is, and
the federated models are written next to this README, never over the production ones. The `api.py` files of A, B and
E1/E2 load them from `model/<task>/` (see "In the web app").

| Task | Model | How it is federated | Output |
|---|---|---|---|
| `disease` | XGBoost multi-class (Objective E1) | hospitals take turns adding trees to one shared model (`FedXgbCyclic`) | `model/disease/` |
| `treatment` | XGBoost success + recovery days (E2) | same, both models per turn; candidate table from one query round | `model/treatment/` |
| `atypical` | statistical detector (A) | hospitals send counts, sums and histograms; distributed EM for the vitals mixtures | `model/atypical/` |
| `outbreak` | cross-site outbreak detector (B) | hospitals send day×disease count summaries (built with the federated A detector) and their labelled outbreak episodes; B's own calibration runs on those | `model/outbreak/` |

Run `atypical` before `outbreak`: its model is the detector each hospital uses locally.

## Run it

Needs Python ≥ 3.11 and `pip install -r requirements.txt` (adds `flwr[simulation]`). From the repo root, with
`CAREPATH_ROOT` set to the repo and the environment's `bin` on `PATH` (Flower installs a copy of the app, so it
cannot find `data/` from its own location):

```bash
export CAREPATH_ROOT=$PWD
flwr run models/objectiveC --run-config 'task="treatment"'      # string values must be quoted
python -m carepath_fl.report --task treatment                   # per-hospital comparison (run from models/objectiveC)
pytest models/objectiveC/tests
```

`flwr run` also creates `~/.flwr/` (config, installed app copies, runtime environments).

### Real deployment (separate processes, TLS, per-hospital keys)

```bash
python models/objectiveC/deploy/credentials.py                  # CA, SuperLink certificate, one key pair per hospital
python models/objectiveC/deploy/local_federation.py run --task disease
python models/objectiveC/deploy/local_federation.py check-security
```

`run` starts a SuperLink on port 9092 (the web app uses 8000, so both can run at once) with TLS and `--enable-supernode-auth`, registers each hospital's public key, starts
one SuperNode per hospital that connects over TLS with its own private key, runs the task, and stops
everything. State goes to `deploy/work/` (via `FLWR_HOME`), credentials to `deploy/secrets/`; both are
gitignored. It is a single-machine demonstration of the real network path: in production each hospital runs
its own SuperNode next to its own data and receives only its own key.

`check-security` verifies that an unregistered key is refused (and that the same key connects once
registered), that the SuperLink does not answer plaintext HTTP, that a SuperNode will not authenticate
without TLS, and that a node not trusting the SuperLink's CA cannot connect.

The server also binds each authenticated node to a hospital (`carepath_fl/common/identity.py`): a reply that
names a different hospital than the node it came from is rejected, as is a missing or duplicate hospital in
a stage that pools everyone. This is only active in the deployment, where a registry is written.

## Results

All on the held-out test split, weighted over hospitals; "alone" = each hospital training only on its own
data with the centralised trainer's method; "centralised" pools every hospital's rows (simulation only).

| Task | Metric | Alone | Federated | Centralised |
|---|---|---|---|---|
| disease | accuracy | 95.85% | 96.55–96.74% (6 runs) | 96.50% |
| treatment | success AUC | 0.6047 | 0.611–0.615 (2 runs) | 0.6159 |
| treatment | recovery MAE (days) | 1.545 | 1.518–1.519 (2 runs) | 1.513 |
| atypical | AUC | 0.846 | 0.860 | 0.858 |
| atypical | recall at ≈1.7% false alarms | 0.397 | 0.431 | 0.416 |
| outbreak | all held-out metrics | – | identical to the centralised pipeline | |

What these do and do not show:

* Federated beats training alone at 10/10 hospitals for disease accuracy and recovery MAE, 9/10 for atypical AUC;
  the treatment success AUC gain is small and only 6–8 of 10 hospitals improve across two runs (the signal is weak: AUC ≈ 0.61).
* Per-hospital rare-class numbers (e.g. disease D003) rest on 30–60 test cases and are noisy.
* Cyclic training order is random per run, so results move a little between runs.
* Disease turns/iterations (30×10) were chosen after looking at test-split results, so its test accuracy is
  slightly optimistic. The treatment settings were chosen on per-hospital validation slices only. The
  federated disease model has 300 boosting iterations; the baselines use 100.
* The atypical statistics match the centralised trainer exactly where they are sums (priors, symptom
  probabilities, vitals mean/SD) and to about 1% where they are histogram quantiles.
* Outbreak parity is expected: its detector works on count summaries, and the federated atypical detector
  changes no alert outcome on the test period (21 labelled outbreaks, so few events to compare).

## In the web app

The main web app has a **Federated learning** tab (`frontend/federated.js`, served by the same backend as the
other tabs; `http://localhost:8000/#federated` opens it directly). It shows the hospitals and what travels, the
federated vs alone vs pooled results with a hospital-by-hospital chart, a replay of the recorded training
(which hospital took each turn and how the model improved), and what is and is not protected.

It reads `GET /api/federated/overview` (`models/objectiveC/api.py`), which only reads the JSON files in
`model/<task>/`; it never imports Flower, trains, or serves a model. After retraining a task, run
`python -m carepath_fl.report --task <task>` (from `models/objectiveC`) to refresh what the tab shows. A task
with no report shows the command to train it. The replay needs a run made after turn history was added to the
training (older models show results without the replay).

The prediction endpoints of the other tabs load the federated files from `model/<task>/` (the `api.py` of
Objectives A, B, E1 and E2 point there). The original, centrally trained files are still in each objective's own
`model/` folder, untouched, and `predict.py` / `train.py` still use them.

### Original vs federated comparison

The **Model comparison** tab (`frontend/comparison.js`, `GET /api/federated/comparison`) scores the original models
and the federated ones on the same held-out patients, in total and per hospital, with a paired-bootstrap 95%
interval for each difference. The numbers come from `models/objectiveC/model/comparison.json`, written by:

```bash
python -m carepath_fl.compare          # from models/objectiveC, CAREPATH_ROOT set; ~30 s, trains nothing
```

It loads both sets the way the web app does and checks that the originals score what their own training recorded
(the run fails if they do not). A difference is called better or worse only when its interval excludes zero. The
outbreak detector has no per-patient score, so its two recorded held-out results are shown side by side. Run it
again after retraining any model. Current picture: the federated models are not distinguishable from the originals
on 7 of the 8 measures that can be tested, and better on atypical-case AUC (+0.005).

## What leaves a hospital, and what is not protected

| Task | Leaves the hospital |
|---|---|
| disease, treatment | the shared model after the hospital's turn (its new trees), evaluation metrics; treatment also a list of disease/severity/treatment combinations seen |
| atypical | per-disease counts and sums, per-component sums for the EM rounds, and histograms of one-dimensional scores |
| outbreak | day × disease case counts with 28 symptom counts, and each labelled outbreak's disease and first/last day |

Clients read only whitelisted columns; tests check that no identifier column reaches a shared model and that
stage replies contain only aggregate-shaped arrays.

Not covered: there is **no differential privacy and no secure aggregation**. Each hospital sees the full shared
model (which contains splits learned from earlier hospitals' data), the server sees each hospital's
statistics individually, and small daily counts can still carry information about individuals. Suppressing
small cells in the outbreak summaries was considered and not done: 73% of the non-empty
hospital×disease×day counts are below 3, so suppression would remove most of the symptom signal the
detector uses. The generated TLS
certificates and keys are prototype credentials for a private network.

## Not done

There is no scheduler for retraining.
