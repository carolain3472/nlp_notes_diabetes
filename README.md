# Clinical Note De-identification in Spanish

Masked-language-model approaches to de-identifying Spanish clinical narratives,
comparing a general-domain, a clinical-domain and a diabetes-adapted encoder.

Research internship, **Instituto Tecnológico de Morelia** (Programa Delfín), Mexico.

---

## What this is

Spanish clinical text is under-served compared to English. The models that exist
are trained on general web Spanish or on biomedical abstracts, neither of which
looks much like a hospital note: notes are terse, abbreviation-heavy, and full of
the kind of identifying detail that has to be removed before the text can be
shared or reused.

This project builds a two-stage pipeline over the DISTEMIST corpus and asks a
narrow question: **does domain adaptation actually help a masked language model
recover clinical text, and does it help more if you adapt it to the specific
condition you care about?**

Three encoders are compared on the same fill-mask task:

| Domain | Model |
|---|---|
| General Spanish | [`dccuchile/bert-base-spanish-wwm-cased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased) (BETO) |
| Clinical Spanish | [`plncmm/beto-clinical-wl-es`](https://huggingface.co/plncmm/beto-clinical-wl-es) |
| Biomedical-clinical Spanish | [`PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`](https://huggingface.co/PlanTL-GOB-ES/roberta-base-biomedical-clinical-es) |

Each is further fine-tuned with `RobertaForMaskedLM` + `Trainer` on the corpus,
producing a **diabetes-adapted** variant alongside the general and clinical ones.

## Pipeline

### 1 — Preprocessing and corpus characterisation

[`notebooks/01_preprocesamiento.ipynb`](notebooks/01_preprocesamiento.ipynb)

Normalises each note (case folding, accent-aware character filtering,
`unidecode`, spaCy tokenisation) and then profiles the corpus, because you cannot
choose a masking strategy without knowing what is in the text:

- **Length distribution** — notes bucketed at ≤200, 200–400, 400–600, >600 words,
  which determines the truncation budget for a 512-token encoder
- **Disease mentions** — a 20-term lexicon (diabetes, cáncer, hipertensión, asma,
  artritis, obesidad, epilepsia, alzheimer, parkinson, anemia…) counted per note
- **Psychological comorbidity** — `ansiedad`, `depresión`, `estrés`:
  **312 mentions across 195 notes**, of which 154 are anxiety, 104 depression and
  54 stress. Roughly two in five notes mention at least one, which is what
  motivated keeping these terms out of the maskable set

### 2 — Masking and evaluation

[`notebooks/02_enmascaramiento.ipynb`](notebooks/02_enmascaramiento.ipynb)

Fine-tunes each encoder with `DataCollatorForLanguageModeling`, then evaluates
fill-mask recovery per domain. Model outputs are scored with an LLM-as-judge step
via the OpenAI API, comparing the predicted token against the original span.

## Data

**The corpus is not included in this repository, by design.**

The notes come from **DISTEMIST** (DISease TExt MIning Shared Task), released by
the Barcelona Supercomputing Center as part of BioASQ. It is built from published,
de-identified Spanish clinical case reports — not hospital records — and is
distributed under its own terms.

> Miranda-Escalada, A., Gascó, L., Lima-López, S., Farré-Maduell, E., Estrada, D.,
> Nentidis, A., Krithara, A., Katsimpras, G., Paliouras, G., & Krallinger, M. (2022).
> *Overview of DISTEMIST at BioASQ: Automatic detection and normalization of
> diseases from clinical texts.* CLEF 2022.

Download it from the [official release](https://zenodo.org/records/6671292) and
place the note files under `data/`. Nothing in `data/` is tracked.

**On reuse.** Even with a public corpus, derived artefacts deserve care. Trained
weights can memorise training text, so the fine-tuned checkpoints are not
published here either. If you adapt this pipeline to records from a hospital
rather than a public corpus, the corpus is the least of your obligations — that
is a matter for your institution's ethics board, not a `.gitignore`.

## Running it

```bash
git clone https://github.com/carolain3472/nlp_notes_diabetes.git
cd nlp_notes_diabetes
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # add your OpenAI key for the evaluation step
```

Then open the notebooks in order. Both were written in Google Colab and still
mount Drive for checkpoint storage; running locally means pointing those paths at
`data/` and `models/` instead.

Outputs have been stripped from both notebooks — they had carried the corpus text
inline, which is exactly how a "de-identified" dataset stops being one.

## Repository layout

```
notebooks/
  01_preprocesamiento.ipynb   normalisation, tokenisation, corpus profiling
  02_enmascaramiento.ipynb    fine-tuning, fill-mask inference, evaluation
data/                         corpus goes here — not tracked
requirements.txt
```

## Known limitations

- Evaluation leans on an LLM judge rather than a labelled gold standard for the
  masked spans; a proper span-level precision/recall against manual annotation is
  the obvious next step
- The disease and disorder lexicons are hand-built and Spanish-specific, so they
  miss abbreviations and misspellings that are common in real notes
- Checkpoint paths are still Colab-shaped and need rewriting for a local run

## License

Code released under the MIT License. The DISTEMIST corpus keeps its own terms.
