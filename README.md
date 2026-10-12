# CISC3024 AI Assignment 2: diagonal VBLL adaptation

All source code was written by the AI assistant. The student does not need to write code.
This is an educational adaptation of Harrison, Willes and Snoek (ICLR 2024),
not an exact reproduction of the official implementation or benchmark.

## Run

Requires Python 3.12 (tested); CPU only. From this directory:

```bash
python -m pip install -r requirements.txt
python run_experiment.py
```

Optional on Linux/macOS, to limit BLAS threads:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiment.py
```

Data: the bundled sklearn load_digits dataset. No downloads or GPU required.
Results can differ slightly across BLAS implementations or dependency versions.

## Files

- run_experiment.py: AI-written experiment and analytic derivatives.
- results.csv: all 18 model/seed/noise evaluations.
- results.json: summaries, optimizer status, feature-learning warnings, versions.
- predictions_seed0.npz: clean VBLL test probabilities and original dataset indices.
- confusion.png / robustness.png: generated result figures.
- AI_Assignment2_Report.pdf: English report; the personal code URL still needs replacing.
- report.md: editable report text, generated from the same content as the PDF.

## Method

Train a 64-32-10 ReLU MLP only on the training partition, discard its output
layer and freeze its 32-dimensional hidden features. Add an intercept feature.
Train both a MAP softmax head and a diagonal-Gaussian VBLL head on the same
features. Select each head's prior variance from 0.1, 1, 10 using validation NLL.
Use a deterministic Jensen lower bound for VBLL training and 512 Gaussian
logit samples for posterior predictive evaluation. There is no learned logit-noise
parameter. These restrictions are documented in the report.

## Publish the assignment code webpage (no coding needed)

1. Sign in to your own GitHub account and click + -> New repository.
2. Name it cisc3024-ai-assignment2 and choose Public so the instructor can open it.
3. Choose Add file -> Upload files and upload run_experiment.py, requirements.txt,
   README.md, results.csv, results.json and the two PNG figures.
4. Click Commit changes. Open run_experiment.py in the repository and copy its URL.
5. Replace the pending URL text in section 6 of the report with that actual URL.
   You may send the URL to the AI assistant to regenerate the PDF.

Uploading existing files is not writing code. Check course policy before making
an assignment repository public; use a private repository with instructor access
if required. Do not substitute the original authors' repository for your own code.

## Sources

Paper: https://arxiv.org/abs/2404.11599
Official venue: https://proceedings.iclr.cc/paper_files/paper/2024/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html
Official package: https://github.com/VectorInstitute/vbll
Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
