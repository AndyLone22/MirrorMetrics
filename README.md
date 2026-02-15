# 🔬 MirrorMetrics: AI Identity Consistency Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)

**MirrorMetrics** is a scientific benchmarking tool for evaluating **Face LoRAs** (Stable Diffusion fine-tuned models). It uses [InsightFace](https://github.com/deepinsight/insightface) (ArcFace) to perform local biometric analysis and generates a rich, interactive **Plotly** dashboard — all running entirely on your machine.

**In two quick and easy words** 
You put the dataset you used to train your LoRA in the Reference_Images folder and the generated images in the Lora_Candidates folder (Create one folder for each LoRA you want to compare. The name of the folder will be used as the name of the LoRA in the dashboard.). 

You can have more LoRAs from the same training, maybe with different settings, or different steps, for example. Or you could have LoRA trained for entirely different models, that's fine too. 

Then you run the script and it will generate a dashboard with all the metrics.
The way I use it the most at the moment is:
-compare the plots for the dataset, to see if I have some outliers that can skew the results.
-compare the plots for the LoRAs, to see which one is the most consistent with the dataset, especially graph one which shows the general similarity score of the various images compared to the median of the dataset.
-look at the plots 5 and 6 to see if the LoRA is good at generating the face in different angles and how much does it tend to to so (of course mind the prompts you're using, if you specifically tell it to generate a right profile image of course you'll have a bunch of dots on the right side of the graph.)


> _"How consistent is the identity your LoRA generates?"_ — MirrorMetrics answers this question with data.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧬 **Biometric Similarity** | Cosine similarity between face embeddings and a reference centroid |
| 🔄 **Leave-One-Out (LOO)** | Robust reference scoring that excludes each image from its own centroid calculation — great for cleaning noisy datasets |
| 📊 **Interactive Plotly Dashboard** | 7-panel dark-themed HTML dashboard with floating control panel |
| 🎯 **Pose Analysis** | Yaw / Pitch scatter plots to evaluate identity stability across head orientations |
| 🗺️ **t-SNE Identity Map** | 2D projection of face embeddings to visualize identity clusters |
| 👤 **Age & Gender Detection** | Per-image age estimation and gender classification |
| 🔒 **Privacy-Focused** | Everything runs locally — no images are ever uploaded |

---

## 📋 Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** with CUDA toolkit installed (for `onnxruntime-gpu`)
  - _If you don't have a CUDA GPU, edit `requirements.txt` and replace `onnxruntime-gpu` with `onnxruntime`, then change the provider in `mirror_metrics.py` to `CPUExecutionProvider`._

---

## � Installation

```bash
# Clone the repository
git clone https://github.com/AndyLone22/MirrorMetrics.git
cd MirrorMetrics

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> On the first run, InsightFace will automatically download the `buffalo_l` model (~300 MB). This is a one-time operation.

---

## 📖 Usage

### 1. Prepare your folders

```
MirrorMetrics/
├── Reference_Images/       ← Your real reference photos (the "ground truth")
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── Lora_Candidates/        ← One subfolder per LoRA to evaluate
│   ├── MyLoRA_v1/
│   │   ├── gen_001.png
│   │   └── gen_002.png
│   ├── MyLoRA_v2/
│   │   └── ...
│   └── ...
└── mirror_metrics.py
```

- **`Reference_Images/`** — Place your real face photos here. More variety (angles, lighting) = better centroid.
- **`Lora_Candidates/`** — Create one subfolder per LoRA (or per experiment). Each subfolder will appear as a separate group in the dashboard.

### 2. Run the analysis

```bash
python mirror_metrics.py
```

### 3. View the results

The script generates two output files in the project root:

| File | Description |
|---|---|
| `Dashboard_<timestamp>.html` | Interactive Plotly dashboard — open in any browser |
| `Data_<timestamp>.csv` | Raw data export for further analysis |

---

## 📊 Dashboard Panels

The generated dashboard contains **7 analysis panels**:

1. **Face Similarity** — Box plot of cosine similarity scores per group
2. **Age Distribution** — Violin plot of estimated ages
3. **Face Ratio** — Bounding-box aspect ratio distribution
4. **Detection Confidence** — Face detector confidence scores
5. **Profile Stability** — Similarity vs. absolute yaw angle (does identity hold in profile views?)
6. **Pose Variety** — Yaw vs. Pitch bubble chart (bubble size = identity strength)
7. **Identity Map** — t-SNE 2D projection of face embeddings

A **floating control panel** lets you toggle individual groups on/off or cycle through them in solo mode.

---

## 🧪 Methodology

### Leave-One-Out (LOO) Scoring

For **reference images**, each photo is scored against the centroid of _all other_ reference images (excluding itself). This prevents inflated self-similarity and helps identify outlier photos in your reference set.

For **generated images** (LoRA candidates), each photo is compared against the full reference centroid.

### t-SNE Visualization

All face embeddings (reference + generated) are projected into 2D using t-SNE. Tight clusters indicate consistent identity; scattered points suggest identity drift.

---

## ⚙️ Configuration

You can customize these variables at the top of `mirror_metrics.py`:

| Variable | Default | Description |
|---|---|---|
| `PATH_REFERENCE` | `Reference_Images` | Path to reference images folder |
| `PATH_CANDIDATES_ROOT` | `Lora_Candidates` | Path to LoRA candidates root folder |
| `EXTENSIONS` | `jpg, jpeg, png, webp, bmp` | Supported image formats |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.