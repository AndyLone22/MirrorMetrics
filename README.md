# 🔬 MirrorMetrics: AI Identity Consistency Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)

**MirrorMetrics** is a scientific benchmarking tool for evaluating **Face LoRAs** (Stable Diffusion fine-tuned models). It uses [InsightFace](https://github.com/deepinsight/insightface) (ArcFace) to perform local biometric analysis and generates a rich, interactive **Plotly** dashboard — all running entirely on your machine.

## Why use this tool? (The Problem)
Training a LoRA often feels like guessing. You might ask:
* **How do I know if my LoRA is overtrained?**
* **Why does my character look rigid?**
* **Is my dataset consistent?**

MirrorMetrics solves this by measuring **Identity Consistency**, **Face Geometry**, and **Flexibility** mathematically.

**In two quick and easy words** 

You put the dataset you used to train your LoRA in the Reference_Images folder and the generated images in the Lora_Candidates folder (Create one folder for each LoRA you want to compare. The name of the folder will be used as the name of the LoRA in the dashboard.). 

You can have more LoRAs from the same training, maybe with different settings, or different steps, for example. Or you could have LoRA trained for entirely different models, that's fine too. 

Then you run the script and it will generate a dashboard with all the metrics.
The way I use it the most at the moment is:

- Compare the plots for the dataset, to see if I have some outliers that can skew the results.
- Compare the plots for the LoRAs, to see which one is the most consistent with the dataset, especially graph one which shows the general similarity score of the various images compared to the median of the dataset.
- Look at the plots 5 and 6 to see if the LoRA is good at generating the face in different angles and how much does it tend to do so (of course mind the prompts you're using, if you specifically tell it to generate a right profile image of course you'll have a bunch of dots on the right side of the graph).


> _"How consistent is the identity your LoRA generates?"_ — MirrorMetrics answers this question with data.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧬 **Biometric Similarity** | Cosine similarity between face embeddings and a reference centroid |
| 🔄 **Leave-One-Out (LOO)** | Robust reference scoring that excludes each image from its own centroid calculation — great for cleaning noisy datasets |
| 📊 **Interactive Plotly Dashboard** | 7-panel dark-themed HTML dashboard with floating control panel |
| 👯 **Copycat Detector** | Standalone HTML report pairing each generated image with its nearest reference neighbor — instantly spot memorization vs. generalization |
| 🕳️ **Black Hole Ranking** | Ranked list of reference images by how many generated images point to them — find dominant training samples |
| 🎯 **Pose Analysis** | Yaw / Pitch scatter plots to evaluate identity stability across head orientations |
| 🗺️ **t-SNE Identity Map** | 2D projection of face embeddings to visualize identity clusters |
| 👤 **Age Detection** | Per-image age estimation via deep learning |
| 🔄 **Close-Up Rescue** | Automatic padding retry for close-up faces that fill the entire frame (InsightFace workaround) |
| 🔒 **Privacy-Focused** | Everything runs locally — no images are ever uploaded |

---

## 📋 Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** _(recommended, not required)_ — the script runs on CPU too, just slower. See the installation tips below.

---

## 🚀 Installation

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

> [!IMPORTANT]
> The `requirements.txt` includes all NVIDIA CUDA libraries (cuBLAS, cuDNN, etc.) so the setup is **fully standalone** — no system-level CUDA Toolkit installation needed. This makes the total venv size **~4.5 GB**.

> [!TIP]
> **Already have CUDA 12 installed system-wide?** You can save ~3 GB by removing all `nvidia-*` lines from `requirements.txt` before running `pip install`. The script will use your system CUDA libraries instead.
>
> **No NVIDIA GPU?** Replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt` and remove all `nvidia-*` lines. The script will run on CPU (slower but functional).

> [!NOTE]
> On the first run, InsightFace will automatically download the `buffalo_l` model (~300 MB). This is a one-time operation.

---

## 📖 Usage

### 1. Prepare your folders.
It's imperative to use good structured datasets for the Analysis: 
The reference images should be the ones used in the dataset for Training the LoRAs.
Then with those LoRAs, produce at least 10 different prompts, in batches of 3 per prompts, to have a good variety, and mind to have differentiated prompts that reach for many positions, angles, situations... If you only produce images of portraits, of course the variance will be flat, but that won't mean that the model is not variable, just that the dataset was created poorly. I suggest you decide 10 standard prompts and always use those, so that you'll learn better how to read the results with experience.

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

**Windows:** Just double-click `run.bat` — it activates the venv and launches the script automatically.

**Linux / macOS:** Run `chmod +x run.sh` once, then `./run.sh`.

**Or manually from any platform:**

```bash
python mirror_metrics.py
```

### 3. View the results

The script generates three output files in the project root:

| File | Description |
|---|---|
| `Dashboard_<timestamp>.html` | Interactive Plotly dashboard — open in any browser |
| `CopycatReport_<timestamp>.html` | Nearest-neighbor visualizer — see which reference image each generation resembles most |
| `Data_<timestamp>.csv` | Raw data export for further analysis (includes failed detections) |

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

## 👯 Copycat Detector

The Copycat Detector is a standalone HTML report that answers: _"Is my model actually learning the concept, or is it just memorizing a specific training image?"_

For each **generated image**, it finds the **nearest reference image** (by cosine similarity on ArcFace embeddings) and displays them side-by-side in a visual grid.

### What it shows

- **Card grid** — Generated image (left) paired with its closest reference match (right)
- **Color-coded similarity** — 🟢 OK (0.50–0.70) · 🟡 Watch Out (0.70–0.85) · 🔴 Danger Zone (>0.85) · ⚪ LOW sim (<0.50)
- **Failed detections** — Images where no face was found are shown with a "⚠ Failed to recognize face" label
- **Black Hole Ranking** — All reference images ranked by how many generated images point to them (most → least). If many generated images converge on the same reference, that training sample may be dominating the model
- **Sort & Filter** — JS controls to sort by similarity or group, and filter by LoRA

> [!TIP]
> If you see 10 different generated images all pointing to the same reference photo, you've found a **"black hole"** in your dataset — an image so dominant it's pulling everything toward it. Consider removing or de-emphasizing it in your training set.

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

## FAQ

### How do I check if my LoRA is overtrained?
Look at the **Face Similarity** chart. If the score is extremely high (>0.85) on all images and the **Face Ratio** variance is near zero, your model is likely overfitted (memorizing pixels instead of concepts).

### Can I use this for Flux or SDXL?
Yes. MirrorMetrics works on the *output images*, so it is compatible with **Stable Diffusion 1.5, SDXL, Pony, Flux, QWEN, Z-Image and any other model of any kind**.

### How to evaluate dataset quality?
Use the "Reference" box in the charts. If the purple box is very tall or has outliers, check those images to see why they give a high difference evaluation to the mean of the rest of the images, then discard them if you feel it's best.

### Why does the Age Distribution vary so much?
This usually indicates inconsistent skin texture in your dataset. The biometric engine uses high-frequency details (pores, wrinkles, skin smoothing) to estimate age. If you mixed "soft" lighting images with "harsh" realistic photos, the tool might interpret the texture difference as an age difference.

### Why does Detection Confidence drop on good images?
This often happens with extreme angles (e.g. looking back over the shoulder, steep profiles). The detector expects a standard facial geometry, so perspective compression can lower the confidence score even if the anatomy is correct. Low confidence on a profile shot is acceptable; low confidence on a front-facing shot means your model is broken, so interpretation of the data is always needed!

### Why does face detection fail on close-up portraits?
InsightFace sometimes struggles when a face fills the entire frame (extreme close-ups). MirrorMetrics includes a **Close-Up Rescue** mechanism: if the first detection attempt fails, it automatically adds black padding around the image to simulate a "zoom out" and retries. You'll see a `🔄 Rescued via padding` message in the terminal when this kicks in. If it still fails, the image is marked as "Failed" and shown in both the CSV and the Copycat Report.

### Can I use this tool before training?
Yes! You can run the tool pointing only to your Dataset folder to analyze the Purple Box (Reference). If the box is very tall or has dots floating far below it, you have "poison" images (outliers) in your dataset. Removing them before training could save you time and GPU hours, but it's data: not a suggestion so you always must interpret the data before deciding how to act.

### How do I measure Flexibility (Creativity)?
Look at the Face Ratio and Pose Variety charts. If the Face Ratio is a flat line and Pose Variety is clustered at the center, the model is just "photocopying" the data (Low Creativity). If the charts show wide variance (like a violin shape) and scattered dots, the model understands the 3D structure well enough to generate new expressions and angles (High Flexibility). Of course all this needs to have a good set of produced images to evaluate with different poses, angles, positions, backgrounds etc...


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE.txt](LICENSE.txt) file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.
