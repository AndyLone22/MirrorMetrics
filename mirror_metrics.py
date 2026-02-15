import os
import sys

# ==========================================
# CUDA DLL AUTO-DISCOVERY (Windows)
# ==========================================
# Automatically locates NVIDIA libraries installed via pip
# (nvidia-cublas-cu12, nvidia-cudnn-cu12, etc.) so that
# onnxruntime-gpu can find them without system-level PATH changes.
if sys.platform == "win32":
    import importlib.util
    _nvidia_packages = [
        "nvidia.cublas",
        "nvidia.cudnn",
        "nvidia.cuda_runtime",
        "nvidia.cuda_nvrtc",
        "nvidia.cufft",
        "nvidia.curand",
        "nvidia.cusolver",
        "nvidia.cusparse",
        "nvidia.nccl",
        "nvidia.nvjitlink",
    ]
    for _pkg in _nvidia_packages:
        _spec = importlib.util.find_spec(_pkg)
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                _bin = os.path.join(_loc, "bin")
                _lib = os.path.join(_loc, "lib")
                for _dir in (_bin, _lib):
                    if os.path.isdir(_dir):
                        os.add_dll_directory(_dir)
                        os.environ["PATH"] = _dir + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pandas as pd
from tqdm import tqdm
import glob
import shutil
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE

# ==========================================
# ### DASHBOARD CONFIGURATION ###
# ==========================================

PATH_REFERENCE = r"Reference_Images"
PATH_CANDIDATES_ROOT = r"Lora_Candidates"
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_HTML = f"Dashboard_{TIMESTAMP}.html"
OUTPUT_CSV = f"Data_{TIMESTAMP}.csv"

# ==========================================
# END CONFIGURATION
# ==========================================

class FaceAnalyzer:
    def __init__(self):
        print(">>> Initializing InsightFace (GPU)...")
        # Use ['CPUExecutionProvider'] if you don't have CUDA properly configured
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        print(">>> System ready.")

    def analyze_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        
        faces = self.app.get(img)
        if len(faces) == 0: return None
        
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        pitch, yaw, roll = face.pose if face.pose is not None else (0,0,0)
        
        width = face.bbox[2] - face.bbox[0]
        height = face.bbox[3] - face.bbox[1]
        aspect_ratio = width / height

        return {
            'embedding': face.normed_embedding,
            'age': face.age,

            'pitch': pitch, 'yaw': yaw, 'roll': roll,
            'det_score': face.det_score if hasattr(face, 'det_score') else 0.99,
            'aspect_ratio': aspect_ratio,
            'bbox': face.bbox, 'img_obj': img 
        }

    def process_dataset(self):
        dataset_data = [] 
        
        # 1. Process REFERENCE images
        print(f"\n--- Analyzing Reference: {PATH_REFERENCE} ---")
        if not os.path.isdir(PATH_REFERENCE):
            raise FileNotFoundError(
                f"Reference folder '{PATH_REFERENCE}' not found. "
                f"Please create it and add your reference face images."
            )
        ref_files = []
        for ext in EXTENSIONS:
            ref_files.extend(glob.glob(os.path.join(PATH_REFERENCE, ext)))
            
        # FIRST PASS: Extract all embeddings and data
        ref_results = []
        for f in tqdm(ref_files, desc="Extracting Reference Data"):
            data = self.analyze_image(f)
            if data:
                ref_results.append({
                    'file': f,
                    'data': data
                })
        
        if not ref_results:
            raise ValueError("No faces found in Reference images! Please check your images.")

        # Build a numpy matrix with all reference embeddings
        ref_embeddings_matrix = np.stack([r['data']['embedding'] for r in ref_results])
        
        # SECOND PASS: Compute similarity using Leave-One-Out (LOO)
        n_refs = len(ref_results)
        
        print(">>> Computing Reference similarity (Leave-One-Out)...")
        for i, r in enumerate(ref_results):
            data = r['data']
            filename = os.path.basename(r['file'])
            
            # Compute centroid excluding the current image
            if n_refs > 1:
                # Mask to select all indices except i
                mask = np.arange(n_refs) != i
                other_embeddings = ref_embeddings_matrix[mask]
                
                # Centroid of the "others"
                loo_centroid = np.mean(other_embeddings, axis=0)
                loo_centroid = loo_centroid / np.linalg.norm(loo_centroid)
                
                # Similarity between the current image and the others' centroid
                sim = float(np.dot(data['embedding'], loo_centroid))
            else:
                # If there's only one image, similarity is necessarily 1.0 (or undefined)
                sim = 1.0

            dataset_data.append({
                'Type': 'Reference', 'Group': 'Reference',
                'Filename': filename,
                'Similarity': sim, # This value is now meaningful!
                'Age': data['age'],
                'Yaw': data['yaw'], 'Pitch': data['pitch'], 'Roll': data['roll'],
                'DetScore': data['det_score'],
                'AspectRatio': data['aspect_ratio'],
                'Embedding': data['embedding']
            })

        # Compute the FULL centroid for evaluating LoRA candidates
        # Here we use all reference images since LoRAs are external
        centroid_full = np.mean(ref_embeddings_matrix, axis=0)
        centroid_full = centroid_full / np.linalg.norm(centroid_full)

        # 2. Process LORA CANDIDATES
        if not os.path.exists(PATH_CANDIDATES_ROOT):
            print(f"WARNING: Folder {PATH_CANDIDATES_ROOT} not found.")
            return pd.DataFrame(dataset_data)

        subfolders = [f.path for f in os.scandir(PATH_CANDIDATES_ROOT) if f.is_dir()]
        
        for folder in subfolders:
            lora_name = os.path.basename(folder)
            print(f"\nScanning LoRA: {lora_name}")
            files = []
            for ext in EXTENSIONS:
                files.extend(glob.glob(os.path.join(folder, ext)))
                
            for f in tqdm(files, desc=lora_name):
                data = self.analyze_image(f)
                filename = os.path.basename(f)
                
                if data:
                    # For LoRA candidates we use the full reference centroid
                    sim = float(np.dot(data['embedding'], centroid_full))
                    
                    dataset_data.append({
                        'Type': 'Generated', 'Group': lora_name,
                        'Filename': filename,
                        'Similarity': sim,
                        'Age': data['age'],
                        'Yaw': data['yaw'], 'Pitch': data['pitch'], 'Roll': data['roll'],
                        'DetScore': data['det_score'],
                        'AspectRatio': data['aspect_ratio'],
                        'Embedding': data['embedding']
                    })
                else:
                    dataset_data.append({
                        'Type': 'Failed', 'Group': lora_name, 'Filename': filename,
                        'Similarity': 0, 'Age': 0,
                        'Yaw': 0, 'Pitch': 0, 'Roll': 0, 'DetScore': 0, 'AspectRatio': 0, 'Embedding': None
                    })

        return pd.DataFrame(dataset_data)

    def generate_tsne(self, df):
        print("\n>>> Computing t-SNE...")
        valid_df = df[df['Embedding'].notnull()].copy()
        if len(valid_df) < 5: return df
        
        matrix = np.stack(valid_df['Embedding'].values)
        perplex = min(30, len(valid_df) - 1)
        tsne = TSNE(n_components=2, perplexity=perplex, random_state=42, init='pca', learning_rate='auto')
        projections = tsne.fit_transform(matrix)
        valid_df['tsne_x'] = projections[:, 0]
        valid_df['tsne_y'] = projections[:, 1]
        return valid_df

    def create_dashboard(self, df):
        print(f"\n>>> Generating Dashboard (IMAX Layout): {OUTPUT_HTML}")
        clean_df = df[df['Type'] != 'Failed']
        groups = df['Group'].unique()
        colors = px.colors.qualitative.Bold 

        fig = make_subplots(
            rows=7, cols=1,
            subplot_titles=(
                "1. Face Similarity", "2. Age Distribution", "3. Face Ratio", 
                "4. Detection Confidence (All points visible)", 
                "5. Profile Stability", "6. Pose Variety", "7. Identity Map"
            ),
            specs=[[{"type": "xy"}]] * 7,
            # Reduced spacing between charts (was 0.06)
            vertical_spacing=0.025 
        )

        def get_style(grp_name):
            idx = list(groups).index(grp_name)
            return colors[idx % len(colors)]

        for group in groups:
            subset = clean_df[clean_df['Group'] == group]
            color = get_style(group)
            
            # ROW 1: Similarity
            fig.add_trace(go.Box(
                y=subset['Similarity'], name=group, 
                boxpoints='all', jitter=0.5, pointpos=-1.8,
                marker=dict(color=color, size=6), line_color=color,
                text=subset['Filename'], hovertemplate="<b>%{text}</b><br>Sim: %{y:.3f}",
                legendgroup=group, showlegend=False 
            ), row=1, col=1)

            # ROW 2: Age
            fig.add_trace(go.Violin(
                y=subset['Age'], name=group, 
                box_visible=True, meanline_visible=True, points='all',
                line_color=color, opacity=0.8, marker=dict(size=5),
                text=subset['Filename'], hovertemplate="<b>%{text}</b><br>Age: %{y:.1f}",
                legendgroup=group, showlegend=False
            ), row=2, col=1)

            # ROW 3: Ratio
            fig.add_trace(go.Violin(
                y=subset['AspectRatio'], name=group, 
                box_visible=True, points='all', line_color=color, opacity=0.8,
                text=subset['Filename'], hovertemplate="<b>%{text}</b><br>Ratio: %{y:.2f}",
                legendgroup=group, showlegend=False
            ), row=3, col=1)

            # ROW 4: Confidence
            fig.add_trace(go.Box(
                x=subset['DetScore'], name=group, 
                boxpoints='all', jitter=0.5, pointpos=-1.8,
                marker=dict(color=color, size=5), line_color=color,
                text=subset['Filename'], hovertemplate="<b>%{text}</b><br>Conf: %{x:.3f}", 
                orientation='h', legendgroup=group, showlegend=False
            ), row=4, col=1)

            # ROW 5: Profile
            fig.add_trace(go.Scatter(
                x=subset['Yaw'].abs(), y=subset['Similarity'], mode='markers', name=group,
                marker=dict(size=9, color=color), # Slightly larger points
                text=subset['Filename'], hovertemplate="<b>%{text}</b><br>Yaw: %{x:.1f}°<br>Sim: %{y:.3f}",
                legendgroup=group, showlegend=False
            ), row=5, col=1)

            # --- ROW 6: Pose Variety (Bubble Chart 2.0) ---
            # Bubble size indicates whether identity holds under extreme poses
            size_ref_6 = 6 + (subset['Similarity'] ** 4) * 25 # Amplified scaling
            
            fig.add_trace(go.Scatter(
                x=subset['Yaw'], y=subset['Pitch'], mode='markers', name=group,
                marker=dict(
                    size=size_ref_6, # <--- DYNAMIC SIZE
                    color=color, 
                    symbol='circle', # Circle renders size differences better than diamond
                    opacity=0.8,     # Slight transparency to reveal overlaps
                    line=dict(width=1, color='rgba(255,255,255,0.3)')
                ),
                text=subset['Filename'], 
                hovertemplate="<b>%{text}</b><br>Yaw: %{x:.1f}<br>Pitch: %{y:.1f}<br>Sim: %{marker.size:.2f}", # Debug hack to inspect size
                legendgroup=group, showlegend=False
            ), row=6, col=1)

            # ROW 7: t-SNE
            if 'tsne_x' in subset.columns:
                fig.add_trace(go.Scatter(
                    x=subset['tsne_x'], y=subset['tsne_y'], mode='markers', name=group,
                    marker=dict(size=12, color=color, line=dict(width=1, color='white')),
                    text=subset['Filename'], hovertemplate="<b>%{text}</b>",
                    legendgroup=group, showlegend=False
                ), row=7, col=1)

        fig.add_hline(y=0.6, line_dash="dash", line_color="#58a6ff", row=1, col=1)

        fig.update_layout(
            title_text=f"🔬 Biometric Analysis (IMAX) - {TIMESTAMP}",
            # Increased total height
            height=6000, 
            autosize=True,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            # More readable font for large charts
            font=dict(color="#e6edf3", size=15),
            showlegend=False, 
            # Optimized margins
            margin=dict(r=50, l=60, t=80, b=50) 
        )
        
        grid_clr = "#30363d"
        fig.update_xaxes(gridcolor=grid_clr, zerolinecolor=grid_clr)
        fig.update_yaxes(gridcolor=grid_clr, zerolinecolor=grid_clr)
        
        fig.update_yaxes(title_text="Sim", row=1, col=1)
        fig.update_yaxes(title_text="Age", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=1)
        fig.update_xaxes(title_text="Confidence", row=4, col=1)
        fig.update_xaxes(title_text="Yaw", row=5, col=1)
        fig.update_yaxes(title_text="Sim", row=5, col=1)
        fig.update_xaxes(title_text="Yaw", row=6, col=1)
        fig.update_yaxes(title_text="Pitch", row=6, col=1)
        fig.update_xaxes(title_text="Latent 1", row=7, col=1)
        fig.update_yaxes(title_text="Latent 2", row=7, col=1)

        fig.write_html(OUTPUT_HTML)
        
        self.inject_custom_css(OUTPUT_HTML)
        self.inject_floating_legend(OUTPUT_HTML, clean_df) 
        
        print(">>> Dashboard complete.")
        clean_df.drop(columns=['Embedding', 'img_obj'], errors='ignore').to_csv(OUTPUT_CSV, index=False)

    def inject_custom_css(self, html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        custom_style = """
        <style>
            html, body { margin: 0; padding: 0; background-color: #0d1117; overflow-x: hidden; }
            .main-svg { border-radius: 0px !important; }
        </style>
        """
        content = content.replace("</head>", f"{custom_style}\n</head>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

    def inject_floating_legend(self, html_path, df):
        """Create the floating control panel with JS"""
        groups = df['Group'].unique().tolist()
        colors = px.colors.qualitative.Bold
        
        # Generate HTML for each legend entry
        legend_items = ""
        for i, group in enumerate(groups):
            color = colors[i % len(colors)]
            safe_name = group.replace("'", "\\'") # Escape for JS
            
            legend_items += f"""
            <label class="lg-item" data-group="{safe_name}">
                <input type="checkbox" checked onchange="toggleGroup('{safe_name}', this.checked)">
                <span class="dot" style="background:{color}"></span>
                <span class="lbl">{group}</span>
            </label>
            """

        # The payload includes CSS, panel HTML, and the JS script
        payload = f"""
        <style>
            #fl-legend {{
                position: fixed; top: 80px; right: 20px; z-index: 9999;
                width: 220px; max-height: 80vh; overflow-y: auto;
                background: rgba(22, 27, 34, 0.95);
                border: 1px solid #30363d; border-radius: 8px;
                padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                backdrop-filter: blur(5px); font-family: sans-serif;
            }}
            .lg-head {{
                display: flex; justify-content: space-between; align-items: center;
                border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 10px;
            }}
            .lg-head span {{ color: #e6edf3; font-weight: bold; font-size: 14px; }}
            .lg-item {{
                display: flex; align-items: center; gap: 10px; padding: 6px;
                cursor: pointer; border-radius: 4px; transition: background 0.2s;
            }}
            .lg-item:hover {{ background: rgba(255,255,255,0.05); }}
            .lg-item input {{ display: none; }} /* Hide default checkbox */
            .dot {{
                width: 12px; height: 12px; border-radius: 50%; display: inline-block;
                box-shadow: 0 0 5px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.2);
            }}
            .lbl {{ color: #c9d1d9; font-size: 13px; }}
            
            /* Style when unchecked */
            .lg-item input:not(:checked) ~ .dot {{ opacity: 0.3; background: #333 !important; }}
            .lg-item input:not(:checked) ~ .lbl {{ opacity: 0.5; text-decoration: line-through; }}

            .ctrls {{
                display: flex; gap: 5px; margin-top: 15px; pt: 10px; border-top: 1px solid #30363d;
            }}
            .ctrls button {{
                flex: 1; background: #21262d; border: 1px solid #30363d; color: #c9d1d9;
                padding: 5px; border-radius: 4px; cursor: pointer; font-size: 11px;
            }}
            .ctrls button:hover {{ background: #30363d; color: white; }}
        </style>

        <div id="fl-legend">
            <div class="lg-head">
                <span>📊 Control Panel</span>
                <button onclick="document.getElementById('lg-body').style.display = document.getElementById('lg-body').style.display==='none'?'block':'none'" style="background:none;border:none;color:#8b949e;cursor:pointer;">_</button>
            </div>
            <div id="lg-body">
                {legend_items}
                <div class="ctrls">
                    <button onclick="setAll(true)">All ON</button>
                    <button onclick="setAll(false)">All OFF</button>
                    <button onclick="soloMode()">Solo ⟳</button>
                </div>
            </div>
        </div>

        <script>
            // Map group names to Plotly trace indices
            var groupMap = {{}};
            var plot = document.querySelector('.js-plotly-plot');
            
            // Wait for Plotly to load
            if(plot && plot.data) {{
                plot.data.forEach(function(trace, i) {{
                    var g = trace.legendgroup; 
                    if(g) {{
                        if(!groupMap[g]) groupMap[g] = [];
                        groupMap[g].push(i);
                    }}
                }});
            }}

            window.toggleGroup = function(name, isVisible) {{
                var indices = groupMap[name];
                if(indices) {{
                    Plotly.restyle(plot, {{visible: isVisible}}, indices);
                }}
            }};

            window.setAll = function(state) {{
                document.querySelectorAll('.lg-item input').forEach(cb => {{
                    cb.checked = state;
                    toggleGroup(cb.parentElement.getAttribute('data-group'), state);
                }});
            }};

            var soloIndex = -1;
            window.soloMode = function() {{
                var inputs = document.querySelectorAll('.lg-item input');
                soloIndex = (soloIndex + 1) % inputs.length;
                
                // Turn everything off
                setAll(false);
                
                // Turn on only the target
                inputs[soloIndex].checked = true;
                var group = inputs[soloIndex].parentElement.getAttribute('data-group');
                toggleGroup(group, true);
            }};
        </script>
        """
        
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("</body>", payload + "\n</body>")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    try:
        analyzer = FaceAnalyzer()
        df = analyzer.process_dataset()
        df_rich = analyzer.generate_tsne(df)
        analyzer.create_dashboard(df_rich)
        print(f"\n✅ DONE! Open the file: {OUTPUT_HTML}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()