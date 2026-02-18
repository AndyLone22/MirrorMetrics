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
OUTPUT_COPYCAT = f"CopycatReport_{TIMESTAMP}.html"

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

        # Rescue Protocol: if no face found, pad the image to handle close-ups
        # where the face fills the entire frame (InsightFace struggles with these)
        if len(faces) == 0:
            h, w = img.shape[:2]
            pad = int(max(h, w) * 0.4)  # 40% padding on each side
            padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = self.app.get(padded)
            if len(faces) > 0:
                print(f"\n  🔄 Rescued via padding: {os.path.basename(img_path)}")
                # Adjust bounding box back to original image coordinates
                for face in faces:
                    face.bbox[0] -= pad
                    face.bbox[1] -= pad
                    face.bbox[2] -= pad
                    face.bbox[3] -= pad

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
                filename = os.path.basename(f)
                try:
                    data = self.analyze_image(f)
                    
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
                        print(f"\n  ⚠ No face detected: {filename}")
                        dataset_data.append({
                            'Type': 'Failed', 'Group': lora_name, 'Filename': filename,
                            'Similarity': 0, 'Age': 0,
                            'Yaw': 0, 'Pitch': 0, 'Roll': 0, 'DetScore': 0, 'AspectRatio': 0, 'Embedding': None
                        })
                except Exception as ex:
                    print(f"\n  ❌ Error processing {filename}: {ex}")
                    dataset_data.append({
                        'Type': 'Failed', 'Group': lora_name, 'Filename': filename,
                        'Similarity': 0, 'Age': 0,
                        'Yaw': 0, 'Pitch': 0, 'Roll': 0, 'DetScore': 0, 'AspectRatio': 0, 'Embedding': None
                    })

        return pd.DataFrame(dataset_data)

    def generate_tsne(self, df):
        print("\n>>> Computing t-SNE...")
        valid_mask = df['Embedding'].notnull()
        valid_df = df[valid_mask].copy()
        if len(valid_df) < 5: return df
        
        matrix = np.stack(valid_df['Embedding'].values)
        perplex = min(30, len(valid_df) - 1)
        tsne = TSNE(n_components=3, perplexity=perplex, random_state=42, init='pca', learning_rate='auto')
        projections = tsne.fit_transform(matrix)
        # Merge t-SNE coordinates back into the full DataFrame (Failed rows get NaN)
        result = df.copy()
        result.loc[valid_mask, 'tsne_x'] = projections[:, 0]
        result.loc[valid_mask, 'tsne_y'] = projections[:, 1]
        result.loc[valid_mask, 'tsne_z'] = projections[:, 2]
        return result

    def create_dashboard(self, df):
        print(f"\n>>> Generating Dashboard (IMAX Layout): {OUTPUT_HTML}")
        clean_df = df[df['Type'] != 'Failed']
        groups = df['Group'].unique()
        colors = px.colors.qualitative.Bold 

        fig = make_subplots(
            rows=6, cols=1,
            subplot_titles=(
                "1. Face Similarity", "2. Age Distribution", "3. Face Ratio", 
                "4. Detection Confidence (All points visible)", 
                "5. Profile Stability", "6. Pose Variety"
            ),
            specs=[[{"type": "xy"}]] * 6,
            # Reduced spacing between charts (was 0.06)
            vertical_spacing=0.03 
        )

        # Separate 3D figure for t-SNE Identity Map
        fig3d = go.Figure()

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

            # t-SNE 3D (separate figure)
            if 'tsne_x' in subset.columns:
                fig3d.add_trace(go.Scatter3d(
                    x=subset['tsne_x'], y=subset['tsne_y'], z=subset['tsne_z'],
                    mode='markers', name=group,
                    marker=dict(size=5, color=color, line=dict(width=0.5, color='white')),
                    text=subset['Filename'], hovertemplate="<b>%{text}</b>",
                    legendgroup=group, showlegend=True
                ))

        fig.add_hline(y=0.6, line_dash="dash", line_color="#58a6ff", row=1, col=1)

        fig.update_layout(
            title_text=f"🔬 Biometric Analysis (IMAX) - {TIMESTAMP}",
            # Increased total height
            height=5200, 
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

        # Configure the 3D t-SNE figure
        fig3d.update_layout(
            title_text="7. Identity Map (3D t-SNE)",
            height=700,
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3", size=15),
            margin=dict(r=50, l=50, t=60, b=50),
            scene=dict(
                xaxis_title="Latent 1",
                yaxis_title="Latent 2",
                zaxis_title="Latent 3",
                bgcolor="#161b22",
                xaxis=dict(gridcolor=grid_clr, zerolinecolor=grid_clr, color="#e6edf3"),
                yaxis=dict(gridcolor=grid_clr, zerolinecolor=grid_clr, color="#e6edf3"),
                zaxis=dict(gridcolor=grid_clr, zerolinecolor=grid_clr, color="#e6edf3"),
            ),
            legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor=grid_clr, borderwidth=1),
        )

        # Write main dashboard
        fig.write_html(OUTPUT_HTML)
        
        # Inject the 3D t-SNE chart into the same HTML file
        tsne_3d_html = fig3d.to_html(full_html=False, include_plotlyjs=False)
        with open(OUTPUT_HTML, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace("</body>", f'\n<div style="max-width:100%;padding:0 50px;">{tsne_3d_html}</div>\n</body>')
        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.inject_custom_css(OUTPUT_HTML)
        self.inject_floating_legend(OUTPUT_HTML, clean_df) 
        
        print(">>> Dashboard complete.")
        df.drop(columns=['Embedding', 'img_obj'], errors='ignore').to_csv(OUTPUT_CSV, index=False)

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

    def create_copycat_report(self, df):
        """Generate a standalone HTML report pairing each Generated image
        with its nearest Reference neighbor (by cosine similarity)."""
        import base64

        print(f"\n>>> Generating Copycat Report: {OUTPUT_COPYCAT}")

        ref_rows = df[(df['Type'] == 'Reference') & df['Embedding'].notnull()].copy()
        gen_rows = df[(df['Type'] == 'Generated') & df['Embedding'].notnull()].copy()
        failed_rows = df[df['Type'] == 'Failed'].copy()

        if ref_rows.empty or gen_rows.empty:
            print("WARNING: Not enough data for Copycat Report (need both Reference and Generated images).")
            return

        ref_embeddings = np.stack(ref_rows['Embedding'].values)
        ref_files = ref_rows['Filename'].values.tolist()
        # Build full paths for reference images
        ref_paths = [os.path.join(PATH_REFERENCE, fn) for fn in ref_files]

        # --- Compute nearest neighbor for every generated image ---
        pairs = []  # (gen_filename, gen_group, gen_path, ref_filename, ref_path, similarity)
        failed_list = []  # images where face detection failed
        hit_counter = {fn: 0 for fn in ref_files}  # count how often each ref is the NN

        for _, row in gen_rows.iterrows():
            gen_emb = row['Embedding']
            sims = np.dot(ref_embeddings, gen_emb)  # cosine sim (embeddings are normed)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_ref_fn = ref_files[best_idx]
            best_ref_path = ref_paths[best_idx]
            hit_counter[best_ref_fn] += 1

            # Resolve gen full path
            gen_path = os.path.join(PATH_CANDIDATES_ROOT, row['Group'], row['Filename'])

            pairs.append({
                'gen_fn': row['Filename'],
                'gen_group': row['Group'],
                'gen_path': gen_path,
                'ref_fn': best_ref_fn,
                'ref_path': best_ref_path,
                'sim': best_sim,
            })

        # Collect failed images
        for _, row in failed_rows.iterrows():
            gen_path = os.path.join(PATH_CANDIDATES_ROOT, row['Group'], row['Filename'])
            failed_list.append({
                'gen_fn': row['Filename'],
                'gen_group': row['Group'],
                'gen_path': gen_path,
            })

        # Sort pairs by similarity descending (most suspicious first)
        pairs.sort(key=lambda p: p['sim'], reverse=True)

        # --- Helper: image -> base64 data URI (resized thumbnail) ---
        def img_to_b64(path, max_size=256):
            img = cv2.imread(path)
            if img is None:
                return ""
            h, w = img.shape[:2]
            scale = max_size / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode('ascii')

        # --- Helper: similarity -> color ---
        def sim_color(s):
            if s > 0.85:
                return "#f85149"  # red
            elif s > 0.70:
                return "#d29922"  # orange
            elif s > 0.50:
                return "#3fb950"  # green
            return "#8b949e"      # grey — low similarity

        def sim_label(s):
            if s > 0.85:
                return f'{s:.3f} <span style="color:#f85149;font-weight:700">⚠ Danger Zone!</span>'
            elif s > 0.70:
                return f'{s:.3f} <span style="color:#d29922">⚡ Watch Out</span>'
            elif s > 0.50:
                return f'{s:.3f} <span style="color:#3fb950">✓ OK</span>'
            return f'{s:.3f} <span style="color:#8b949e">↓ LOW sim</span>'

        # --- Summary stats ---
        sims_all = [p['sim'] for p in pairs]
        avg_sim = np.mean(sims_all)
        min_sim = np.min(sims_all)
        max_sim = np.max(sims_all)
        n_danger = sum(1 for s in sims_all if s > 0.85)
        n_failed = len(failed_list)

        # --- Build card HTML ---
        cards_html = ""
        for p in tqdm(pairs, desc="Building cards"):
            gen_b64 = img_to_b64(p['gen_path'])
            ref_b64 = img_to_b64(p['ref_path'])
            border_col = sim_color(p['sim'])
            label = sim_label(p['sim'])
            cards_html += f'''
            <div class="cc-card" data-sim="{p['sim']:.4f}" data-group="{p['gen_group']}" style="border-left:4px solid {border_col}">
                <div class="cc-imgs">
                    <div class="cc-side">
                        <div class="cc-tag">Generated</div>
                        <img src="{gen_b64}" alt="gen">
                        <div class="cc-fn">{p['gen_fn']}</div>
                        <div class="cc-grp">{p['gen_group']}</div>
                    </div>
                    <div class="cc-arrow">⟶</div>
                    <div class="cc-side">
                        <div class="cc-tag ref">Nearest Ref</div>
                        <img src="{ref_b64}" alt="ref">
                        <div class="cc-fn">{p['ref_fn']}</div>
                    </div>
                </div>
                <div class="cc-sim">Similarity: {label}</div>
            </div>'''

        # --- Build failed cards ---
        failed_html = ""
        for fl in failed_list:
            gen_b64 = img_to_b64(fl['gen_path'])
            failed_html += f'''
            <div class="cc-card cc-failed" style="border-left:4px solid #484f58">
                <div class="cc-imgs">
                    <div class="cc-side">
                        <div class="cc-tag">Generated</div>
                        <img src="{gen_b64}" alt="gen">
                        <div class="cc-fn">{fl['gen_fn']}</div>
                        <div class="cc-grp">{fl['gen_group']}</div>
                    </div>
                    <div class="cc-arrow" style="color:#484f58">✘</div>
                    <div class="cc-side">
                        <div class="cc-tag" style="color:#484f58">No Match</div>
                        <div style="width:200px;height:200px;border-radius:8px;border:2px dashed #30363d;display:flex;align-items:center;justify-content:center;color:#484f58;font-size:40px;">❓</div>
                    </div>
                </div>
                <div class="cc-sim" style="color:#f0883e">⚠ Failed to recognize face</div>
            </div>'''

        # --- Black Hole ranking (all refs, sorted by hits desc) ---
        ranked_refs = sorted(hit_counter.items(), key=lambda x: x[1], reverse=True)
        bh_html = ""
        for rank, (ref_fn, hits) in tqdm(enumerate(ranked_refs, 1), total=len(ranked_refs), desc="Black Hole ranking"):
            ref_path = os.path.join(PATH_REFERENCE, ref_fn)
            ref_b64 = img_to_b64(ref_path, max_size=128)
            bar_w = min(hits / max(1, ranked_refs[0][1]) * 100, 100)
            danger_cls = ' bh-danger' if hits >= 3 else ''
            bh_html += f'''
            <div class="bh-row{danger_cls}">
                <span class="bh-rank">#{rank}</span>
                <img src="{ref_b64}" class="bh-thumb" alt="ref">
                <div class="bh-info">
                    <div class="bh-fn">{ref_fn}</div>
                    <div class="bh-bar-bg"><div class="bh-bar" style="width:{bar_w}%"></div></div>
                </div>
                <span class="bh-count">{hits}</span>
            </div>'''

        # --- Assemble full HTML ---
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Copycat Detector — {TIMESTAMP}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin:0; padding:0; background:#0d1117; color:#e6edf3; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:30px 20px; }}
  h1 {{ font-size:28px; margin-bottom:5px; }}
  h1 span {{ font-size:16px; color:#8b949e; font-weight:400; }}
  .stats {{ display:flex; gap:20px; flex-wrap:wrap; margin:20px 0 30px; }}
  .stat {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 20px; min-width:140px; }}
  .stat .val {{ font-size:26px; font-weight:700; }}
  .stat .lbl {{ font-size:12px; color:#8b949e; margin-top:4px; }}
  .controls {{ display:flex; gap:12px; margin-bottom:24px; flex-wrap:wrap; align-items:center; }}
  .controls select, .controls button {{ background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:8px 14px; border-radius:6px; cursor:pointer; font-size:13px; }}
  .controls select:hover, .controls button:hover {{ background:#30363d; color:#fff; }}
  .grid {{ display:flex; flex-direction:column; gap:14px; }}
  .cc-card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:16px; transition: transform .15s, box-shadow .15s; }}
  .cc-card:hover {{ transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.4); }}
  .cc-imgs {{ display:flex; align-items:center; gap:16px; justify-content:center; flex-wrap:wrap; }}
  .cc-side {{ text-align:center; }}
  .cc-side img {{ max-width:200px; max-height:260px; object-fit:contain; border-radius:8px; border:2px solid #30363d; }}
  .cc-tag {{ font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#8b949e; margin-bottom:6px; }}
  .cc-tag.ref {{ color:#58a6ff; }}
  .cc-fn {{ font-size:12px; color:#8b949e; margin-top:6px; word-break:break-all; max-width:200px; }}
  .cc-grp {{ font-size:11px; color:#58a6ff; margin-top:2px; }}
  .cc-arrow {{ font-size:32px; color:#484f58; }}
  .cc-sim {{ text-align:center; margin-top:12px; font-size:15px; }}
  /* Black Hole section */
  h2 {{ margin-top:50px; font-size:22px; border-bottom:1px solid #30363d; padding-bottom:10px; }}
  .bh-row {{ display:flex; align-items:center; gap:14px; padding:10px 12px; border-radius:8px; transition:background .15s; }}
  .bh-row:hover {{ background:#161b22; }}
  .bh-danger {{ background:rgba(248,81,73,.08); }}
  .bh-rank {{ font-size:16px; font-weight:700; color:#484f58; min-width:36px; text-align:right; }}
  .bh-thumb {{ width:64px; height:64px; object-fit:cover; border-radius:6px; border:2px solid #30363d; }}
  .bh-info {{ flex:1; }}
  .bh-fn {{ font-size:13px; margin-bottom:4px; }}
  .bh-bar-bg {{ height:8px; background:#21262d; border-radius:4px; overflow:hidden; }}
  .bh-bar {{ height:100%; background:linear-gradient(90deg,#3fb950,#d29922,#f85149); border-radius:4px; transition:width .3s; }}
  .bh-count {{ font-size:22px; font-weight:700; min-width:40px; text-align:center; }}
  .bh-danger .bh-count {{ color:#f85149; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>👯 Copycat Detector <span>— {TIMESTAMP}</span></h1>
  <div class="stats">
    <div class="stat"><div class="val">{len(pairs)}</div><div class="lbl">Total Pairs</div></div>
    <div class="stat"><div class="val" style="color:{sim_color(avg_sim)}">{avg_sim:.3f}</div><div class="lbl">Avg Similarity</div></div>
    <div class="stat"><div class="val">{max_sim:.3f}</div><div class="lbl">Max Similarity</div></div>
    <div class="stat"><div class="val">{min_sim:.3f}</div><div class="lbl">Min Similarity</div></div>
    <div class="stat"><div class="val" style="color:#f85149">{n_danger}</div><div class="lbl">Danger Zone (>0.85)</div></div>
    <div class="stat"><div class="val" style="color:#f0883e">{n_failed}</div><div class="lbl">Failed Detection</div></div>
  </div>

  <div class="controls">
    <select id="sortSel" onchange="applySort()">
      <option value="sim-desc">Sort: Similarity ↓</option>
      <option value="sim-asc">Sort: Similarity ↑</option>
      <option value="group">Sort: Group A-Z</option>
    </select>
    <select id="groupFilter" onchange="applyFilter()">
      <option value="all">All Groups</option>
    </select>
    <button onclick="document.getElementById('sortSel').value='sim-desc';document.getElementById('groupFilter').value='all';applySort();applyFilter();">Reset</button>
  </div>

  <div class="grid" id="cardGrid">
    {cards_html}
  </div>

  {failed_html}

  <h2>🕳️ Black Hole Ranking</h2>
  <p style="color:#8b949e;font-size:14px;">Which reference images attract the most generated outputs? Sorted most → least.</p>
  {bh_html}
</div>

<script>
(function(){{
  // Populate group filter
  var groups = new Set();
  document.querySelectorAll('.cc-card').forEach(function(c){{ groups.add(c.dataset.group); }});
  var sel = document.getElementById('groupFilter');
  groups.forEach(function(g){{
    var o = document.createElement('option'); o.value = g; o.textContent = g; sel.appendChild(o);
  }});
}})();

function applySort(){{
  var grid = document.getElementById('cardGrid');
  var cards = Array.from(grid.children);
  var mode = document.getElementById('sortSel').value;
  cards.sort(function(a,b){{
    if(mode==='sim-desc') return parseFloat(b.dataset.sim) - parseFloat(a.dataset.sim);
    if(mode==='sim-asc') return parseFloat(a.dataset.sim) - parseFloat(b.dataset.sim);
    return a.dataset.group.localeCompare(b.dataset.group);
  }});
  cards.forEach(function(c){{ grid.appendChild(c); }});
}}

function applyFilter(){{
  var g = document.getElementById('groupFilter').value;
  document.querySelectorAll('.cc-card').forEach(function(c){{
    c.style.display = (g==='all' || c.dataset.group===g) ? '' : 'none';
  }});
}}
</script>
</body>
</html>'''

        with open(OUTPUT_COPYCAT, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f">>> Copycat Report complete — {len(pairs)} pairs analyzed.")

if __name__ == "__main__":
    try:
        analyzer = FaceAnalyzer()
        df = analyzer.process_dataset()
        df_rich = analyzer.generate_tsne(df)
        analyzer.create_dashboard(df_rich)
        analyzer.create_copycat_report(df_rich)
        print(f"\n✅ DONE! Open the files:")
        print(f"   📊 Dashboard:  {OUTPUT_HTML}")
        print(f"   👯 Copycat:    {OUTPUT_COPYCAT}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()