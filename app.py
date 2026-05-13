import streamlit as st
import numpy as np
import cv2
import joblib
import json
import time
import os
from pathlib import Path
from PIL import Image, ImageEnhance
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Garbage Classifier",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #2ecc71, #3498db);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center; color: #888; font-size: 1rem; margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e; border-radius: 12px; padding: 1rem 1.5rem;
        border-left: 4px solid #2ecc71; margin-bottom: 0.8rem;
    }
    .pred-label {
        font-size: 1.6rem; font-weight: 700; color: #2ecc71;
    }
    .conf-text {
        font-size: 1rem; color: #aaa;
    }
    .warning-box {
        background: #2d1b00; border-left: 4px solid #f39c12;
        border-radius: 8px; padding: 0.8rem 1.2rem; color: #f39c12;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #0d2137; border-left: 4px solid #3498db;
        border-radius: 8px; padding: 0.8rem 1.2rem; color: #7fb3d3;
        margin: 0.5rem 0;
    }
    .stProgress > div > div { background-color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODELS & CLASSES
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "SVM (RBF) — Best": "model_svm_best.pkl",
        "SVM (RBF)":        "model_svm_rbf.pkl",
        "SVM (Linear)":     "model_svm_linear.pkl",
        "Random Forest":    "model_rf.pkl",
        "KNN (PCA-50)":     "model_knn.pkl",
    }
    for name, fname in model_files.items():
        if os.path.exists(fname):
            try:
                models[name] = joblib.load(fname)
            except Exception as e:
                st.warning(f"Could not load {fname}: {e}")

    le = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None

    classes = []
    if os.path.exists("classes.json"):
        with open("classes.json") as f:
            classes = json.load(f)

    results = {}
    if os.path.exists("model_results.csv"):
        import pandas as pd
        results = pd.read_csv("model_results.csv", index_col=0)

    return models, le, classes, results


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (same as training)
# ─────────────────────────────────────────────
def extract_features(img_rgb, size=(64, 64)):
    """Extract the same ~120-dim feature vector used during training."""
    img = cv2.resize(img_rgb, size)
    img = img.astype(np.uint8)

    # RGB stats
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    rgb = np.array([r.mean(),g.mean(),b.mean(),r.std(),g.std(),b.std()])

    # HSV stats + histograms
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hh, ss, vv = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    hsv_stats = np.array([hh.mean(),ss.mean(),vv.mean(),hh.std(),ss.std(),vv.std()])
    h_hist = np.histogram(hh, bins=32, range=(0,180), density=True)[0]
    s_hist = np.histogram(ss, bins=8,  range=(0,256), density=True)[0]
    v_hist = np.histogram(vv, bins=8,  range=(0,256), density=True)[0]

    # LBP texture
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp  = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist = np.histogram(lbp, bins=26, range=(0,26), density=True)[0]

    # HOG texture
    hog_feats = hog(gray, orientations=8, pixels_per_cell=(16,16),
                    cells_per_block=(1,1), feature_vector=True)

    # Hu moments
    moments = cv2.moments(gray)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return np.concatenate([rgb, hsv_stats, h_hist, s_hist, v_hist,
                           lbp_hist, hog_feats, hu]).reshape(1, -1)


def predict_image(img_rgb, model, le, classes):
    """Run prediction and return (label, confidence, all_probs)."""
    feat = extract_features(img_rgb)
    probs = model.predict_proba(feat)[0]
    pred_id = np.argmax(probs)
    label = le.inverse_transform([pred_id])[0]
    conf  = probs[pred_id]
    return label, conf, probs


# ─────────────────────────────────────────────
#  CONFIDENCE BAR CHART
# ─────────────────────────────────────────────
def plot_confidence(probs, classes, top_n=6):
    top_idx  = np.argsort(probs)[::-1][:top_n]
    top_probs = probs[top_idx]
    top_cls   = [classes[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    bar_colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(top_n)]
    bars = ax.barh(top_cls[::-1], top_probs[::-1], color=bar_colors[::-1],
                   edgecolor="none", height=0.55)

    for bar, val in zip(bars, top_probs[::-1]):
        ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height()/2,
                f"{val*100:.1f}%", va="center", color="white", fontsize=9)

    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Confidence", color="#aaa")
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color("#aaa")
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  RECYCLING TIP
# ─────────────────────────────────────────────
TIPS = {
    "cardboard"  : "♻️ Flatten boxes before recycling. Remove tape and staples.",
    "glass"      : "🍾 Rinse bottles. Do NOT mix with ceramics or broken glass.",
    "metal"      : "🥫 Rinse cans. Aluminium is 100% recyclable indefinitely.",
    "paper"      : "📄 Keep dry. Wet or greasy paper goes in general waste.",
    "plastic"    : "🧴 Check the number (1-7). Bottles/containers usually OK.",
    "trash"      : "🗑️ General waste. Cannot be recycled — minimise this!",
    "biological" : "🍃 Compost this! Great for garden or food-waste bin.",
    "battery"    : "🔋 HAZARDOUS — take to a designated battery drop-off point.",
    "clothes"    : "👕 Donate if wearable. Textile banks for worn items.",
    "shoes"      : "👟 Donate if wearable. Some brands offer take-back schemes.",
    "white-glass": "🪟 Separate from coloured glass if your area requires it.",
    "green-glass": "🍶 Rinse and place in glass recycling bin.",
}

def get_tip(label):
    for key in TIPS:
        if key in label.lower():
            return TIPS[key]
    return "♻️ Check your local recycling guidelines for this item."


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────
models, le, classes, results_df = load_models()

# ── HEADER ───────────────────────────────────
st.markdown('<div class="main-title">🗑️ Garbage Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered waste classification · 12 categories · scikit-learn SVM</div>',
            unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if not models:
        st.error("No models found! Run the training notebook first.")
        st.stop()

    model_name = st.selectbox("🤖 Choose Model", list(models.keys()))
    selected_model = models[model_name]

    st.divider()
    st.subheader("📊 Model Performance")
    if len(results_df) > 0 and model_name in results_df.index:
        row = results_df.loc[model_name]
        st.metric("Accuracy",    f"{row.get('accuracy',0)*100:.1f}%")
        st.metric("Weighted F1", f"{row.get('f1_weighted',0):.3f}")
        st.metric("Macro F1",    f"{row.get('f1_macro',0):.3f}")
    else:
        st.info("Run the full notebook to see metrics here.")

    st.divider()
    st.subheader("🏷️ Classes")
    for i, c in enumerate(classes):
        st.caption(f"• {c}")

# ── TABS ─────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Upload Image",
    "🎥 Webcam (Real-time)",
    "🎬 Video File",
    "📈 Model Comparison"
])


# ════════════════════════════════════════════
#  TAB 1 — UPLOAD IMAGE
# ════════════════════════════════════════════
with tab1:
    st.subheader("Upload an image to classify")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png","webp","bmp"])

    if uploaded:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image")

        with col2:
            with st.spinner("Classifying..."):
                img_np = np.array(img)
                label, conf, probs = predict_image(img_np, selected_model, le, classes)

            st.markdown(f'<div class="metric-card">                <div class="conf-text">Predicted class</div>                <div class="pred-label">{label.upper()}</div>                <div class="conf-text">Confidence: {conf*100:.1f}%</div>                </div>', unsafe_allow_html=True)

            st.progress(float(conf))
            tip = get_tip(label)
            st.markdown(f'<div class="info-box">{tip}</div>', unsafe_allow_html=True)

            st.subheader("Top-6 Predictions")
            fig = plot_confidence(probs, classes)
            st.pyplot(fig)
            plt.close()

        # All model comparison on same image
        if st.checkbox("Compare all models on this image"):
            st.subheader("All Models — Prediction Comparison")
            rows = []
            for mname, mmodel in models.items():
                lbl, cf, _ = predict_image(img_np, mmodel, le, classes)
                rows.append({"Model": mname, "Prediction": lbl,
                              "Confidence": f"{cf*100:.1f}%"})
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# ════════════════════════════════════════════
#  TAB 2 — WEBCAM REAL-TIME
# ════════════════════════════════════════════
with tab2:
    st.subheader("Real-time Webcam Classification")
    st.markdown('<div class="info-box">📸 Uses your device camera to classify garbage in real time. '
                'Each frame is processed by the selected model.</div>', unsafe_allow_html=True)

    col_cam1, col_cam2 = st.columns([1.2, 1], gap="large")

    with col_cam1:
        run_webcam = st.toggle("▶️ Start Webcam", key="webcam_toggle")
        FRAME_WINDOW = st.empty()
        status_box   = st.empty()

    with col_cam2:
        pred_display  = st.empty()
        conf_display  = st.empty()
        chart_display = st.empty()
        tip_display   = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot open webcam. Make sure your camera is connected and not in use by another app.")
        else:
            frame_count = 0
            PREDICT_EVERY = 5  # predict every N frames for speed

            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    status_box.warning("⚠️ Lost webcam feed.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw crosshair guide
                h, w = frame_rgb.shape[:2]
                cx, cy = w//2, h//2
                size = min(w,h)//4
                cv2.rectangle(frame_rgb, (cx-size,cy-size), (cx+size,cy+size),
                              (46,204,113), 2)
                cv2.putText(frame_rgb, "Place item here",
                            (cx-size, cy-size-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (46,204,113), 2)

                FRAME_WINDOW.image(frame_rgb, channels="RGB", use_column_width=True)

                if frame_count % PREDICT_EVERY == 0:
                    # Crop center region for prediction
                    crop = frame_rgb[cy-size:cy+size, cx-size:cx+size]
                    if crop.size > 0:
                        label, conf, probs = predict_image(crop, selected_model, le, classes)
                        tip = get_tip(label)

                        pred_display.markdown(
                            f'<div class="metric-card">                            <div class="conf-text">Detected</div>                            <div class="pred-label">{label.upper()}</div>                            <div class="conf-text">Confidence: {conf*100:.1f}%</div>                            </div>', unsafe_allow_html=True)

                        conf_display.progress(float(conf))
                        tip_display.markdown(
                            f'<div class="info-box">{tip}</div>', unsafe_allow_html=True)

                        fig = plot_confidence(probs, classes)
                        chart_display.pyplot(fig)
                        plt.close()

                frame_count += 1
                time.sleep(0.03)  # ~30 FPS cap

                # Re-check toggle
                run_webcam = st.session_state.get("webcam_toggle", False)

            cap.release()
            status_box.success("✅ Webcam stopped.")
    else:
        FRAME_WINDOW.info("👆 Toggle the switch above to start your webcam.")


# ════════════════════════════════════════════
#  TAB 3 — VIDEO FILE
# ════════════════════════════════════════════
with tab3:
    st.subheader("Classify Frames from a Video File")
    st.markdown('<div class="info-box">Upload a video (.mp4, .mov, .avi). '
                'The app samples frames and classifies each one.</div>',
                unsafe_allow_html=True)

    video_file = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"],
                                   key="video_uploader")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        sample_rate = st.slider("Sample every N frames", 1, 60, 15,
                                help="Lower = more frames analysed (slower)")
        max_frames  = st.slider("Max frames to analyse", 5, 100, 30)

    with col_v2:
        show_frames = st.checkbox("Show frame previews", value=True)

    if video_file and st.button("🎬 Analyse Video", type="primary"):

        # Save uploaded video to temp file
        tmp_path = Path("_tmp_video" + Path(video_file.name).suffix)
        tmp_path.write_bytes(video_file.read())

        cap = cv2.VideoCapture(str(tmp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25
        duration     = total_frames / fps

        st.info(f"📹 Video: {total_frames} frames · {fps:.1f} FPS · {duration:.1f}s duration")

        progress_bar = st.progress(0)
        frame_results = []
        preview_cols  = st.columns(5) if show_frames else []

        frame_idx   = 0
        analysed    = 0
        col_cursor  = 0

        with st.spinner("Analysing video frames..."):
            while cap.isOpened() and analysed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, conf, probs = predict_image(frame_rgb, selected_model, le, classes)
                    ts = frame_idx / fps

                    frame_results.append({
                        "Frame": frame_idx,
                        "Time (s)": round(ts, 2),
                        "Prediction": label,
                        "Confidence": round(conf * 100, 1)
                    })

                    if show_frames and col_cursor < 5:
                        img_pil = Image.fromarray(frame_rgb).resize((160, 120))
                        preview_cols[col_cursor].image(
                            img_pil,
                            caption=f"{label}\n{conf*100:.0f}%"
                        )
                        col_cursor = (col_cursor + 1) % 5

                    analysed += 1
                    progress_bar.progress(min(analysed / max_frames, 1.0))

                frame_idx += 1

        cap.release()
        tmp_path.unlink(missing_ok=True)
        progress_bar.empty()

        if frame_results:
            import pandas as pd
            df_vid = pd.DataFrame(frame_results)
            st.success(f"✅ Analysed {len(df_vid)} frames from the video")

            # Summary
            st.subheader("📊 Results Summary")
            col_r1, col_r2, col_r3 = st.columns(3)
            most_common = df_vid["Prediction"].mode()[0]
            avg_conf    = df_vid["Confidence"].mean()
            col_r1.metric("Most Common Class", most_common)
            col_r2.metric("Average Confidence", f"{avg_conf:.1f}%")
            col_r3.metric("Frames Analysed", len(df_vid))

            # Class frequency chart
            st.subheader("Class Frequency Across Frames")
            freq = df_vid["Prediction"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            fig2.patch.set_facecolor("#1e1e2e")
            ax2.set_facecolor("#1e1e2e")
            ax2.bar(freq.index, freq.values,
                    color=["#2ecc71" if i==0 else "#3498db" for i in range(len(freq))])
            ax2.set_ylabel("Frame count", color="white")
            ax2.tick_params(colors="white", axis="x", rotation=30)
            ax2.tick_params(colors="white", axis="y")
            ax2.spines[:].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Confidence timeline
            st.subheader("Confidence Over Time")
            fig3, ax3 = plt.subplots(figsize=(8, 2.5))
            fig3.patch.set_facecolor("#1e1e2e")
            ax3.set_facecolor("#1e1e2e")
            ax3.plot(df_vid["Time (s)"], df_vid["Confidence"],
                     color="#2ecc71", lw=2, marker="o", markersize=4)
            ax3.axhline(50, color="#e74c3c", ls="--", lw=0.8, label="50% threshold")
            ax3.set_xlabel("Time (s)", color="white")
            ax3.set_ylabel("Confidence (%)", color="white")
            ax3.tick_params(colors="white")
            ax3.spines[:].set_visible(False)
            ax3.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

            # Full results table
            st.subheader("Frame-by-frame Results")
            st.dataframe(df_vid, hide_index=True)

            tip = get_tip(most_common)
            st.markdown(f'<div class="info-box">💡 Recycling tip for <b>{most_common}</b>: {tip}</div>',
                        unsafe_allow_html=True)


# ════════════════════════════════════════════
#  TAB 4 — MODEL COMPARISON
# ════════════════════════════════════════════
with tab4:
    st.subheader("Model Performance Comparison")

    if len(results_df) == 0:
        st.warning("Run the full training notebook first to generate model_results.csv")
    else:
        import pandas as pd

        display_cols = ["accuracy","f1_weighted","f1_macro","precision","recall"]
        disp = results_df[[c for c in display_cols if c in results_df.columns]].copy()
        disp.columns = ["Accuracy","F1 (Weighted)","F1 (Macro)","Precision","Recall"]
        disp = disp.map(lambda x: f"{x*100:.2f}%" if isinstance(x, float) else x)
        st.dataframe(disp)

        # Bar charts
        numeric = results_df[[c for c in display_cols if c in results_df.columns]]
        fig4, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig4.patch.set_facecolor("#1e1e2e")
        colors_bar = ["#2ecc71","#3498db","#f39c12","#e74c3c","#9b59b6"]

        for ax, col, title in zip(axes,
            ["accuracy","f1_weighted","f1_macro"],
            ["Accuracy","Weighted F1","Macro F1"]):
            if col in numeric.columns:
                vals = numeric[col].values
                ax.set_facecolor("#1e1e2e")
                bars = ax.bar(range(len(vals)), vals*100,
                              color=colors_bar[:len(vals)], edgecolor="none")
                ax.set_title(title, color="white", fontweight="bold")
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels([n[:12] for n in numeric.index],
                                   rotation=25, ha="right", fontsize=7, color="white")
                ax.tick_params(axis="y", colors="white")
                ax.spines[:].set_visible(False)
                ax.set_ylim(0, 110)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            f"{val*100:.1f}%", ha="center", fontsize=8, color="white")

        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.divider()
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    | Item | Detail |
    |---|---|
    | Dataset | Garbage Classification v2 (sumn2u) — ~15K images |
    | Classes | 12 (cardboard, glass, metal, paper, plastic, trash, ...) |
    | Features | RGB stats · HSV histograms · LBP texture · HOG · Hu moments (~120 dims) |
    | Models | SVM (RBF & Linear) · Random Forest · KNN+PCA |
    | Tuning | GridSearchCV on SVM (C × gamma grid) |
    | Framework | scikit-learn only — no GPU required |
    """)
