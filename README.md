# Project 2 — Joint Detection of AI-Generated Images and Post-Processing Alterations

> **Course:** Computer Vision · 2025–26  
> **Task:** Given a single input image, simultaneously predict (1) whether it is a real photograph or AI-generated, and (2) which post-processing transformation has been applied to it.

---

## Authors & Institutional Context

This project was developed cooperatively as part of the Erasmus+ Exchange Program at **Sapienza Università di Roma**.

* **Alberto Rivas** — [ Polytechnic university of Oviedo / Uniovi]
* **Carlos Fernández** — [ Polytechnic university of Oviedo / Uniovi]
* **Joaquín Avilés** — [University of Sevilla / US]

---

---

## Repository Structure

```
├── code.ipynb                  # Full pipeline: data → model → training → evaluation
├── Dataset/                    # See link below — too large for Git
├── Project_Presentation/       # Slides
└── README.md
└── Figures                     # Contains the Figures for this readme
```

## Dataset

The project uses the **RRDataset** — a real-world robustness benchmark containing real photographs and AI-generated images across three post-processing splits: **Original**, **Internet-Transmitted**, and **Re-digitized**.

> **Dataset download:** [https://drive.google.com/drive/folders/1CzWAxIhxyBlK4XRAYrQrbexKosUDD1yP?usp=drive_link]

The dataset is not included in this repository due to its size. After downloading, place it at the path defined in the `BASE_DATA_PATH` global (see §2 Globals).

---

## Requirements

```bash
torch torchvision
opencv-python
Pillow
numpy
matplotlib
pandas
scikit-learn
```

All code is designed to run on **Google Colab** with a GPU runtime. Mount your Google Drive before executing any cell.

---

## How to Run

1. **Environment Setup:** Mount Google Drive and verify that `BASE_DATA_PATH` and `MODEL_SAVE_DIR` point correctly to your target directory paths.
2. **Commented Code & Execution Avoidance:** Multiple sections of the notebook are commented out by default because they do not need to be re-run:
   * **Data Pipeline:** Cells managing dataset downloading, balanced downsampling, and partitioning are locked. **It is not necessary to uncomment or run these cells** if you use the pre-processed, structured dataset provided in the **Dataset** section link.
   * **Network Architecture:** The model section contains commented-out code from a preliminary exploratory experiment used to evaluate alternative backbones. **It is not necessary to re-run this exploration**, as the notebook is pre-configured to execute end-to-end using the final selected **ConvNeXt-Tiny** architecture.
3. **Pipeline Execution:** Run remaining active cells sequentially from top to bottom (**Runtime → Run all**).
4. **Hyperparameter Sweeps:** To transition between experiments (e.g., changing baselines or running the ablation sweep), update only the `ALPHA` variable in the Globals section and execute from the training initialization cells onward.

> ⚠️ **State Constraint:** Avoid executing cells out of order. The custom `Dataset` object caches its internal image-to-target path mapping at initialization (`__init__`). If folders or geometric transforms are modified, you must re-run both the Dataset initialization and the downstream `DataLoader` cells to re-collate active memory batches.
---

## Code Walkthrough

### 1. Imports
The import block establishes the dependency surface of the pipeline, with each library serving a distinct structural role:

* **PyTorch Ecosystem (`torch`, `nn`, `optim`, `torchvision`):** Forms the computational core. `torch.nn` structures the shared backbone and dual classification heads. **AdamW** is selected over standard Adam for its decoupled weight decay, which is critical during fine-tuning to prevent $L2$ regularization from interfering with adaptive learning rates. `torchvision` supplies the pre-trained ConvNeXt-Tiny weights and preprocessing transforms.
* **OpenCV (`cv2`) vs. PIL:** PIL handles standard image loading and basic transforms. OpenCV is reserved exclusively for forensic exploratory data analysis (EDA), providing the kernel-level control and floating-point precision required for advanced operations like Laplacian edge maps, Difference of Gaussians (DoG), and 2D FFT spectra.
* **scikit-learn (`accuracy_score`, `confusion_matrix`):** Strictly limited to post-hoc evaluation metrics. No sklearn estimators interact with the training loop to avoid breaking PyTorch's automatic differentiation (`autograd`) graph construction.
* **`Image.MAX_IMAGE_PIXELS = None`:** Disables PIL's decompression bomb safety threshold. Since the pipeline processes a controlled, trusted academic dataset containing high-resolution imagery, this eliminates silent batch-loading stalls or warnings.

---

### 2. Globals

```python
# Hardware routing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DATA_PATH = "/content/drive/MyDrive/CV_project/CV_Project_data/RRDataset_PyTorch_Ready"
MODEL_SAVE_DIR = "/content/drive/MyDrive/CV_project/CV_Project_models"

# Hyperparameters
CROP_SIZE      = 224
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-4
EPOCHS         = 20
```

Centralizing all mutable configuration parameters in a single block ensures notebook-wide consistency, eliminating the risk of parameter mismatch between training, evaluation, and checkpointing.

* **Hardware Routing:** Dynamically selects the execution device via `torch.cuda.is_available()`. Logging the specific GPU device name provides vital runtime verification in volatile cloud environments like Google Colab.
* **`CROP_SIZE = 224`:** Matches the input dimensions required by pre-trained ConvNeXt-Tiny weights. Because the backbone's patchify stem relies on fixed 4×4 non-overlapping patches, deviating from 224×224 would disrupt positional embedding expectations and degrade features.
* **`BATCH_SIZE = 32`:** Balances GPU memory consumption, computational throughput, and gradient variance. It provides stable optimization path convergence during fine-tuning.
* **`LEARNING_RATE = 1e-4`:** Sets the baseline learning rate for the new classification heads. The backbone uses a differential learning rate scale factor ($0.1 \times \text{LR} = 10^{-5}$) to protect pre-trained ImageNet representations from catastrophic forgetting while allowing the heads to learn rapidly.
* **`EPOCHS = 20`:** Selected empirically based on baseline convergence trends (where metrics stabilized between epochs 7 and 15), providing sufficient headroom for slower multi-task adjustments without wasting compute.
* **Path Separation:** Explicitly isolates the read-only input source (`BASE_DATA_PATH`) from the write-only checkpoint target directory (`MODEL_SAVE_DIR`), preventing accidental data overwrites during subsequent pipeline executions.

---

### 3. Utilities — Forensic EDA

Diagnostic and visualization utilities are split into training progression trackers and pre-training exploratory data analysis (EDA) diagnostics used to isolate real vs. AI structural variations.

#### 3.1 Training Visualization
* **`plot_training_history`:** Plots train and validation loss curves on a shared axis to detect overfitting and confirm optimal early-stopping checkpoint selection.

#### 3.2 Classical Feature Extraction
* **`extract_course_features`:** Computes four legacy descriptors to expose low-level synthetic anomalies:
  * **CLAHE:** Normalizes local contrast within 8×8 pixel windows to reveal micro-texture uniformities inside flat AI surfaces (e.g., skin, sky).
  * **Difference of Gaussians (DoG):** Functions as a bandpass filter to detect structural residuals left behind by neural upsampling operations.
  * **HOG Trajectories:** Maps gradient orientation histograms to contrast natural geometric alignments against smoothed, isotropic AI edge distributions.
  * **ORB Keypoints:** Plots FAST corner density distributions to flag semantically implausible keypoint clustering or unnatural texture voids.

#### 3.3 Comparative Diagnostic Dashboard
* **`run_syllabus_eda_comparison`:** Renders a 2×4 matrix comparing real vs. fake images side-by-side. Empirical analysis of this grid justified using `NEAREST` interpolation resizing to protect raw high-frequency features from blurring artifacts.

#### 3.4 Frequency Domain Analysis
* **`plot_advanced_forensics`:** Visualizes signal behaviors across explicit spatial and frequency boundaries:
  * **Laplacian Edge Maps:** Measures second spatial derivatives to highlight differences between sharp physical transitions and smooth diffusion model outputs.
  * **2D FFT Magnitude Spectrum:** Extracts global spatial frequencies to identify distinct grid-like structural patterns ("spectral artifacts") generated by convolutional decoding blocks.
---

### 4. Data

### 4.1 Dataset Profiling & Imbalance Resolution

The raw dataset contains an asymmetrical distribution across its transformation layers, which introduces optimization challenges that are resolved prior to building the network pipeline.

* **Raw Imbalance Discovery:** Profiling the source distribution reveals a structural bottleneck: the "Original" transformation category caps out at exactly 1,250 images per class, whereas "Transmitted" and "Redigitized" variants contain roughly 8,500 images each, totaling 36,499 raw images. 
* **Hardware & Runtime Constraints:** Execution is bounded by a shared **Google Colab T4 GPU** environment. The project requires training 7 separate configurations (2 unimodal baselines, 1 joint model, and 4 ablation sweeps) running at roughly 1 hour per model. Training on the uncurated 36.5k image pool would cause compute-quota exhaustion and immense training stalls.
* **Algorithmic Downsampling:** To satisfy the rubric's class-balance criteria and accommodate hardware limitations, a deterministic subsampling strategy is applied. Using a fixed random initialization seed (`seed(42)`), `random.sample` extracts a uniform subset matched to the lowest common denominator: 1,250 images per distinct sub-category.
* **Mathematical Balance Verification:** The resulting subset scales down to a perfectly balanced pool of exactly 7,500 total images. This complete balance across both Authenticity (Real vs. Fake) and Transformation splits prevents the network from developing majority-class prediction biases (such as blindly outputting "Transmitted"), ensuring that accuracy values accurately reflect learned forensic artifacts.

### 4.2 Stratified Partitioning & Vault Isolation

To prevent overfitting and secure an unbiased final validation protocol, the 7,500 balanced image pool is divided into three strictly isolated sets using a custom nested partition strategy.

* **Split Distribution:** The dataset is split into an **80% Training Set (6,000 images)**, a **10% Validation Set (750 images)**, and a **10% Test Set (750 images)**. The Test Set remains completely locked during training and hyperparameter tuning, serving as the final evaluation benchmark.
* **Multi-Task Stratification:** Splitting is applied uniformly across the combined multi-task label boundary (`Authenticity + Transformation`). This strict stratification guarantees that each distinct subset retains a perfect uniform distribution (1,000 images per class in Train, 125 images per class in Val, and 125 images per class in Test).


### 4.3 Training Set Profiling & Leakage-Free Diagnostics

Statistical feature analysis is performed exclusively on the isolated Training set. This acts as a representative proxy for the dataset without inducing data leakage into the validation or testing pipelines.

* **Forensic Metrics Discovered:** Comparing 500 sampled Real vs. AI images inside the `Original` category revealed distinct mathematical boundaries:
  * **Brightness & Contrast:** AI images exhibit higher average pixel intensity ($+0.0647$) and wider variance ($+0.0603$) than natural images.
  * **Sharpness (Laplacian Variance):** AI images are significantly sharper than real camera images, yielding a higher Laplacian variance ($+200.94$). This indicates the presence of crisp, artificial high-frequency textures or edge profiles characteristic of generative models.
* **Distribution Shift Verification:** Comparative boxplots mapping raw data against the subsampled training vault confirm overlapping distributions across brightness and sharpness domains. This statistical match verifies that downsampling preserved the natural feature distributions of the source data.

![Distribution Shift Verification](figures/distribution_shift_check.png)

#### 4.3.1 Statistical Distribution Shift Verification

To confirm that our algorithmic downsampling did not introduce selection bias or alter the underlying feature distributions of the dataset, a comparative statistical analysis was performed between the full **Raw Original Data** and the **Subsampled Train Vault**. 

* **Brightness Distribution Parity:** The mean pixel intensity distributions are virtually identical. The medians ($\approx 95$), interquartile ranges (IQR, spanning $\approx 65$ to $125$), and full range bounds show near-perfect structural alignment. This confirms that downsampling did not skew the dataset towards abnormally dark or bright subsets.
* **Sharpness (Laplacian Variance) Fidelity:** Measured on a log scale due to extreme structural outliers, the second-order gradient distributions remain highly consistent between both pools. The median variance ($\approx 600$) and the heavy-tailed outlier profiles track each other seamlessly. This ensures that high-frequency structural traits, such as sensor noise and sharp edges, were completely preserved.
* **Pipeline Validation:** The overlapping morphology of these boxplots mathematically proves that the random sampling strategy successfully maintained the statistical characteristics of the source data. The Subsampled Train Vault serves as a reliable, statistically uncorrupted proxy for wide-scale model optimization.
