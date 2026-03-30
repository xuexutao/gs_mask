# Semantic Label Refinement for Gaussian Splatting

This document describes improvements made to address redundant Gaussians in semantic query results for the SAM+CLIP enhanced Gaussian Splatting pipeline.

## Problem Statement

The current semantic labeling pipeline uses SAM (Segment Anything Model) to generate object masks from multiple views, then uses CLIP to assign semantic labels to each mask. 3D Gaussians are projected to 2D views and assigned semantic labels based on which masks they fall into (multi-view voting).

However, this approach leads to two main issues:

1. **Scattered/Outlier Gaussians**: Due to projection errors and noise, some Gaussians get assigned to wrong categories, creating scattered points that don't form coherent objects.

2. **Large Background Gaussians**: Large-scale Gaussians (often representing background or large planar surfaces) can be incorrectly assigned to foreground object categories, leading to "redundant" Gaussians in query results.

These issues manifest as:
- `interactive_semantic_query` returning incorrect assets
- Large redundant Gaussians being retrieved in query results
- Poor accuracy of masked assets

## Solution Overview

We implemented a post-processing refinement step that:
1. **Per-category DBSCAN clustering** to remove outlier Gaussians
2. **Scale-based filtering** to remove large Gaussians likely belonging to background
3. **Optional opacity-based filtering** for further refinement

The refinement is integrated into the training pipeline and also available as a standalone script for post-processing existing PLY files.

## Implementation Details

### 1. DBSCAN Clustering per Category

For each semantic category, we apply DBSCAN clustering to the 3D positions of Gaussians assigned to that category:

```python
def cluster_and_filter(points: np.ndarray, labels: np.ndarray, 
                       eps: float = 0.1, min_samples: int = 10,
                       min_cluster_size: int = 50) -> np.ndarray:
    """Perform DBSCAN clustering per category and keep only large clusters."""
```

**Parameters:**
- `eps`: Maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_samples`: Number of samples in a neighborhood for a point to be considered a core point
- `min_cluster_size`: Minimum cluster size to keep (smaller clusters are treated as outliers)

**Effect:** Removes isolated Gaussians and small clusters that are likely projection errors.

### 2. Scale-based Filtering

Large Gaussians (with large scale magnitude) often represent background elements or large planar surfaces. We filter them out based on a scale threshold:

```python
def filter_by_scale_and_opacity(labels: np.ndarray, scale_mag: np.ndarray, 
                                 opacity: Optional[np.ndarray] = None,
                                 scale_threshold: float = 2.0,
                                 opacity_threshold: float = 0.5) -> np.ndarray:
    """Filter Gaussians based on scale and opacity thresholds."""
```

**Parameters:**
- `scale_threshold`: Gaussians with scale magnitude > threshold are marked as background (label -1)
- `opacity_threshold`: Optional filtering based on opacity (currently commented out)

**Effect:** Removes large background Gaussians that were incorrectly assigned to object categories.

### 3. Integration with Training Pipeline

The refinement is integrated into `train.py` with new command-line arguments:

```bash
# Enable semantic refinement
python train.py --refine_semantic \
                --refine_eps 0.1 \
                --refine_min_samples 10 \
                --refine_min_cluster_size 50 \
                --refine_scale_threshold 2.0 \
                --refine_opacity_threshold 0.5
```

**New arguments in `train.py`:**
- `--refine_semantic`: Enable semantic label refinement after assignment
- `--refine_eps`: DBSCAN epsilon for clustering (default: 0.1)
- `--refine_min_samples`: DBSCAN min_samples for clustering (default: 10)
- `--refine_min_cluster_size`: Minimum cluster size to keep (default: 50)
- `--refine_scale_threshold`: Scale threshold for background filtering (default: 2.0)
- `--refine_opacity_threshold`: Opacity threshold for background filtering (default: 0.5)

The refinement runs after `scene.assign_semantic_labels_multi()` and updates the semantic labels in the Gaussian model.

### 4. Standalone Refinement Script

For post-processing existing PLY files, use `scripts/refine_semantic_labels.py`:

```bash
python scripts/refine_semantic_labels.py \
    --input_ply path/to/input.ply \
    --output_ply path/to/output.ply \
    --eps 0.1 \
    --min_samples 10 \
    --min_cluster_size 50 \
    --scale_threshold 2.0
```

The script reads a PLY file with semantic labels, applies refinement, and writes a new PLY file.

## File Changes

### 1. `train.py` (Modified)
- Added new command-line arguments for semantic refinement
- Integrated refinement logic after semantic label assignment
- Modified `training()` function signature to accept refinement parameters

**Key additions:**
```python
# New command-line arguments
parser.add_argument("--refine_semantic", action="store_true", default=False,
                    help="Enable semantic label refinement after assignment")
parser.add_argument("--refine_eps", type=float, default=0.1,
                    help="DBSCAN epsilon for clustering")
# ... additional arguments ...

# Refinement logic after semantic label assignment
if refine_args is not None and refine_args.get('enabled', False):
    print("Refining semantic labels...")
    # Get Gaussian positions and semantic labels
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    semantic = gaussians._semantic.cpu().numpy()
    # Apply DBSCAN clustering and filtering
    # Update semantic labels
```

### 2. `scripts/refine_semantic_labels.py` (New)
- Standalone script for post-processing semantic labels
- Implements DBSCAN clustering per category
- Implements scale-based and opacity-based filtering
- Handles PLY file I/O with semantic labels

## Usage Examples

### During Training
```bash
# Train with semantic refinement enabled
python train.py -s /path/to/dataset \
                --refine_semantic \
                --refine_eps 0.15 \
                --refine_min_cluster_size 100 \
                --refine_scale_threshold 1.5
```

### Post-processing Existing Models
```bash
# Refine semantic labels in an existing PLY file
python scripts/refine_semantic_labels.py \
    --input_ply output/point_cloud/iteration_30000/point_cloud.ply \
    --output_ply output/point_cloud/iteration_30000/point_cloud_refined.ply \
    --eps 0.1 \
    --min_samples 5 \
    --min_cluster_size 30 \
    --scale_threshold 1.8
```

### Integration with Query System
The refined semantic labels are stored in the `_semantic` tensor of the Gaussian model and are automatically used by:
- `query_semantic_scene.py` for semantic queries
- `interactive_semantic_query.py` for interactive querying
- PLY export/import functions in `scene/gaussian_model.py`

## Parameter Tuning Guidelines

### DBSCAN Parameters
- **`eps`**: Start with 0.1-0.2 for scene-scale datasets. Increase if categories are too fragmented.
- **`min_samples`**: 5-10 for typical scenes. Higher values make clustering more conservative.
- **`min_cluster_size`**: 30-100 depending on scene density. Larger values remove more outliers.

### Scale Filtering
- **`scale_threshold`**: 1.5-3.0. Monitor the distribution of scale magnitudes to set appropriate threshold.
- **`opacity_threshold`**: 0.3-0.7. Use if transparent Gaussians are problematic.

### Scene-specific Adjustments
- **Indoor scenes**: Smaller `eps` (0.05-0.1), smaller `min_cluster_size` (20-50)
- **Outdoor scenes**: Larger `eps` (0.2-0.3), larger `min_cluster_size` (100-200)
- **Dense objects**: Lower `scale_threshold` (1.0-1.5)
- **Sparse scenes**: Higher `scale_threshold` (2.0-3.0)

## Expected Improvements

1. **Reduced Redundancy**: Query results contain fewer irrelevant Gaussians
2. **Improved Accuracy**: Semantic queries return more coherent object representations
3. **Better Visual Quality**: Refined labels produce cleaner segmented objects
4. **Faster Queries**: Fewer Gaussians to process in query operations

## Limitations and Future Work

### Current Limitations
1. **Fixed parameters**: Parameters are global, not adaptive per category
2. **Scale-based filtering**: May remove legitimate large objects
3. **Computational cost**: DBSCAN clustering adds overhead for large scenes

### Future Improvements
1. **Adaptive parameters**: Category-specific clustering parameters
2. **Multi-scale analysis**: Consider object size relative to scene scale
3. **Temporal consistency**: Leverage temporal information in video sequences
4. **Learning-based refinement**: Train a classifier to identify outlier Gaussians

## Conclusion

The semantic label refinement pipeline addresses the key issue of redundant Gaussians in query results by combining spatial clustering and scale-based filtering. This improves the accuracy and coherence of semantic queries while maintaining compatibility with the existing Gaussian Splatting pipeline.

The solution is minimally invasive, requiring only minor modifications to `train.py` and adding a standalone refinement script. The refinement parameters are exposed as command-line arguments, allowing easy experimentation and tuning for different scenes and datasets.