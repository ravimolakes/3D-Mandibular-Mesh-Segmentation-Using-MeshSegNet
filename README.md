# 3D-Mandibular-Mesh-Segmentation-Using-MeshSegNet

A comprehensive deep learning framework for 3D dental mesh segmentation using multiple state-of-the-art approaches. This project implements and compares MeshSegNet, DGCNN, and PointNet++ for segmenting mandibular structures from 3D dental meshes.

## 🎯 Project Overview

This project focuses on automatic segmentation of dental meshes into three anatomical classes:
- **Lower Teeth** (Class 1) - Yellow
- **Left Mandibular Canal** (Class 2) - Red  
- **Right Mandibular Canal** (Class 3) - Blue

## 🧠 Implemented Models

### 1. MeshSegNet
- **Architecture**: Graph Convolutional Network (GCN)
- **Input**: 3D mesh vertices with edge connectivity
- **Features**: 3D coordinates (x, y, z)
- **Framework**: PyTorch Geometric
- **Best Performance**: 72.2% Average Dice Score

### 2. DGCNN (Dynamic Graph CNN)
- **Architecture**: Dynamic Graph Convolutional Neural Network
- **Input**: Point clouds sampled from meshes
- **Features**: 3D coordinates with local graph structure
- **Framework**: torch-points3d
- **Best Performance**: 73.7% Average Dice Score

### 3. PointNet++
- **Architecture**: Hierarchical Point Cloud Processing
- **Input**: Point clouds with hierarchical sampling
- **Features**: 3D coordinates with multi-scale features
- **Framework**: torch-points3d
- **Best Performance**: 68.3% Average Dice Score

## 📁 Project Structure

```
mesh-mandibular-segmentator/
├── DentalSegDataset/           # Raw dataset
│   ├── imagesTr/              # STL mesh files
│   ├── labelTr/               # PLY labeled files
│   └── imagesTs/              # ASTL mesh files
│   
├── pointclouds/               # Generated point cloud data
├── *.ipynb                    # Jupyter notebooks for each model
├── *.yaml                     # Model configuration files
├── *.pth                      # Trained model weights
└── *.json                     # Training metrics
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-points3d
pip install open3d trimesh
pip install monai scikit-learn
pip install matplotlib plotly
pip install joblib tqdm
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mesh-mandibular-segmentator.git
cd mesh-mandibular-segmentator
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare dataset**
```bash
# Convert STL meshes to point clouds (run once)
python prepare_dataset.py
```

### Training Models

#### MeshSegNet Training
```bash
jupyter notebook meshsegnet.ipynb
```

#### DGCNN Training  
```bash
jupyter notebook DGCN_dental.ipynb
```

#### PointNet++ Training
```bash
jupyter notebook pointNet++.ipynb
```

### Model Comparison
```bash
jupyter notebook comparison.ipynb
```

## 📊 Results

### Performance Comparison

| Model | Avg Dice Score | Teeth Dice | Left Canal Dice | Right Canal Dice | Accuracy |
|-------|---------------|------------|-----------------|------------------|----------|
| **DGCNN** | **73.7%** | 95.7% | 57.6% | 67.4% | 92.1% |
| **MeshSegNet** | **72.2%** | 95.8% | 54.6% | 62.4% | 92.0% |
| **PointNet++** | **68.3%** | 95.7% | 51.0% | 58.1% | 91.7% |

### Key Findings
- **DGCNN** achieved the best overall performance with 73.7% average Dice score
- All models performed well on **teeth segmentation** (>95% Dice)
- **Canal segmentation** was more challenging, with DGCNN showing best results
- **Right canal** segmentation was consistently more difficult than left canal

## 🔧 Configuration

### Model Configurations

#### DGCNN (`dgcnn_dental.yaml`)
```yaml
model_name: dgcnn
data:
  number_of_classes: 3
  class_weights: [1.0, 4.0, 4.0]
model:
  k: 20  # number of neighbors
  aggr: max  # aggregation method
  dropout: 0.5
```

#### PointNet++ (`pointnet2_dental.yaml`)
```yaml
model_name: pointnet2
model:
  input_nc: 3
  output_nc: 3
  use_category: False
```

## 📈 Training Metrics

Training metrics are automatically saved as JSON files:
- `metrics_meshsegnet.json`
- `metrics_dgcnn.json` 
- `metrics_pointnet.json`

Each file contains:
- Training losses per epoch
- Dice scores per class
- Average Dice scores
- Overall accuracy

## 🎨 Visualization

The project includes comprehensive 3D visualization capabilities:

- **Matplotlib 3D plots** with multiple viewing angles
- **Plotly interactive visualizations**
- **Open3D mesh rendering**
- **PyVista overlay comparisons**

## 📝 Usage Examples

### Load and Visualize Results
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load predictions
preds = np.load("predicted_labels_meshsegnet.npy")
points = np.load("pointclouds/sample_case_01.npz")['pos']

# Visualize
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=preds, s=1)
plt.show()
```

### Model Inference
```python
import torch
from torch_geometric.nn import GCNConv

# Load trained model
model = MeshSegNet(num_classes=3)
model.load_state_dict(torch.load("best_meshsegnet_model.pth"))
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(mesh_data)
```

## 🏥 Medical Applications

This framework is designed for:
- **Dental implant planning**
- **Orthodontic treatment planning** 
- **Oral surgery preparation**
- **3D dental model analysis**
- **Automated dental anatomy segmentation**

## 📚 Research Background

This project implements state-of-the-art deep learning approaches for 3D medical mesh segmentation:

- **MeshSegNet**: Graph-based approach for mesh processing
- **DGCNN**: Dynamic graph construction for point clouds
- **PointNet++**: Hierarchical feature learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Ravi Molake
- **Email**: [your-email@domain.com]
- **Project Link**: [https://github.com/yourusername/mesh-mandibular-segmentator](https://github.com/yourusername/mesh-mandibular-segmentator)

## 🙏 Acknowledgments

- **torch-points3d** for DGCNN and PointNet++ implementations
- **PyTorch Geometric** for graph neural network tools
- **Open3D** for 3D data processing
- **MONAI** for medical imaging utilities

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{mesh_mandibular_segmentator,
  title={Mesh Mandibular Segmentator: A Deep Learning Framework for 3D Dental Mesh Segmentation},
  author={Ravi Molake},
  year={2024},
  url={https://github.com/yourusername/mesh-mandibular-segmentator}
}
```

---

**Note**: This project is for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
