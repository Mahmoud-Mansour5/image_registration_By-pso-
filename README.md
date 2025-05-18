# Image Registration using Particle Swarm Optimization (PSO)

## Description
This project implements an advanced medical image registration system that utilizes Particle Swarm Optimization (PSO) algorithm to align and match medical images. The system features a user-friendly graphical interface, real-time visualization, and comprehensive quality metrics. It's designed to handle various medical image formats and provide detailed analysis reports.

## Team Members
- Mahmoud Mansour Ibrahim
- Esraa Hamdy Hussein
- Nour Emad Abdullah
- Omnia Mahmoud El-Ashry
- Mostafa Mohamed Gamal

## Project Overview

This project implements an advanced medical image registration system using Particle Swarm Optimization (PSO). The system aims to align two medical images by finding the optimal geometric transformations (rotation, translation, and scaling) that make the distorted image match the reference image.

## Core Components and Technical Features

### 1. Image Distortion
- **Purpose**: Create a distorted version of the original image to test the algorithm's matching capabilities
- **Applied Transformations**:
  - Random rotation: -45° to +45° degrees
  - Horizontal translation: -30 to +30 pixels
  - Vertical translation: -30 to +30 pixels
  - Scaling: 0.8 to 1.2 of original size
- **Implementation**: Using OpenCV's geometric transformation matrix

### 2. PSO Registration Algorithm
- **Core Principle**: Simulates swarm behavior to search for optimal solution
- **Particle Representation**: Each particle represents a set of transformation parameters:
  ```python
  particle = [angle, tx, ty, scale]
  ```

- **Swarm Initialization**:
  - Number of particles: 100 (configurable)
  - Multi-range initialization strategy:
    * 60% particles: narrow range (±30° rotation, ±20 translation, 0.9-1.1 scale)
    * 30% particles: medium range (±45° rotation, ±30 translation, 0.8-1.2 scale)
    * 10% particles: wide range (±60° rotation, ±40 translation, 0.7-1.3 scale)

- **Fitness Function**:
  Combines multiple similarity metrics with weights optimized for medical images:
  ```python
  fitness = 0.3 * MSE - 2.0 * SSIM - 1.0 * NCC - 1.5 * MI
  ```
  Where:
  - MSE: Mean Squared Error (reduced weight due to sensitivity to intensity variations)
  - SSIM: Structural Similarity Index (increased weight for better structural alignment)
  - NCC: Normalized Cross-Correlation (maintained weight for robustness)
  - MI: Mutual Information (increased weight for medical image effectiveness)

- **Particle Updates**:
  - Dynamic update parameters:
    * w: Inertia weight (0.9 → 0.2)
    * c1: Cognitive coefficient (2.5 → 0.5)
    * c2: Social coefficient (0.5 → 2.5)

- **Termination Criteria**:
  - Maximum iterations reached (50 iterations)
  - Or satisfactory matching achieved

### 3. Graphical User Interface (GUI)
- **Key Functions**:
  1. Image Loading: Support for multiple medical image formats
  2. Image Distortion: Apply random distortion
  3. Registration: Execute PSO algorithm
  4. Results Display: 
     - Reference image
     - Distorted image
     - Registered image
     - Difference map
     - Quality metrics
     - Convergence plot

### 4. Registration Quality Metrics

#### Mean Squared Error (MSE)
- **Description**: Measures average squared difference between images
- **Formula**: `MSE = 1/N * Σ(img1 - img2)²`
- **Interpretation**: Lower values indicate better match
- **Range**: [0, ∞) where 0 is perfect match

#### Structural Similarity Index (SSIM)
- **Description**: Measures structural similarity between images
- **Components**: Luminance, contrast, and structure
- **Range**: [-1, 1] where 1 is perfect match
- **Advantages**: Better correlation with human perception

#### Normalized Cross-Correlation (NCC)
- **Description**: Measures correlation between normalized images
- **Formula**: `NCC = Σ((img1 - mean1) * (img2 - mean2)) / (std1 * std2)`
- **Range**: [-1, 1] where 1 indicates perfect match
- **Advantage**: Robust to intensity variations

#### Mutual Information (MI)
- **Description**: Measures shared information between images
- **Calculation**: Based on joint histogram analysis
- **Range**: [0, ∞) where higher values indicate better match
- **Advantage**: Effective for multi-modal image registration

### 5. PSO Algorithm Details

#### Initialization Phase
```python
for particle in range(n_particles):
    if random() < 0.7:  # 70% narrow range
        angle = uniform(-30, 30)
        tx = uniform(-20, 20)
        ty = uniform(-20, 20)
        scale = uniform(0.8, 1.2)
    else:  # 30% wide range
        angle = uniform(-60, 60)
        tx = uniform(-40, 40)
        ty = uniform(-40, 40)
        scale = uniform(0.6, 1.4)
```

#### Update Phase
```python
# Update velocity
velocity = (w * velocity +
           c1 * r1 * (p_best - current) +
           c2 * r2 * (g_best - current))

# Update position
position += velocity

# Dynamic parameter adjustment
w = w_start - (w_start - w_end) * (iteration / max_iterations)
c1 = c1_start - (c1_start - c1_end) * (iteration / max_iterations)
c2 = c2_start + (c2_end - c2_start) * (iteration / max_iterations)
```

### 6. Implementation Benefits
- **Robust Registration**: Combines multiple similarity metrics
- **Adaptive Search**: Dynamic parameter adjustment
- **Efficient Convergence**: Multi-range particle initialization
- **Comprehensive Evaluation**: Multiple quality metrics
- **User-Friendly**: Interactive GUI with real-time feedback

## Requirements

- Python 3.6+
- Required Python packages (install using `pip install -r requirements.txt`):
  - numpy>=1.19.2
  - opencv-python>=4.5.1
  - PyQt5>=5.15.4
  - matplotlib>=3.3.4
  - Pillow>=8.2.0
  - scikit-image>=0.18.1
  - scipy>=1.6.2

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Using the GUI:
   - Click "Load Images" to select one or more images
   - Click "Distort Images" to apply random distortion
   - Click "Start Registration" to begin the PSO registration process
   - Click "Generate PDF Report" to save the results

## Project Structure

- `main.py`: Main application file containing the GUI implementation
- `image_registration_pso.py`: Core PSO and image processing functions
- `requirements.txt`: List of required Python packages

## Features Details

### Image Loading and Processing
- Supports common image formats (PNG, JPG, JPEG, BMP, TIF, TIFF)
- Automatic image resizing for consistency
- Random distortion generation for testing

### PSO Algorithm
- Configurable number of particles and iterations
- Dynamic parameter adjustment during optimization
- Multi-metric fitness function for better accuracy

### Visualization
- Side-by-side display of reference and target images
- Real-time preview of registration progress
- Difference map visualization
- PSO convergence graph

### PDF Report
- Comprehensive results for each processed image
- Quality metrics and transformation parameters
- Visual comparisons and difference maps
- Convergence history graph

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 