#!/usr/bin/env python
# coding: utf-8

"""
Advanced Image Registration System with GUI
----------------------------------------

This application provides a professional graphical interface for image registration
using Particle Swarm Optimization (PSO). It features:

- Interactive image loading and visualization
- Real-time registration progress
- PDF report generation
- Multi-image batch processing
- Advanced visualization options
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QProgressBar, QScrollArea, QMessageBox, QSpinBox,
                           QDoubleSpinBox, QGroupBox, QGridLayout, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from PIL import Image
import io
import random

# Import the core PSO functionality
from image_registration_pso import (transform_image, mse, ncc, 
                                  mutual_information, compute_ssim, 
                                  post_process_image, distort_image, PSO)

def create_difference_map(img1, img2):
    """Create a colored difference map between two images"""
    diff = cv2.absdiff(img1, img2)
    # Convert to grayscale if the difference is in color
    if len(diff.shape) == 3:
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return diff

def distort_image(image):
    """Apply random distortion to an image"""
    # Create random transformations
    angle = random.uniform(-45, 45)     # Random rotation between -45 and 45 degrees
    tx = random.uniform(-30, 30)        # Random X translation
    ty = random.uniform(-30, 30)        # Random Y translation
    scale = random.uniform(0.8, 1.2)    # Random scaling between 0.8x and 1.2x
    
    # Create transformation matrix
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    distorted = cv2.warpAffine(image, M, (cols, rows))
    return distorted

class ImageRegistrationWorker(QThread):
    """Worker thread for processing image registration"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    update_preview = pyqtSignal(dict)

    def __init__(self, reference_image, target_image, pso_params):
        super().__init__()
        self.reference_image = reference_image
        self.target_image = target_image
        self.pso_params = pso_params
        
    def run(self):
        try:
            # Initialize PSO parameters
            n_particles = self.pso_params.get('n_particles', 100)
            n_iterations = self.pso_params.get('n_iterations', 50)
            w_start = self.pso_params.get('w_start', 0.9)
            w_end = self.pso_params.get('w_end', 0.2)
            c1_start = self.pso_params.get('c1_start', 2.5)
            c1_end = self.pso_params.get('c1_end', 0.5)
            c2_start = self.pso_params.get('c2_start', 0.5)
            c2_end = self.pso_params.get('c2_end', 2.5)
            
            # Initialize particles
            particles = []
            for _ in range(n_particles):
                if np.random.random() < 0.7:
                    particles.append(np.array([
                        np.random.uniform(-30, 30),
                        np.random.uniform(-20, 20),
                        np.random.uniform(-20, 20),
                        np.random.uniform(0.8, 1.2)
                    ]))
                else:
                    particles.append(np.array([
                        np.random.uniform(-60, 60),
                        np.random.uniform(-45, 45),
                        np.random.uniform(-45, 45),
                        np.random.uniform(0.6, 1.4)
                    ]))
            
            velocities = [np.zeros(4) for _ in range(n_particles)]
            p_best = particles[:]
            p_best_scores = [float('inf')] * n_particles
            g_best = None
            g_best_score = float('inf')
            
            # Main PSO loop
            convergence_history = []
            
            for it in range(n_iterations):
                # Update progress
                self.progress.emit(int((it + 1) * 100 / n_iterations))
                
                # Dynamic parameter updates
                progress = it / n_iterations
                w = w_start - (w_start - w_end) * progress
                c1 = c1_start - (c1_start - c1_end) * progress
                c2 = c2_start + (c2_end - c2_start) * progress
                
                for i, p in enumerate(particles):
                    # Apply transformation and calculate metrics
                    angle, tx, ty, scale = p
                    transformed = transform_image(self.target_image, angle, tx, ty, scale)
                    
                    ssim_score = compute_ssim(self.reference_image, transformed)
                    ncc_score = ncc(self.reference_image, transformed)
                    mse_score = mse(self.reference_image, transformed)
                    mi_score = mutual_information(self.reference_image, transformed)
                    
                    # Calculate score
                    score = (0.5 * mse_score - 1.5 * ssim_score - 
                            1.0 * ncc_score - 0.8 * mi_score)
                    
                    if score < p_best_scores[i]:
                        p_best_scores[i] = score
                        p_best[i] = p.copy()
                    
                    if score < g_best_score:
                        g_best_score = score
                        g_best = p.copy()
                        
                        # Emit preview update
                        if it % 5 == 0:
                            self.update_preview.emit({
                                'transformed': transformed,
                                'metrics': {
                                    'ssim': ssim_score,
                                    'ncc': ncc_score,
                                    'mse': mse_score,
                                    'mi': mi_score
                                }
                            })
                
                convergence_history.append(g_best_score)
                
                # Update particles
                for i in range(n_particles):
                    r1, r2 = np.random.random(2)
                    velocities[i] = (w * velocities[i] +
                                   c1 * r1 * (p_best[i] - particles[i]) +
                                   c2 * r2 * (g_best - particles[i]))
                    particles[i] += velocities[i]
            
            # Generate final results
            final_img = transform_image(self.target_image, *g_best)
            final_img = post_process_image(final_img)
            
            # Calculate final metrics
            final_metrics = {
                'ssim': compute_ssim(self.reference_image, final_img),
                'ncc': ncc(self.reference_image, final_img),
                'mse': mse(self.reference_image, final_img),
                'mi': mutual_information(self.reference_image, final_img),
                'parameters': {
                    'angle': g_best[0],
                    'tx': g_best[1],
                    'ty': g_best[2],
                    'scale': g_best[3]
                }
            }
            
            # Create difference map
            diff_map = create_difference_map(self.reference_image, final_img)
            
            # Emit results
            self.finished.emit({
                'final_image': final_img,
                'diff_map': diff_map,
                'metrics': final_metrics,
                'convergence': convergence_history
            })
            
        except Exception as e:
            print(f"Error in worker thread: {str(e)}")
            self.finished.emit(None)

class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Registration System")
        self.setMinimumSize(1200, 800)
        
        # Initialize variables
        self.original_images = []  # Original images before distortion
        self.reference_images = []  # Will store original images
        self.target_images = []    # Will store distorted images
        self.current_index = 0
        self.results = []
        self.processing_mode = None  # 'single' or 'pair'
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create UI elements
        self.create_toolbar(layout)
        self.create_main_area(layout)
        self.create_status_bar()
        
        # Apply styling
        self.apply_styling()

    def create_toolbar(self, layout):
        """Create the application toolbar"""
        toolbar = QHBoxLayout()
        
        # Load button
        load_images_btn = QPushButton("Load Images")
        load_images_btn.clicked.connect(self.load_images)
        
        # Distort button
        self.distort_btn = QPushButton("Distort Images")
        self.distort_btn.clicked.connect(self.distort_images)
        self.distort_btn.setEnabled(False)
        
        # Process button
        self.process_btn = QPushButton("Start Registration")
        self.process_btn.clicked.connect(self.start_registration)
        self.process_btn.setEnabled(False)
        
        # Save button
        self.save_btn = QPushButton("Generate PDF Report")
        self.save_btn.clicked.connect(self.generate_pdf_report)
        self.save_btn.setEnabled(False)
        
        # Add buttons to toolbar
        toolbar.addWidget(load_images_btn)
        toolbar.addWidget(self.distort_btn)
        toolbar.addWidget(self.process_btn)
        toolbar.addWidget(self.save_btn)
        
        layout.addLayout(toolbar)
        
    def create_main_area(self, layout):
        """Create the main display area"""
        main_area = QHBoxLayout()
        
        # Image display area
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(800, 600)
        
        # Settings panel
        settings_panel = self.create_settings_panel()
        
        # Add to main area
        main_area.addWidget(self.image_display, stretch=2)
        main_area.addLayout(settings_panel, stretch=1)
        
        layout.addLayout(main_area)
        
    def create_settings_panel(self):
        """Create the settings panel"""
        settings_layout = QVBoxLayout()
        
        # PSO Parameters group
        pso_group = QGroupBox("PSO Parameters")
        pso_layout = QGridLayout()
        
        # Add parameter controls with wider ranges
        self.n_particles = QSpinBox()
        self.n_particles.setRange(10, 500)  # Increased range
        self.n_particles.setValue(100)
        self.n_particles.setToolTip("Number of particles (solutions) to use in optimization.\nMore particles = better accuracy but slower processing")
        pso_layout.addWidget(QLabel("Particles:"), 0, 0)
        pso_layout.addWidget(self.n_particles, 0, 1)
        
        self.n_iterations = QSpinBox()
        self.n_iterations.setRange(10, 200)  # Increased range
        self.n_iterations.setValue(50)
        self.n_iterations.setToolTip("Number of iterations for optimization.\nMore iterations = better results but longer processing time")
        pso_layout.addWidget(QLabel("Iterations:"), 1, 0)
        pso_layout.addWidget(self.n_iterations, 1, 1)
        
        pso_group.setLayout(pso_layout)
        settings_layout.addWidget(pso_group)
        
        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        settings_layout.addWidget(progress_group)
        
        # Metrics group
        metrics_group = QGroupBox("Current Metrics")
        metrics_layout = QVBoxLayout()
        
        self.metrics_labels = {
            'ssim': QLabel("SSIM: -"),
            'ncc': QLabel("NCC: -"),
            'mse': QLabel("MSE: -"),
            'mi': QLabel("MI: -")
        }
        
        # Add tooltips for metrics
        self.metrics_labels['ssim'].setToolTip("Structural Similarity Index\n1.0 means perfect match")
        self.metrics_labels['ncc'].setToolTip("Normalized Cross-Correlation\n1.0 means perfect match")
        self.metrics_labels['mse'].setToolTip("Mean Squared Error\nLower values indicate better match")
        self.metrics_labels['mi'].setToolTip("Mutual Information\nHigher values indicate better match")
        
        for label in self.metrics_labels.values():
            metrics_layout.addWidget(label)
            
        metrics_group.setLayout(metrics_layout)
        settings_layout.addWidget(metrics_group)
        
        # Batch Processing Status
        if len(self.original_images) > 1:
            batch_group = QGroupBox("Batch Processing")
            batch_layout = QVBoxLayout()
            self.batch_label = QLabel(f"Image {self.current_index + 1} of {len(self.original_images)}")
            batch_layout.addWidget(self.batch_label)
            batch_group.setLayout(batch_layout)
            settings_layout.addWidget(batch_group)
        
        settings_layout.addStretch()
        return settings_layout
        
    def create_status_bar(self):
        """Create the application status bar"""
        self.statusBar().showMessage("Ready")
        
    def apply_styling(self):
        """Apply professional styling to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                margin-top: 12px;
            }
            QLabel {
                color: #212121;
            }
            QProgressBar {
                border: 1px solid #BDBDBD;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        
    def ask_processing_mode(self):
        """Ask user whether they want to process a single image or image pair"""
        msg = QMessageBox()
        msg.setWindowTitle("Processing Mode")
        msg.setText("Please select the processing mode:")
        msg.setIcon(QMessageBox.Question)
        
        # Add buttons for both options
        single_button = msg.addButton("Single Image (Needs Correction)", QMessageBox.ActionRole)
        pair_button = msg.addButton("Two Images (Reference + Distorted)", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Cancel)
        
        msg.exec_()
        
        if msg.clickedButton() == single_button:
            return 'single'
        elif msg.clickedButton() == pair_button:
            return 'pair'
        else:
            return None
        
    def load_images(self):
        """Load images based on selected processing mode"""
        # Ask for processing mode
        self.processing_mode = self.ask_processing_mode()
        if self.processing_mode is None:
            return
            
        if self.processing_mode == 'single':
            # Load single image that needs correction
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Image to Process",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            
            if files:
                self.original_images = []
                self.reference_images = []
                self.target_images = []
                self.results = []
                self.current_index = 0
                
                for file in files:
                    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (256, 256))
                        self.original_images.append(img)  # Store original image
                        self.reference_images.append(img.copy())  # Reference will be the original clean image
                
                self.show_original_images()
                self.distort_btn.setEnabled(True)  # Enable distort button to create distorted version
                self.process_btn.setEnabled(False)  # Disable process button until image is distorted
                self.save_btn.setEnabled(False)
                
        else:  # pair mode
            # First load reference image
            ref_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Reference (Clean) Image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            
            if ref_file:
                # Then load distorted image
                target_file, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select Distorted Image",
                    "",
                    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
                )
                
                if target_file:
                    self.original_images = []
                    self.reference_images = []
                    self.target_images = []
                    self.results = []
                    self.current_index = 0
                    
                    # Load reference image
                    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
                    if ref_img is not None:
                        ref_img = cv2.resize(ref_img, (256, 256))
                        self.reference_images.append(ref_img)
                        self.original_images.append(ref_img)
                    
                    # Load target (distorted) image
                    target_img = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
                    if target_img is not None:
                        target_img = cv2.resize(target_img, (256, 256))
                        self.target_images.append(target_img)
                    
                    self.show_current_images()
                    self.distort_btn.setEnabled(False)  # Disable distort button as we already have distorted image
                    self.process_btn.setEnabled(True)
                    self.save_btn.setEnabled(False)
            
    def show_original_images(self):
        """Display current original image"""
        if not self.original_images:
            return
            
        # Show the current original image
        current_img = self.original_images[self.current_index]
        
        # Convert to QPixmap and display
        height, width = current_img.shape
        bytes_per_line = width
        q_img = QImage(current_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit display while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_display.setPixmap(scaled_pixmap)
        
    def distort_images(self):
        """Apply distortion to loaded images"""
        if not self.original_images:
            return
            
        self.target_images = []
        for img in self.original_images:
            distorted = distort_image(img)
            self.target_images.append(distorted)
            
        self.show_current_images()
        self.process_btn.setEnabled(True)
        
    def show_current_images(self):
        """Display current reference and distorted images"""
        if not self.reference_images or not self.target_images:
            return
            
        # Create a side-by-side display
        ref_img = self.reference_images[self.current_index]
        distorted_img = self.target_images[self.current_index]
        
        # Create combined image
        combined = np.hstack((ref_img, distorted_img))
        
        # Convert to QPixmap and display
        height, width = combined.shape
        bytes_per_line = width
        q_img = QImage(combined.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit display while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_display.setPixmap(scaled_pixmap)
        
    def start_registration(self):
        """Start the registration process"""
        if not self.reference_images or not self.target_images:
            return
            
        # Get PSO parameters
        pso_params = {
            'n_particles': self.n_particles.value(),
            'n_iterations': self.n_iterations.value(),
            'w_start': 0.9,
            'w_end': 0.2,
            'c1_start': 2.5,
            'c1_end': 0.5,
            'c2_start': 0.5,
            'c2_end': 2.5
        }
        
        # Create and start worker thread
        self.worker = ImageRegistrationWorker(
            self.reference_images[self.current_index],
            self.target_images[self.current_index],
            pso_params
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_complete)
        self.worker.update_preview.connect(self.update_preview)
        
        # Disable controls
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        # Start processing
        self.worker.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_preview(self, data):
        """Update preview and metrics during processing"""
        # Update metrics
        metrics = data['metrics']
        self.metrics_labels['ssim'].setText(f"SSIM: {metrics['ssim']:.4f}")
        self.metrics_labels['ncc'].setText(f"NCC: {metrics['ncc']:.4f}")
        self.metrics_labels['mse'].setText(f"MSE: {metrics['mse']:.2f}")
        self.metrics_labels['mi'].setText(f"MI: {metrics['mi']:.4f}")
        
        # Update preview image
        transformed = data['transformed']
        ref_img = self.reference_images[self.current_index]
        combined = np.hstack((ref_img, transformed))
        
        height, width = combined.shape
        bytes_per_line = width
        q_img = QImage(combined.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_display.setPixmap(scaled_pixmap)
        
    def process_complete(self, results):
        """Handle completion of registration process"""
        if results is None:
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred during image registration."
            )
            return
            
        # Store results
        self.results.append(results)
        
        # Enable controls
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Update status
        self.status_label.setText("Registration complete")
        
        # Show final results
        self.show_final_results(results)
        
        # Move to next image if available
        self.current_index += 1
        if self.current_index < len(self.reference_images):
            self.show_current_images()
        else:
            self.current_index = 0
            QMessageBox.information(
                self,
                "Complete",
                "All images have been processed. You can now generate the PDF report."
            )
            
    def show_final_results(self, results):
        """Display final registration results"""
        plt.close('all')
        # Increase figure size and add spacing between subplots
        fig = plt.figure(figsize=(15, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle('Results Summary', fontsize=14, fontweight='bold', y=0.95)
        
        # Reference Image
        plt.subplot(231)
        plt.imshow(self.reference_images[self.current_index], cmap='gray')
        plt.title('Reference', fontsize=12, fontweight='bold', pad=10)
        plt.axis('off')
        
        # Distorted Image
        plt.subplot(232)
        plt.imshow(self.target_images[self.current_index], cmap='gray')
        plt.title('Distorted', fontsize=12, fontweight='bold', pad=10)
        plt.axis('off')
        
        # Registered Image
        plt.subplot(233)
        plt.imshow(results['final_image'], cmap='gray')
        plt.title('Registered', fontsize=12, fontweight='bold', pad=10)
        plt.axis('off')
        
        # Difference Map with improved colormap
        plt.subplot(234)
        plt.imshow(results['diff_map'], cmap='jet')  # Changed to 'jet' for better visualization
        plt.title('Difference Map', fontsize=12, fontweight='bold', pad=10)
        plt.colorbar(label='Difference')
        plt.axis('off')
        
        # Registration Quality with improved layout
        plt.subplot(235)
        metrics = results['metrics']
        quality_text = f"""Registration Quality

Best Transformation Parameters:
• Rotation: {metrics['parameters']['angle']:.2f}°
• Translation X: {metrics['parameters']['tx']:.2f} px
• Translation Y: {metrics['parameters']['ty']:.2f} px
• Scaling: {metrics['parameters']['scale']:.2f}

Metrics:
• MSE: {metrics['mse']:.2f}
  Lower values indicate better match

• SSIM: {metrics['ssim']:.4f}
  1.0 means perfect match

• NCC: {metrics['ncc']:.4f}
  1.0 means perfect match

• MI: {metrics['mi']:.4f}
  Higher values indicate better match"""
        
        plt.text(0.05, 0.95, quality_text, fontsize=10, va='top', 
                transform=plt.gca().transAxes, linespacing=1.8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.axis('off')
        
        # PSO Convergence with improved styling
        plt.subplot(236)
        plt.plot(results.get('convergence', []), 'b-', linewidth=2)
        plt.title('PSO Convergence History', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Fitness Score', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Convert plot to image with higher DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_img = Image.open(buf)
        plot_array = np.array(plot_img)
        
        # Convert to QPixmap and display
        height, width, channel = plot_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(plot_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit display while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_display.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.image_display.setPixmap(scaled_pixmap)
        plt.close()
        
    def generate_pdf_report(self):
        """Generate PDF report with results"""
        if not self.results:
            return
            
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/registration_report_{timestamp}.pdf"
        
        with PdfPages(filename) as pdf:
            for i, result in enumerate(self.results):
                plt.figure(figsize=(11, 8))
                
                # Reference Image
                plt.subplot(231)
                plt.imshow(self.reference_images[i], cmap='gray')
                plt.title('Reference', fontsize=12, fontweight='bold')
                plt.axis('off')
                
                # Distorted Image
                plt.subplot(232)
                plt.imshow(self.target_images[i], cmap='gray')
                plt.title('Distorted', fontsize=12, fontweight='bold')
                plt.axis('off')
                
                # Registered Image
                plt.subplot(233)
                plt.imshow(result['final_image'], cmap='gray')
                plt.title('Registered', fontsize=12, fontweight='bold')
                plt.axis('off')
                
                # Difference Map
                plt.subplot(234)
                plt.imshow(result['diff_map'], cmap='hot')
                plt.title('Difference Map', fontsize=12, fontweight='bold')
                plt.axis('off')
                
                # Registration Quality
                plt.subplot(235)
                metrics = result['metrics']
                quality_text = f"""Registration Quality
                
                MSE: {metrics['mse']:.2f}
                SSIM: {metrics['ssim']:.4f}
                NCC: {metrics['ncc']:.4f}
                MI: {metrics['mi']:.4f}
                
                Best Transformation Parameters:
                Rotation: {metrics['parameters']['angle']:.2f}°
                Translation X: {metrics['parameters']['tx']:.2f}
                Translation Y: {metrics['parameters']['ty']:.2f}
                Scaling: {metrics['parameters']['scale']:.2f}"""
                
                plt.text(0.1, 0.5, quality_text, fontsize=9, va='center')
                plt.axis('off')
                
                # PSO Convergence
                plt.subplot(236)
                plt.plot(result.get('convergence', []), 'b-')
                plt.title('PSO Convergence', fontsize=12, fontweight='bold')
                plt.xlabel('Iteration')
                plt.ylabel('Fitness Score')
                plt.grid(True)
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                
        QMessageBox.information(
            self,
            "Success",
            f"PDF report has been generated: {filename}"
        )

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 