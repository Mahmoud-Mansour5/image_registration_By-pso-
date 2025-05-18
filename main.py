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

# Import all functionality from image_registration_pso
from image_registration_pso import (
    transform_image, mse, ncc, mutual_information, compute_ssim, 
    post_process_image, distort_image, PSO, apply_nonlinear_distortion,
    apply_local_distortion, apply_multiscale_distortion
)

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
            # Create PSO instance with our parameters
            pso = PSO(
                self.reference_image,
                self.target_image,
                n_particles=self.pso_params.get('n_particles', 100),
                n_iterations=self.pso_params.get('n_iterations', 50),
                w_start=self.pso_params.get('w_start', 0.9),
                w_end=self.pso_params.get('w_end', 0.2),
                c1_start=self.pso_params.get('c1_start', 2.5),
                c1_end=self.pso_params.get('c1_end', 0.5),
                c2_start=self.pso_params.get('c2_start', 0.5),
                c2_end=self.pso_params.get('c2_end', 2.5)
            )

            # Define callback for progress updates
            def callback(iteration, g_best, g_best_score):
                progress = int((iteration + 1) * 100 / self.pso_params.get('n_iterations', 50))
                self.progress.emit(progress)
                
                if iteration % 5 == 0:
                    # Apply current best transformation
                    transformed = transform_image(self.target_image, *g_best)
                    
                    # Calculate metrics
                    ssim_score = compute_ssim(self.reference_image, transformed)
                    ncc_score = ncc(self.reference_image, transformed)
                    mse_score = mse(self.reference_image, transformed)
                    mi_score = mutual_information(self.reference_image, transformed)
                    
                    # Emit preview update
                    self.update_preview.emit({
                        'transformed': transformed,
                        'metrics': {
                            'ssim': ssim_score,
                            'ncc': ncc_score,
                            'mse': mse_score,
                            'mi': mi_score
                        }
                    })

            # Run optimization
            g_best, g_best_score, convergence_history = pso.optimize(callback=callback)
            
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
            
            # Calculate difference map
            diff_map = cv2.absdiff(self.reference_image, final_img)
            
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
        
        # Add parameter controls
        self.n_particles = QSpinBox()
        self.n_particles.setRange(10, 500)
        self.n_particles.setValue(100)
        self.n_particles.setToolTip("Number of particles (solutions) to use in optimization")
        pso_layout.addWidget(QLabel("Particles:"), 0, 0)
        pso_layout.addWidget(self.n_particles, 0, 1)
        
        self.n_iterations = QSpinBox()
        self.n_iterations.setRange(10, 200)
        self.n_iterations.setValue(50)
        self.n_iterations.setToolTip("Number of iterations for optimization")
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
        
        for label in self.metrics_labels.values():
            metrics_layout.addWidget(label)
            
        metrics_group.setLayout(metrics_layout)
        settings_layout.addWidget(metrics_group)
        
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
        self.processing_mode = self.ask_processing_mode()
        if self.processing_mode is None:
            return
            
        if self.processing_mode == 'single':
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
                        self.original_images.append(img)
                        self.reference_images.append(img.copy())
                
                self.show_original_images()
                self.distort_btn.setEnabled(True)
                self.process_btn.setEnabled(False)
                self.save_btn.setEnabled(False)
                
        else:  # pair mode
            ref_file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Reference (Clean) Image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            
            if ref_file:
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
                    
                    ref_img = cv2.imread(ref_file, cv2.IMREAD_GRAYSCALE)
                    if ref_img is not None:
                        ref_img = cv2.resize(ref_img, (256, 256))
                        self.reference_images.append(ref_img)
                        self.original_images.append(ref_img)
                    
                    target_img = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
                    if target_img is not None:
                        target_img = cv2.resize(target_img, (256, 256))
                        self.target_images.append(target_img)
                    
                    self.show_current_images()
                    self.distort_btn.setEnabled(False)
                    self.process_btn.setEnabled(True)
                    self.save_btn.setEnabled(False)
            
    def show_original_images(self):
        """Display current original image"""
        if not self.original_images:
            return
            
        current_img = self.original_images[self.current_index]
        
        height, width = current_img.shape
        bytes_per_line = width
        q_img = QImage(current_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        
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
            # Using distort_image from image_registration_pso.py
            distorted, _ = distort_image(img)
            self.target_images.append(distorted)
            
        self.show_current_images()
        self.process_btn.setEnabled(True)
        
    def show_current_images(self):
        """Display current reference and distorted images"""
        if not self.reference_images or not self.target_images:
            return
            
        ref_img = self.reference_images[self.current_index]
        distorted_img = self.target_images[self.current_index]
        
        combined = np.hstack((ref_img, distorted_img))
        
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
        
    def start_registration(self):
        """Start the registration process"""
        if not self.reference_images or not self.target_images:
            return
            
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
        
        self.worker = ImageRegistrationWorker(
            self.reference_images[self.current_index],
            self.target_images[self.current_index],
            pso_params
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.process_complete)
        self.worker.update_preview.connect(self.update_preview)
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        self.worker.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_preview(self, data):
        """Update preview and metrics during processing"""
        metrics = data['metrics']
        self.metrics_labels['ssim'].setText(f"SSIM: {metrics['ssim']:.4f}")
        self.metrics_labels['ncc'].setText(f"NCC: {metrics['ncc']:.4f}")
        self.metrics_labels['mse'].setText(f"MSE: {metrics['mse']:.2f}")
        self.metrics_labels['mi'].setText(f"MI: {metrics['mi']:.4f}")
        
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
            
        self.results.append(results)
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_label.setText("Registration complete")
        self.show_final_results(results)
        
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
        
        # Difference Map
        plt.subplot(234)
        plt.imshow(results['diff_map'], cmap='jet')
        plt.title('Difference Map', fontsize=12, fontweight='bold', pad=10)
        plt.colorbar(label='Difference')
        plt.axis('off')
        
        # Registration Quality
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
        
        # PSO Convergence
        plt.subplot(236)
        plt.plot(results.get('convergence', []), 'b-', linewidth=2)
        plt.title('PSO Convergence History', fontsize=12, fontweight='bold', pad=10)
        plt.xlabel('Iteration', fontsize=10)
        plt.ylabel('Fitness Score', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plot_img = Image.open(buf)
        plot_array = np.array(plot_img)
        
        height, width, channel = plot_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(plot_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
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
            
        os.makedirs("results", exist_ok=True)
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