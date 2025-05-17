import cv2
import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim

def transform_image(image, angle, tx, ty, scale):
    """Apply geometric transformation to an image."""
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Add translation
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    transformed = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
    return transformed

def mse(img1, img2):
    """Calculate Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)

def ncc(img1, img2):
    """Calculate Normalized Cross-Correlation between two images."""
    return np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / (np.std(img1) * np.std(img2))

def mutual_information(img1, img2, bins=32):
    """Calculate Mutual Information between two images."""
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    
    # Calculate marginal histograms
    hist_1d_1 = np.sum(hist_2d, axis=0)
    hist_1d_2 = np.sum(hist_2d, axis=1)
    
    # Calculate entropies
    entropy_1 = entropy(hist_1d_1)
    entropy_2 = entropy(hist_1d_2)
    
    # Calculate joint entropy
    hist_2d_normalized = hist_2d / float(np.sum(hist_2d))
    joint_entropy = -np.sum(hist_2d_normalized * np.log2(hist_2d_normalized + np.finfo(float).eps))
    
    # Calculate mutual information
    mutual_info = entropy_1 + entropy_2 - joint_entropy
    return mutual_info

def compute_ssim(img1, img2):
    """Calculate Structural Similarity Index between two images."""
    return ssim(img1, img2, data_range=img1.max() - img1.min())

def post_process_image(image):
    """Post-process the registered image to improve quality."""
    # Ensure pixel values are in valid range
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def distort_image(image):
    """Apply random distortion to an image for testing."""
    # Create random transformations within reasonable ranges
    angle = np.random.uniform(-30, 30)      # Random rotation between -30 and 30 degrees
    tx = np.random.uniform(-20, 20)         # Random X translation
    ty = np.random.uniform(-20, 20)         # Random Y translation
    scale = np.random.uniform(0.8, 1.2)     # Random scaling between 0.8x and 1.2x
    
    # Apply transformation
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    
    # Create transformation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Apply transformation
    distorted = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
    
    # Store the actual transformation parameters for reference
    true_params = {
        'angle': angle,
        'tx': tx,
        'ty': ty,
        'scale': scale
    }
    
    return distorted, true_params

class PSO:
    def __init__(self, reference_image, target_image, n_particles=30, n_iterations=50,
                 w_start=0.9, w_end=0.4, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5):
        """Initialize PSO for image registration."""
        self.reference = reference_image
        self.target = target_image
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        
        # PSO parameters
        self.w_start = w_start
        self.w_end = w_end
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        
        # Initialize particles
        self.particles = self._initialize_particles()
        self.velocities = [np.zeros(4) for _ in range(n_particles)]
        self.p_best = self.particles.copy()
        self.p_best_scores = [float('inf')] * n_particles
        self.g_best = None
        self.g_best_score = float('inf')
        
        # Store convergence history
        self.convergence_history = []
        
    def _initialize_particles(self):
        """Initialize particles with random positions."""
        particles = []
        for _ in range(self.n_particles):
            if np.random.random() < 0.7:
                # Normal initialization range
                particle = np.array([
                    np.random.uniform(-30, 30),    # angle
                    np.random.uniform(-20, 20),    # tx
                    np.random.uniform(-20, 20),    # ty
                    np.random.uniform(0.8, 1.2)    # scale
                ])
            else:
                # Wider initialization range for some particles
                particle = np.array([
                    np.random.uniform(-60, 60),    # angle
                    np.random.uniform(-40, 40),    # tx
                    np.random.uniform(-40, 40),    # ty
                    np.random.uniform(0.6, 1.4)    # scale
                ])
            particles.append(particle)
        return particles
    
    def _evaluate_fitness(self, particle):
        """Evaluate the fitness of a particle."""
        transformed = transform_image(self.target, *particle)
        
        # Calculate multiple similarity metrics
        ssim_score = compute_ssim(self.reference, transformed)
        ncc_score = ncc(self.reference, transformed)
        mse_score = mse(self.reference, transformed)
        mi_score = mutual_information(self.reference, transformed)
        
        # Combined fitness score (weighted sum of metrics)
        # Note: We want to minimize MSE and maximize others
        fitness = (0.5 * mse_score - 1.5 * ssim_score - 
                  1.0 * ncc_score - 0.8 * mi_score)
        
        return fitness
    
    def optimize(self, callback=None):
        """Run PSO optimization."""
        for iteration in range(self.n_iterations):
            # Update PSO parameters
            progress = iteration / self.n_iterations
            w = self.w_start - (self.w_start - self.w_end) * progress
            c1 = self.c1_start - (self.c1_start - self.c1_end) * progress
            c2 = self.c2_start + (self.c2_end - self.c2_start) * progress
            
            # Update each particle
            for i, particle in enumerate(self.particles):
                # Evaluate current position
                score = self._evaluate_fitness(particle)
                
                # Update personal best
                if score < self.p_best_scores[i]:
                    self.p_best_scores[i] = score
                    self.p_best[i] = particle.copy()
                
                # Update global best
                if score < self.g_best_score:
                    self.g_best_score = score
                    self.g_best = particle.copy()
            
            # Store best score for convergence history
            self.convergence_history.append(self.g_best_score)
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.random(2)
                
                # Update velocity
                self.velocities[i] = (w * self.velocities[i] +
                                    c1 * r1 * (self.p_best[i] - self.particles[i]) +
                                    c2 * r2 * (self.g_best - self.particles[i]))
                
                # Update position
                self.particles[i] += self.velocities[i]
            
            # Call callback if provided
            if callback:
                callback(iteration, self.g_best, self.g_best_score)
        
        # Return best transformation found
        return self.g_best, self.g_best_score, self.convergence_history 