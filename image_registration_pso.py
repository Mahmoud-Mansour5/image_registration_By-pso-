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

def apply_nonlinear_distortion(image, strength=0.3):
    """Apply simple nonlinear distortion to an image."""
    rows, cols = image.shape
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    
    # Create distortion maps
    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = j + strength * np.sin(i/30.0)
            map_y[i, j] = i + strength * np.cos(j/30.0)
    
    # Apply distortion
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

def apply_local_distortion(image, center, radius, strength):
    """Apply local distortion in a specific region."""
    rows, cols = image.shape
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    
    # Create coordinate maps
    for i in range(rows):
        for j in range(cols):
            # Calculate distance from center
            dx = j - center[0]
            dy = i - center[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Apply distortion based on distance
            if distance < radius:
                factor = (1 - distance/radius) * strength
                map_x[i, j] = j + dx * factor
                map_y[i, j] = i + dy * factor
            else:
                map_x[i, j] = j
                map_y[i, j] = i
    
    # Apply distortion
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

def apply_multiscale_distortion(image, scales=[0.5, 1.0, 2.0]):
    """Apply distortions at multiple scales."""
    result = image.copy()
    
    for scale in scales:
        # Resize image to current scale
        scaled = cv2.resize(result, None, fx=scale, fy=scale)
        
        # Apply basic distortion at this scale
        distorted, _ = distort_image(scaled)
        
        # Resize back to original size
        result = cv2.resize(distorted, (image.shape[1], image.shape[0]))
    
    return result

def apply_gaussian_noise(image, mean=0, sigma=25):
    """Apply Gaussian noise to an image."""
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_salt_pepper_noise(image, prob=0.05):
    """Apply salt and pepper noise to an image."""
    noisy = np.copy(image)
    # Salt noise
    salt_mask = np.random.random(image.shape) < prob/2
    noisy[salt_mask] = 255
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < prob/2
    noisy[pepper_mask] = 0
    return noisy

def apply_noise_distortion(image, noise_type='gaussian'):
    """Apply noise-based distortion to an image."""
    if noise_type == 'gaussian':
        return apply_gaussian_noise(image)
    elif noise_type == 'salt_pepper':
        return apply_salt_pepper_noise(image)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def distort_image(image, distortion_type='all'):
    """Apply all types of distortion to an image."""
    # 1. First apply geometric transformations
    angle = np.random.uniform(-45, 45)
    tx = np.random.uniform(-30, 30)
    ty = np.random.uniform(-30, 30)
    scale = np.random.uniform(0.8, 1.2)
    
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    
    distorted = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR)
    
    # 2. Apply Gaussian noise
    sigma = np.random.uniform(5, 15)  # Reduced range for more subtle noise
    gauss = np.random.normal(0, sigma, (rows, cols))
    distorted = np.clip(distorted + gauss, 0, 255).astype(np.uint8)
    
    # 3. Apply salt & pepper noise
    prob = np.random.uniform(0.01, 0.03)  # Reduced probability for more subtle noise
    # Salt noise
    salt_mask = np.random.random(image.shape) < prob/2
    distorted[salt_mask] = 255
    # Pepper noise
    pepper_mask = np.random.random(image.shape) < prob/2
    distorted[pepper_mask] = 0
    
    # 4. Apply nonlinear distortion
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            map_x[i, j] = j + 3.0 * np.sin(i/30.0)  # Subtle wave distortion
            map_y[i, j] = i + 3.0 * np.cos(j/30.0)
    
    distorted = cv2.remap(distorted, map_x, map_y, cv2.INTER_LINEAR)
    
    # Store all the distortion parameters
    true_params = {
        'type': 'all',
        'geometric': {
            'angle': angle,
            'tx': tx,
            'ty': ty,
            'scale': scale
        },
        'noise': {
            'gaussian_sigma': sigma,
            'salt_pepper_prob': prob
        }
    }
    
    return distorted, true_params

class PSO:
    def __init__(self, reference_image, target_image, n_particles=100, n_iterations=50,
                 w_start=0.9, w_end=0.2, c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5):
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
        
        # Initialize particles with improved distribution
        self.particles = self._initialize_particles()
        self.velocities = [np.zeros(4) for _ in range(n_particles)]
        self.p_best = self.particles.copy()
        self.p_best_scores = [float('inf')] * n_particles
        self.g_best = None
        self.g_best_score = float('inf')
        
        # Store convergence history
        self.convergence_history = []
        
    def _initialize_particles(self):
        """Initialize particles with improved distribution strategy."""
        particles = []
        
        # Calculate number of particles for each initialization strategy
        n_narrow = int(0.6 * self.n_particles)  # 60% narrow range
        n_medium = int(0.3 * self.n_particles)  # 30% medium range
        n_wide = self.n_particles - n_narrow - n_medium  # 10% wide range
        
        # Narrow range initialization (focused search)
        for _ in range(n_narrow):
            particle = np.array([
                np.random.uniform(-30, 30),    # angle
                np.random.uniform(-20, 20),    # tx
                np.random.uniform(-20, 20),    # ty
                np.random.uniform(0.9, 1.1)    # scale
            ])
            particles.append(particle)
        
        # Medium range initialization
        for _ in range(n_medium):
            particle = np.array([
                np.random.uniform(-45, 45),    # angle
                np.random.uniform(-30, 30),    # tx
                np.random.uniform(-30, 30),    # ty
                np.random.uniform(0.8, 1.2)    # scale
            ])
            particles.append(particle)
        
        # Wide range initialization (exploration)
        for _ in range(n_wide):
            particle = np.array([
                np.random.uniform(-60, 60),    # angle
                np.random.uniform(-40, 40),    # tx
                np.random.uniform(-40, 40),    # ty
                np.random.uniform(0.7, 1.3)    # scale
            ])
            particles.append(particle)
        
        # Shuffle particles to mix different ranges
        np.random.shuffle(particles)
        return particles
    
    def _evaluate_fitness(self, particle):
        """Evaluate the fitness of a particle with improved weights for medical images."""
        transformed = transform_image(self.target, *particle)
        
        # Calculate multiple similarity metrics
        ssim_score = compute_ssim(self.reference, transformed)
        ncc_score = ncc(self.reference, transformed)
        mse_score = mse(self.reference, transformed)
        mi_score = mutual_information(self.reference, transformed)
        
        # Calculate gradient-based similarity (more robust to noise)
        ref_grad_x = cv2.Sobel(self.reference, cv2.CV_64F, 1, 0, ksize=3)
        ref_grad_y = cv2.Sobel(self.reference, cv2.CV_64F, 0, 1, ksize=3)
        trans_grad_x = cv2.Sobel(transformed, cv2.CV_64F, 1, 0, ksize=3)
        trans_grad_y = cv2.Sobel(transformed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient similarity
        grad_similarity = (np.mean(np.abs(ref_grad_x - trans_grad_x)) + 
                          np.mean(np.abs(ref_grad_y - trans_grad_y))) / 2.0
        
        # Normalize gradient similarity to [0,1] range
        grad_similarity = 1.0 / (1.0 + grad_similarity)
        
        # Combined fitness score with adjusted weights for noisy medical images
        fitness = (0.2 * mse_score -      # Reduced weight for MSE (sensitive to noise)
                  1.5 * ssim_score -      # Reduced weight for SSIM
                  1.0 * ncc_score -       # Maintained weight for NCC
                  1.5 * mi_score -        # Maintained weight for MI
                  2.0 * grad_similarity)  # High weight for gradient similarity
        
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