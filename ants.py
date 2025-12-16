import pygame
import numpy as np
import scipy.ndimage
import matplotlib.cm as cm
from numba import njit, prange

# --- 1. CONFIGURATION ---
WIDTH, HEIGHT = 1200, 1000       # Display Window
GRID_W, GRID_H = 400, 300        # Simulation Grid (Keep small for speed)

# --- THE "MILLION ANT" SETTINGS ---
NUM_ANTS = 1000000               # The Goal.
UPDATE_FRACTION = 2              # 2 = Update 50% of ants per frame (Doubles Performance)
                                 # 4 = Update 25% (Quadruples Performance, choppier)

# Trigonometry Table Precision
TRIG_PRECISION = 4096            # Higher = Smoother movement
INV_2PI = 1.0 / (2 * np.pi)
DTYPE = np.float32               # Use 32-bit floats to save RAM

# --- 2. PRE-COMPUTE TRIG TABLES ---
# We calculate sin/cos once at startup to avoid doing it 1M times per second
angles = np.linspace(0, 2*np.pi, TRIG_PRECISION, endpoint=False, dtype=DTYPE)
COS_TABLE = np.cos(angles)
SIN_TABLE = np.sin(angles)

# --- 3. THE NUMBA KERNEL (The Engine) ---
@njit(parallel=True, fastmath=True, nogil=True)
def update_ants_optimized(x, y, angle, grid, 
                          sensor_angle_offset, sensor_dist, 
                          turn_speed, random_strength, 
                          chaos_rate, dash_dist, 
                          width, height, 
                          cos_table, sin_table, 
                          step_count, update_fraction):
    
    num_ants = x.shape[0]
    prec_mask = TRIG_PRECISION - 1 
    
    # STOCHASTIC: Determine which "slice" of ants to update this frame
    start_idx = step_count % update_fraction
    
    # FIX: Numba prange doesn't like variable steps.
    # We calculate how many ants we need to update, and iterate that count.
    # Formula: ceil((total - start) / step)
    loop_count = (num_ants - start_idx + update_fraction - 1) // update_fraction

    # prange runs 0, 1, 2, ... loop_count
    for j in prange(loop_count):
        
        # Calculate the ACTUAL ant index manually
        i = start_idx + j * update_fraction
        
        # --- A. FAST ANGLE LOOKUP ---
        norm = angle[i] * INV_2PI
        norm = norm - np.floor(norm)
        main_idx = int(norm * TRIG_PRECISION)
        
        dir_x = cos_table[main_idx]
        dir_y = sin_table[main_idx]
        
        # --- B. SENSING ---
        idx_l = (main_idx - sensor_angle_offset) & prec_mask
        idx_r = (main_idx + sensor_angle_offset) & prec_mask
        
        lx_vec = cos_table[idx_l] * sensor_dist
        ly_vec = sin_table[idx_l] * sensor_dist
        rx_vec = cos_table[idx_r] * sensor_dist
        ry_vec = sin_table[idx_r] * sensor_dist
        
        # Manual Modulo wrapping
        cx = int(x[i] + dir_x * sensor_dist) % width
        cy = int(y[i] + dir_y * sensor_dist) % height
        
        lx = int(x[i] + lx_vec) % width
        ly = int(y[i] + ly_vec) % height
        
        rx = int(x[i] + rx_vec) % width
        ry = int(y[i] + ry_vec) % height
        
        # Sample Pheromones
        c_val = grid[cx, cy]
        l_val = grid[lx, ly]
        r_val = grid[rx, ry]
        
        # --- C. BEHAVIOR ---
        if np.random.random() < chaos_rate:
            angle[i] = np.random.random() * 6.2831
            
            norm_dash = (angle[i] * INV_2PI)
            norm_dash = norm_dash - np.floor(norm_dash)
            dash_idx = int(norm_dash * TRIG_PRECISION)
            
            x[i] += cos_table[dash_idx] * dash_dist
            y[i] += sin_table[dash_idx] * dash_dist
            
        else:
            if c_val > l_val and c_val > r_val:
                angle[i] += (np.random.random() - 0.5) * random_strength
            elif c_val < l_val and c_val < r_val:
                angle[i] += (np.random.random() - 0.5) * 2 * turn_speed
            elif l_val > r_val:
                angle[i] -= turn_speed
            elif r_val > l_val:
                angle[i] += turn_speed
            else:
                 angle[i] += (np.random.random() - 0.5) * random_strength
            
            norm_new = (angle[i] * INV_2PI)
            norm_new = norm_new - np.floor(norm_new)
            new_idx = int(norm_new * TRIG_PRECISION)
            
            x[i] += cos_table[new_idx]
            y[i] += sin_table[new_idx]

        # --- D. BOUNDARIES & DEPOSIT ---
        x[i] = x[i] % width
        y[i] = y[i] % height
        
        grid[int(x[i]), int(y[i])] += 1.0 * update_fraction
# --- 4. DATA CLASS ---
class AntColony:
    def __init__(self, num_ants, width, height):
        self.width = width
        self.height = height
        self.num_ants = num_ants
        
        # Circle Init
        center_x = width / 2
        center_y = height / 2
        radius = min(width, height) * 0.4
        
        theta = np.random.rand(num_ants).astype(DTYPE) * 2 * np.pi
        r = (radius * np.sqrt(np.random.rand(num_ants))).astype(DTYPE)
        
        self.x = (center_x + r * np.cos(theta)).astype(DTYPE)
        self.y = (center_y + r * np.sin(theta)).astype(DTYPE)
        self.angle = (np.random.rand(num_ants) * 2 * np.pi).astype(DTYPE)
        
        self.grid = np.zeros((width, height), dtype=DTYPE)

    def update(self, params, step_count):
        # Convert angle (radians) to integer offset for the table
        sensor_offset = int((params["SENSOR_ANGLE"] * INV_2PI) * TRIG_PRECISION)
        
        update_ants_optimized(
            self.x, self.y, self.angle, self.grid,
            sensor_offset,
            params["SENSOR_DIST"],
            params["TURN_SPEED"],
            params["RANDOM_STRENGTH"],
            params["CHAOS_RATE"],
            params["DASH_DIST"],
            self.width, self.height,
            COS_TABLE, SIN_TABLE,
            step_count, UPDATE_FRACTION
        )

    def diffuse(self, params):
        self.grid = scipy.ndimage.gaussian_filter(self.grid, sigma=params["TRAIL_WIDTH"])
        self.grid *= params["DECAY_RATE"]


# --- 5. UI & MAIN ---
# Starting Params
PARAMS = {
    "SENSOR_ANGLE": np.pi / 5, 
    "SENSOR_DIST": 20, 
    "TURN_SPEED": 0.2,
    "RANDOM_STRENGTH": 0.05, 
    "DECAY_RATE": 0.96, # Higher decay for massive numbers of ants
    "CHAOS_RATE": 0.005,
    "DASH_DIST": 30, 
    "TRAIL_WIDTH": 1.0
}

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, param_key, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.param_key = param_key
        self.label = label
        self.dragging = False
        self.handle_x = x + int(((PARAMS[param_key] - min_val) / (max_val - min_val)) * w)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
            self.update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(event.pos[0])

    def update_value(self, mouse_x):
        self.handle_x = max(self.rect.left, min(self.rect.right, mouse_x))
        norm = (self.handle_x - self.rect.left) / self.rect.width
        PARAMS[self.param_key] = self.min_val + norm * (self.max_val - self.min_val)

    def draw(self, screen, font):
        pygame.draw.rect(screen, (80, 80, 80), self.rect)
        pygame.draw.rect(screen, (200, 200, 200), (self.handle_x - 5, self.rect.y - 5, 10, self.rect.height + 10))
        text = font.render(f"{self.label}: {PARAMS[self.param_key]:.3f}", True, (255, 255, 255))
        screen.blit(text, (self.rect.x, self.rect.y - 20))

def main():
    pygame.init()
    
    # Setup Window
    WIN_W, WIN_H = 1200 + 350, 1000
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.DOUBLEBUF)
    pygame.display.set_caption(f"Ant Simulation [CPU] - {NUM_ANTS:,} Ants")
    
    colony = AntColony(NUM_ANTS, GRID_W, GRID_H)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)
    
    # Sliders
    ui_x = 1220
    sliders = [
        Slider(ui_x, 50, 200, 15, 0.1, 1.5, "SENSOR_ANGLE", "Sensor Angle"),
        Slider(ui_x, 100, 200, 15, 1, 50, "SENSOR_DIST", "Sensor Dist"),
        Slider(ui_x, 150, 200, 15, 0.05, 1.0, "TURN_SPEED", "Turn Speed"),
        Slider(ui_x, 200, 200, 15, 0.0, 0.5, "RANDOM_STRENGTH", "Random"),
        Slider(ui_x, 250, 200, 15, 0.80, 0.999, "DECAY_RATE", "Decay"),
        Slider(ui_x, 300, 200, 15, 0.5, 3.0, "TRAIL_WIDTH", "Trail Width"),
        Slider(ui_x, 350, 200, 15, 0.0, 0.1, "CHAOS_RATE", "Chaos Rate"),
    ]

    print("Compiling Numba Kernel (Please Wait ~5s)...")
    # Dry run to trigger JIT compilation before window opens
    colony.update(PARAMS, 0)
    print("Compilation Complete. Starting Sim.")

    running = True
    show_visuals = True
    step_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: show_visuals = not show_visuals
                if event.key == pygame.K_r: colony.grid.fill(0)
            for s in sliders: s.handle_event(event)

        # Update (Stochastic)
        colony.update(PARAMS, step_counter)
        colony.diffuse(PARAMS)
        step_counter += 1

        # Draw
        if show_visuals:
            # Normalize grid for display
            # We raise to power 0.5 to make faint trails visible (Gamma correction)
            view_grid = np.power(colony.grid, 0.4) 
            view_grid = view_grid / (np.max(view_grid) + 0.001)
            
            # Color map
            colored = (cm.magma(view_grid)[:, :, :3] * 255).astype(np.uint8)
            
            # Blit
            surf = pygame.surfarray.make_surface(colored)
            surf = pygame.transform.scale(surf, (1200, 1000))
            screen.blit(surf, (0, 0))

        # UI Overlay
        pygame.draw.rect(screen, (30, 30, 30), (1200, 0, 350, 1000))
        for s in sliders: s.draw(screen, font)
        
        # Stats
        stats = f"FPS: {clock.get_fps():.1f} | Ants: {NUM_ANTS:,}"
        screen.blit(font.render(stats, True, (0, 255, 0)), (ui_x, 950))
        screen.blit(font.render(f"Update Fraction: 1/{UPDATE_FRACTION}", True, (0, 255, 0)), (ui_x, 970))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()