import pygame
import numpy as np
import scipy.ndimage
import matplotlib.cm as cm

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1200, 1000  # Window size
GRID_W, GRID_H = 400, 300 # Simulation grid size (smaller = faster)
SCALE_X = WIDTH / GRID_W
SCALE_Y = HEIGHT / GRID_H

NUM_ANTS = 100000
SENSOR_ANGLE = np.pi / 5  # 45 degrees
SENSOR_DIST = 10           # How far ahead they see
TURN_SPEED = 0.4          # How sharply they turn
RANDOM_STRENGTH = 0.3    # Random wobble (probability factor)
DECAY_RATE = 0.65         # How fast trails vanish
CHAOS_RATE = 0.01   # 1% chance to change direction randomly
DASH_DIST = 10 

class AntColony:
    def __init__(self, num_ants, width, height):
        self.width = width
        self.height = height
        self.num_ants = num_ants
        
        # --- 1. CIRCLE INITIALIZATION ---
        
        # A. Settings
        center_x = width / 2
        center_y = height / 2
        # Radius: 40% of the smallest screen dimension
        radius = min(width, height) * 0.4 
        
        # B. Math: Polar -> Cartesian
        # Pick a random angle (0 to 2pi) for every ant
        theta = np.random.rand(num_ants) * 2 * np.pi
        
        # Pick a random distance from center
        # TRICK: np.sqrt() ensures they are spread evenly. 
        # If you want a HOLLOW RING, set this to: r = radius
        r = radius * np.sqrt(np.random.rand(num_ants))
        
        # Convert to X, Y
        self.x = center_x + r * np.cos(theta)
        self.y = center_y + r * np.sin(theta)
        
        # --- 2. Orientation ---
        # Random direction (Chaos)
        self.angle = np.random.rand(num_ants) * 2 * np.pi
        
        # OPTION: Point them OUTWARDS from the start?
        # self.angle = theta 
        
        # OPTION: Point them INWARDS (Implosion)?
        # self.angle = theta + np.pi

        # The Pheromone Grid (Environment)
        self.grid = np.zeros((width, height))
    # Helper to sample grid safely (wrapping around edges)
    def get_sensor_values(self, x_arr, y_arr):
            # Clip coordinates to grid size
            ix = np.clip(x_arr.astype(int), 0, self.width - 1)
            iy = np.clip(y_arr.astype(int), 0, self.height - 1)
            return self.grid[ix, iy]
    def update(self):
        """The core simulation step."""
        
        # --- A. SENSING (The "Vision Cone") ---
        # We calculate 3 sensor positions for ALL ants simultaneously
        # 1. Center Sensor
        cx = self.x + np.cos(self.angle) * SENSOR_DIST
        cy = self.y + np.sin(self.angle) * SENSOR_DIST
        
        # 2. Left Sensor
        lx = self.x + np.cos(self.angle - SENSOR_ANGLE) * SENSOR_DIST
        ly = self.y + np.sin(self.angle - SENSOR_ANGLE) * SENSOR_DIST
        
        # 3. Right Sensor
        rx = self.x + np.cos(self.angle + SENSOR_ANGLE) * SENSOR_DIST
        ry = self.y + np.sin(self.angle + SENSOR_ANGLE) * SENSOR_DIST

        

        c_val = self.get_sensor_values(cx, cy)
        l_val = self.get_sensor_values(lx, ly)
        r_val = self.get_sensor_values(rx, ry)

        # --- B. DECISION LOGIC (Probabilistic + Momentum) ---
        # 1. Forward condition: Center > Left and Center > Right
        # Ant wants to keep moving straight (Momentum)
        forward_mask = (c_val > l_val) & (c_val > r_val)
        
        # 2. Random Steer (The "Probability" factor)
        # Even if pheromones say "go left", we add random noise
        random_steer = (np.random.rand(self.num_ants) - 0.5) * 2 * RANDOM_STRENGTH

        # 3. Turn Logic
        # If Right > Left, turn Right. Else turn Left.
        # We use standard numpy masking to update angles
        turn_mask = ~forward_mask # Ants that need to turn
        
        # Calculate turn direction: +1 (right), -1 (left) based on sensor strength
        # (r_val > l_val) gives True/False, converted to float becomes 1.0/0.0
        turn_dir = (r_val[turn_mask] > l_val[turn_mask]).astype(float) 
        turn_dir = turn_dir * 2 - 1 # Convert to [-1, 1] range
        
        # Apply rotations
        self.angle[turn_mask] += turn_dir * TURN_SPEED
        self.angle += random_steer # Apply noise to everyone

        # --- C. MOVEMENT ---
        # 1. Identify who is bursting this frame
        chaos_roll = np.random.rand(self.num_ants)
        dash_mask = chaos_roll < CHAOS_RATE
        
        # 2. Randomize Angle for bursting ants
        # We give them a totally random direction
        self.angle[dash_mask] = np.random.rand(np.sum(dash_mask)) * 2 * np.pi
        
        # 3. THE KICK: Move them far away instantly
        # Instead of moving 1 step, they move 30 steps in that new direction.
        # This breaks them out of the "pheromone trap" instantly.
        
        # We update x/y for dashing ants differently than normal ants
        # Normal movement (step = 1)
        self.x[~dash_mask] += np.cos(self.angle[~dash_mask])
        self.y[~dash_mask] += np.sin(self.angle[~dash_mask])
        
        # Dash movement (step = 30)
        self.x[dash_mask] += np.cos(self.angle[dash_mask]) * DASH_DIST
        self.y[dash_mask] += np.sin(self.angle[dash_mask]) * DASH_DIST
        
        # Wrap
        self.x = self.x % self.width
        self.y = self.y % self.height

        # Deposit (Keep it to 1 pixel for speed!)
        ix = self.x.astype(int)
        iy = self.y.astype(int)
        self.grid[ix, iy] += 1.0 

    def diffuse(self):
        """The Eulerian step: Blur and Decay the trail map."""
        #clip the grid to avoid overflow
        #self.grid = np.clip(self.grid, 0, 2)
        # 1. Blur (Diffusion) - Spreads pheromones to neighbors
        # This simulates the gas spreading in the air/ground
        self.grid = scipy.ndimage.gaussian_filter(self.grid, sigma=1)
        
        # 2. Decay - Old trails vanish
        self.grid *= DECAY_RATE


def main():
    pygame.init()
    # Use HWACCEL for potential speedup
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    pygame.display.set_caption("Ant Colony Simulation")
    clock = pygame.time.Clock()

    # Initialize Logic
    colony = AntColony(NUM_ANTS, GRID_W, GRID_H)
    
    running = True
    show_visuals = True # Toggle flag

    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    show_visuals = not show_visuals
                    print(f"Visuals: {show_visuals}")
                elif event.key == pygame.K_r:
                    # Reset grid on 'R'
                    colony.grid.fill(0)

        # 2. Simulation Step
        colony.update()
        colony.diffuse()

        # 3. Visualization
        if show_visuals:
                    # 1. Normalize the grid for the colormap (0.0 to 1.0)
                    # using a power law to boost faint trails
                    max_val = np.max(colony.grid) + 0.001 # avoid div by zero
                    norm_grid = np.power(colony.grid, 0.5) / np.power(max_val, 0.5)
                    
                    # 2. Apply a colormap (e.g., 'magma', 'inferno', 'plasma', 'viridis')
                    # cm.magma(grid) returns (Width, Height, 4) -> RGBA floats 0.0-1.0
                    colored_grid = cm.magma(norm_grid) 
                    
                    # 3. Convert to 0-255 uint8 for Pygame
                    # We drop the Alpha channel ([:, :, :3])
                    surf_array = (colored_grid[:, :, :3] * 255).astype(np.uint8)
                    
                    # 4. Blit
                    surface = pygame.surfarray.make_surface(surf_array)
                    surface = pygame.transform.scale(surface, (WIDTH, HEIGHT))
                    screen.blit(surface, (0, 0))
                    
                    pygame.display.flip()
        pygame.display.set_caption(f"Ants: {NUM_ANTS} | FPS: {clock.get_fps():.1f}")
        clock.tick(60) # Limit to 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()