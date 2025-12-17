import pygame
import numpy as np
import scipy.ndimage
from numba import njit, prange

# --- 1. CONFIGURATION ---
WIDTH, HEIGHT = 1200, 1000       
GRID_W, GRID_H = 400, 300        

MAX_ANTS = 1000000               
START_ANTS = 50000               
UPDATE_FRACTION = 2              

TRIG_PRECISION = 4096            
INV_2PI = 1.0 / (2 * np.pi)
DTYPE = np.float32               

NEST_X, NEST_Y = WIDTH // 2, HEIGHT // 2
NEST_RADIUS = 50 
NEST_GRID_X = int(NEST_X * (GRID_W / WIDTH))
NEST_GRID_Y = int(NEST_Y * (GRID_H / HEIGHT))

MAX_ENERGY = 1000.0              
ENERGY_COST = 0.5                

# --- 2. TRIG TABLES ---
angles = np.linspace(0, 2*np.pi, TRIG_PRECISION, endpoint=False, dtype=DTYPE)
COS_TABLE = np.cos(angles)
SIN_TABLE = np.sin(angles)

# --- 3. NUMBA KERNEL ---
@njit(parallel=True, fastmath=True, nogil=True)
def update_ants_optimized(x, y, angle, ant_status, energy,
                          grid_blue, grid_red, food_grid, 
                          static_home_grid,
                          sensor_angle_offset, sensor_dist, 
                          turn_speed, random_strength, 
                          chaos_rate, dash_dist, 
                          width, height, 
                          cos_table, sin_table, 
                          step_count, update_fraction,
                          nest_x, nest_y, nest_radius,
                          repro_rate,
                          blue_p_str, red_p_str): 
    
    num_ants = x.shape[0]
    prec_mask = TRIG_PRECISION - 1 
    start_idx = step_count % update_fraction
    loop_count = (num_ants - start_idx + update_fraction - 1) // update_fraction
    
    margin = 2.0
    
    for j in prange(loop_count):
        i = start_idx + j * update_fraction
        
        if ant_status[i] == -1: continue
            
        energy[i] -= ENERGY_COST * update_fraction
        if energy[i] <= 0:
            ant_status[i] = -1
            continue

        norm = angle[i] * INV_2PI
        norm = norm - np.floor(norm)
        main_idx = int(norm * TRIG_PRECISION)
        dir_x = cos_table[main_idx]
        dir_y = sin_table[main_idx]
        
        idx_l = (main_idx - sensor_angle_offset) & prec_mask
        idx_r = (main_idx + sensor_angle_offset) & prec_mask
        
        lx_vec = cos_table[idx_l] * sensor_dist
        ly_vec = sin_table[idx_l] * sensor_dist
        rx_vec = cos_table[idx_r] * sensor_dist
        ry_vec = sin_table[idx_r] * sensor_dist
        
        cx = min(width-1, max(0, int(x[i] + dir_x * sensor_dist)))
        cy = min(height-1, max(0, int(y[i] + dir_y * sensor_dist)))
        lx = min(width-1, max(0, int(x[i] + lx_vec)))
        ly = min(height-1, max(0, int(y[i] + ly_vec)))
        rx = min(width-1, max(0, int(x[i] + rx_vec)))
        ry = min(height-1, max(0, int(y[i] + ry_vec)))
        
        if ant_status[i] == 1:
            home_pull = 5.0
            c_val = grid_blue[cx, cy] + static_home_grid[cx, cy] * home_pull
            l_val = grid_blue[lx, ly] + static_home_grid[lx, ly] * home_pull
            r_val = grid_blue[rx, ry] + static_home_grid[rx, ry] * home_pull
        else:
            c_val = grid_red[cx, cy] * 5.0 + grid_blue[cx, cy] * 0.1
            l_val = grid_red[lx, ly] * 5.0 + grid_blue[lx, ly] * 0.1
            r_val = grid_red[rx, ry] * 5.0 + grid_blue[rx, ry] * 0.1
        
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

        if x[i] < margin:
            x[i] = margin
            angle[i] = np.pi - angle[i]
        elif x[i] > width - margin:
            x[i] = width - margin
            angle[i] = np.pi - angle[i]
            
        if y[i] < margin:
            y[i] = margin
            angle[i] = -angle[i]
        elif y[i] > height - margin:
            y[i] = height - margin
            angle[i] = -angle[i]
        
        ix, iy = int(x[i]), int(y[i])
        
        if ant_status[i] == 1:
            dx = x[i] - nest_x
            dy = y[i] - nest_y
            if (dx*dx + dy*dy) < nest_radius * nest_radius:
                ant_status[i] = 0 
                angle[i] += 3.1415 
                energy[i] = MAX_ENERGY 
                if np.random.random() < repro_rate:
                    child_idx = np.random.randint(0, num_ants)
                    if ant_status[child_idx] == -1:
                        ant_status[child_idx] = 0 
                        energy[child_idx] = MAX_ENERGY
                        x[child_idx] = nest_x
                        y[child_idx] = nest_y
                        angle[child_idx] = np.random.random() * 6.2831
            grid_red[ix, iy] += red_p_str * update_fraction
        else:
            if food_grid[ix, iy] > 0:
                # --- FIXED CONSUMPTION ---
                # Eat 5 units per bite (Faster clearing)
                # Ensure we don't go below zero
                take = 5 
                if food_grid[ix, iy] < 5: take = food_grid[ix, iy]
                food_grid[ix, iy] -= take
                
                ant_status[i] = 1 
                
                b_c = grid_blue[cx, cy]
                b_l = grid_blue[lx, ly]
                b_r = grid_blue[rx, ry]
                
                if b_c > b_l and b_c > b_r: angle[i] += 3.1415 
                elif b_l > b_r: angle[i] += 3.1415 - 0.5
                elif b_r > b_l: angle[i] += 3.1415 + 0.5
                else: angle[i] += 3.1415
                
                grid_red[ix, iy] += (red_p_str * 50.0) 
            
            grid_blue[ix, iy] += blue_p_str * update_fraction


# --- 4. DATA CLASS ---
class AntColony:
    def __init__(self, num_ants, width, height):
        self.width = width
        self.height = height
        self.num_ants = num_ants
        
        self.sim_nest_x = NEST_X * (GRID_W / WIDTH)
        self.sim_nest_y = NEST_Y * (GRID_H / HEIGHT)
        self.sim_nest_r = NEST_RADIUS * (GRID_W / WIDTH)
        
        self.x = np.zeros(num_ants, dtype=DTYPE)
        self.y = np.zeros(num_ants, dtype=DTYPE)
        self.angle = np.zeros(num_ants, dtype=DTYPE)
        self.energy = np.zeros(num_ants, dtype=DTYPE)
        self.ant_status = np.full(num_ants, -1, dtype=np.int32) 

        self.ant_status[:START_ANTS] = 0 
        self.energy[:START_ANTS] = np.random.uniform(0.5 * MAX_ENERGY, MAX_ENERGY, START_ANTS).astype(DTYPE)
        self.x[:START_ANTS] = (np.random.normal(self.sim_nest_x, 5, START_ANTS)).astype(DTYPE)
        self.y[:START_ANTS] = (np.random.normal(self.sim_nest_y, 5, START_ANTS)).astype(DTYPE)
        self.angle[:START_ANTS] = (np.random.rand(START_ANTS) * 2 * np.pi).astype(DTYPE)

        self.grid_blue = np.zeros((width, height), dtype=DTYPE)
        self.grid_red = np.zeros((width, height), dtype=DTYPE)
        self.food_grid = np.zeros((width, height), dtype=np.int32)
        
        Y, X = np.ogrid[:height, :width]
        dist_from_nest = np.sqrt((X - self.sim_nest_x)**2 + (Y - self.sim_nest_y)**2)
        max_dist = np.sqrt(width**2 + height**2) / 2
        self.static_home_grid = np.clip((1.0 - (dist_from_nest / max_dist)), 0, 1).astype(DTYPE)

    def spawn_food(self):
        min_dist = self.width * 0.35
        for _ in range(100):
            fx = np.random.randint(20, self.width - 20)
            fy = np.random.randint(20, self.height - 20)
            dist = np.sqrt((fx - self.sim_nest_x)**2 + (fy - self.sim_nest_y)**2)
            if dist > min_dist:
                radius = np.random.randint(5, 25)
                # --- FIXED DENSITY ---
                # Lower amount per pixel (5) so it clears fast
                amount = 10 
                x_idx, y_idx = np.ogrid[:self.width, :self.height]
                mask = (x_idx - fx)**2 + (y_idx - fy)**2 <= radius**2
                self.food_grid[mask] += amount
                break

    def update(self, params, step_count):
        sensor_offset = int((params["SENSOR_ANGLE"] * INV_2PI) * TRIG_PRECISION)
        
        nx, ny, r = int(self.sim_nest_x), int(self.sim_nest_y), int(self.sim_nest_r)
        x0, x1 = max(0, nx-r), min(self.width, nx+r)
        y0, y1 = max(0, ny-r), min(self.height, ny+r)
        self.grid_blue[x0:x1, y0:y1] = np.maximum(self.grid_blue[x0:x1, y0:y1], 50.0)
        
        update_ants_optimized(
            self.x, self.y, self.angle, self.ant_status, self.energy,
            self.grid_blue, self.grid_red, self.food_grid, self.static_home_grid,
            sensor_offset, params["SENSOR_DIST"], params["TURN_SPEED"],
            params["RANDOM_STRENGTH"], params["CHAOS_RATE"], params["DASH_DIST"],
            self.width, self.height, COS_TABLE, SIN_TABLE,
            step_count, UPDATE_FRACTION,
            self.sim_nest_x, self.sim_nest_y, self.sim_nest_r,
            params["REPRO_RATE"],
            params["BLUE_STR"], params["RED_STR"]
        )

    def diffuse(self, params):
        self.grid_blue = scipy.ndimage.gaussian_filter(self.grid_blue, sigma=params["TRAIL_WIDTH"])
        self.grid_blue *= params["DECAY_RATE"]
        self.grid_red = scipy.ndimage.gaussian_filter(self.grid_red, sigma=params["TRAIL_WIDTH"])
        self.grid_red *= 0.98

# --- 5. UI & MAIN ---
PARAMS = {
    "SENSOR_ANGLE": np.pi / 6, "SENSOR_DIST": 30, "TURN_SPEED": 0.4,
    "RANDOM_STRENGTH": 0.05, "DECAY_RATE": 0.97, 
    "CHAOS_RATE": 0.005, "DASH_DIST": 10, "TRAIL_WIDTH": 1.0, 
    "FOOD_RATE": 0.001, "REPRO_RATE": 0.5,
    "BLUE_STR": 0.5, "RED_STR": 3.0,
    "VIS_BOOST": 2.0
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
        val_str = f"{PARAMS[self.param_key]:.3f}"
        text = font.render(f"{self.label}: {val_str}", True, (255, 255, 255))
        screen.blit(text, (self.rect.x, self.rect.y - 20))

def main():
    pygame.init()
    WIN_W, WIN_H = 1200 + 350, 1000
    screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.DOUBLEBUF)
    pygame.display.set_caption("Ultimate Ant Colony")
    
    colony = AntColony(MAX_ANTS, GRID_W, GRID_H)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)
    big_font = pygame.font.SysFont("Arial", 24)
    
    ui_x = 1220
    sliders = [
        Slider(ui_x, 50, 200, 15, 0.1, 1.5, "SENSOR_ANGLE", "Sensor Angle"),
        Slider(ui_x, 100, 200, 15, 1, 50, "SENSOR_DIST", "Sensor Dist"),
        Slider(ui_x, 150, 200, 15, 0.05, 1.0, "TURN_SPEED", "Turn Speed"),
        Slider(ui_x, 200, 200, 15, 0.0, 0.5, "RANDOM_STRENGTH", "Random"),
        Slider(ui_x, 250, 200, 15, 0.80, 0.999, "DECAY_RATE", "Decay"),
        Slider(ui_x, 300, 200, 15, 0.0, 0.01, "FOOD_RATE", "Food Rate"), 
        Slider(ui_x, 350, 200, 15, 0.0, 1.0, "REPRO_RATE", "Repro Rate"),
        
        Slider(ui_x, 400, 200, 15, 0.1, 5.0, "BLUE_STR", "Blue Strength"),
        Slider(ui_x, 450, 200, 15, 0.1, 10.0, "RED_STR", "Red Strength"),
        Slider(ui_x, 500, 200, 15, 1.0, 20.0, "VIS_BOOST", "Visual Boost"), 
        
        Slider(ui_x, 550, 200, 15, 0.0, 0.1, "CHAOS_RATE", "Chaos Rate"),
        Slider(ui_x, 600, 200, 15, 5, 100, "DASH_DIST", "Dash Dist"),
        Slider(ui_x, 650, 200, 15, 0.5, 3.0, "TRAIL_WIDTH", "Trail Width"),
    ]

    print("Compiling...")
    colony.update(PARAMS, 0)
    print("Done.")

    for _ in range(5): colony.spawn_food()

    running = True
    show_visuals = True
    step_counter = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v: show_visuals = not show_visuals
                if event.key == pygame.K_r: 
                    # Reset
                    colony.ant_status.fill(-1)
                    colony.ant_status[:START_ANTS] = 0
                    colony.energy[:START_ANTS] = MAX_ENERGY
                    colony.x[:START_ANTS] = colony.sim_nest_x
                    colony.y[:START_ANTS] = colony.sim_nest_y
                    colony.grid_blue.fill(0)
                    colony.grid_red.fill(0)
                    colony.food_grid.fill(0)
                    for _ in range(3): colony.spawn_food()
                if event.key == pygame.K_f: colony.spawn_food()
            for s in sliders: s.handle_event(event)

        if np.random.rand() < PARAMS["FOOD_RATE"]: 
            colony.spawn_food()

        colony.update(PARAMS, step_counter)
        colony.diffuse(PARAMS)
        step_counter += 1

        if show_visuals:
            img_array = np.zeros((GRID_W, GRID_H, 3), dtype=np.uint8)
            
            boost = PARAMS["VIS_BOOST"]
            b_log = np.log1p(colony.grid_blue) * 10 * boost
            blue_layer = np.clip(b_log, 0, 100).astype(np.uint8)
            r_log = np.log1p(colony.grid_red) * 10 * boost
            red_layer = np.clip(r_log, 0, 100).astype(np.uint8)
            
            img_array[:, :, 2] = blue_layer
            img_array[:, :, 0] = red_layer
            img_array[:, :, 1] = blue_layer // 2 
            
            food_brightness = np.clip(colony.food_grid * 15, 0, 255).astype(np.uint8)
            food_mask = colony.food_grid > 0
            img_array[food_mask, 1] = food_brightness[food_mask]
            img_array[food_mask, 0] = 0
            img_array[food_mask, 2] = 0
            
            nx, ny, nr = int(colony.sim_nest_x), int(colony.sim_nest_y), int(colony.sim_nest_r)
            img_array[nx-nr:nx+nr, ny-nr:ny+nr, :] = 80 
            
            ix = colony.x.astype(np.int32)
            iy = colony.y.astype(np.int32)
            np.clip(ix, 0, GRID_W-1, out=ix)
            np.clip(iy, 0, GRID_H-1, out=iy)
            
            alive_mask = colony.ant_status >= 0
            ix = ix[alive_mask]
            iy = iy[alive_mask]
            status = colony.ant_status[alive_mask]
            
            img_array[ix, iy, 2] = 255
            img_array[ix, iy, 1] = np.maximum(img_array[ix, iy, 1], 150)
            img_array[ix, iy, 0] = np.maximum(img_array[ix, iy, 0], 150)
            
            carry_mask = status == 1
            cx = ix[carry_mask]
            cy = iy[carry_mask]
            img_array[cx, cy, 0] = 255
            img_array[cx, cy, 1] = 0
            img_array[cx, cy, 2] = 0

            surf = pygame.surfarray.make_surface(img_array)
            surf = pygame.transform.scale(surf, (1200, 1000))
            
            pygame.draw.circle(surf, (150, 150, 150), (NEST_X, NEST_Y), NEST_RADIUS, 2)
            screen.blit(surf, (0, 0))

        pygame.draw.rect(screen, (30, 30, 30), (1200, 0, 350, 1000))
        for s in sliders: s.draw(screen, font)
        
        active_count = np.sum(colony.ant_status >= 0)
        pop_text = big_font.render(f"Alive: {active_count:,}", True, (0, 255, 255))
        screen.blit(pop_text, (1220, 800))
        
        stats = font.render(f"FPS: {clock.get_fps():.1f} | Cap: {MAX_ANTS:,}", True, (0, 255, 0))
        screen.blit(stats, (1220, 950))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()