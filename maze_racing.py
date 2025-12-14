import pygame
import random
import numpy as np
from collections import defaultdict, deque
import sys

sys.setrecursionlimit(10000)

ROWS = COLS = 21
CELL = 28
MAZE_W = COLS * CELL
SIDEBAR = 360
WIDTH = MAZE_W + SIDEBAR
HEIGHT = max(ROWS * CELL, 600)

# Scrolling parameters
SIDEBAR_CONTENT_HEIGHT = 0  # Will be calculated dynamically
scroll_offset = 0
scroll_velocity = 0
MAX_SCROLL = 0

TRAIN_TIME = 20_000
MAX_RACE_STEPS = 800
FPS = 60 # Standard 
TRAIL_LENGTH = 40
RACE_SPEED = 1  # Frames to advance per update during race (start slower)
TRAINING_SPEED = 1  # AI training steps per frame

COLORS = {
    'bg': (12, 12, 22),
    'wall': (30, 32, 48),
    'wall_border': (45, 48, 65),
    'floor': (22, 24, 35),
    'start': (50, 200, 120),
    'goal': (255, 200, 50),
    'sidebar_bg': (18, 18, 28),
    'text': (230, 232, 240),
    'text_dim': (130, 135, 150),
    'panel_border': (65, 70, 90),
    'progress_bg': (35, 38, 50),
    'progress_fill': (80, 200, 140),
    'glow': (100, 220, 160),
}

AI_CONFIGS = [
    ("Explorer", (255, 100, 120), 0.10, 0.97, 0.90),
    ("Sprinter", (100, 200, 255), 0.10, 0.97, 0.90),
    ("Balanced", (150, 255, 100), 0.10, 0.97, 0.90),
    ("Adaptive", (255, 200, 100), 0.10, 0.97, 0.90),
]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("AI Maze Racing Tournament")
clock = pygame.time.Clock()

# Scale factors for resizing
scale_x = 1.0
scale_y = 1.0
scaled_surface = None

font_small = pygame.font.SysFont("Segoe UI", 15)
font_medium = pygame.font.SysFont("Segoe UI", 19)
font_large = pygame.font.SysFont("Segoe UI", 24, bold=True)
font_title = pygame.font.SysFont("Segoe UI", 34, bold=True)

def generate_maze(r, c):
    maze = [[1] * c for _ in range(r)]
    stack = []
    
    def carve_iterative(start_x, start_y):
        stack.append((start_x, start_y))
        while stack:
            x, y = stack[-1]
            dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(dirs)
            found = False
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 1 <= nx < c - 1 and 1 <= ny < r - 1 and maze[ny][nx]:
                    maze[ny][nx] = 0
                    maze[y + dy // 2][x + dx // 2] = 0
                    stack.append((nx, ny))
                    found = True
                    break
            if not found:
                stack.pop()
    
    maze[1][1] = 0
    carve_iterative(1, 1)
    
    for _ in range((r * c) // 15):
        x = random.randrange(2, c - 2)
        y = random.randrange(2, r - 2)
        maze[y][x] = 0
    
    return maze

maze = generate_maze(ROWS, COLS)
START = (1, 1)
GOAL = (COLS - 2, ROWS - 2)

def valid(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS and maze[y][x] == 0

def move(pos, a):
    x, y = pos
    if a == 0: y -= 1
    elif a == 1: y += 1
    elif a == 2: x -= 1
    elif a == 3: x += 1
    return (x, y) if valid(x, y) else pos

def bfs_distance(start, goal):
    """Calculate shortest path distance using BFS"""
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        pos, dist = queue.popleft()
        if pos == goal:
            return dist + 1  # +1 to include the goal position
        
        for action in range(4):
            next_pos = move(pos, action)
            if next_pos not in visited and next_pos != pos:
                visited.add(next_pos)
                queue.append((next_pos, dist + 1))
    
    return float('inf')

OPTIMAL_PATH_LENGTH = bfs_distance(START, GOAL)

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_rounded_rect(surface, color, rect, radius=6, border_color=None, border_width=2):
    x, y, w, h = rect
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect, width=border_width, border_radius=radius)

def draw_glow_rect(surface, color, rect, glow_radius=8):
    """Draw a rectangle with a soft glow effect"""
    x, y, w, h = rect
    for i in range(glow_radius, 0, -2):
        alpha = int(30 * (1 - i / glow_radius))
        glow_color = tuple(min(255, c + alpha) for c in color)
        glow_rect = (x - i, y - i, w + 2*i, h + 2*i)
        s = pygame.Surface((w + 2*i, h + 2*i), pygame.SRCALPHA)
        pygame.draw.rect(s, (*glow_color, alpha), (0, 0, w + 2*i, h + 2*i), border_radius=6+i)
        surface.blit(s, (x - i, y - i))

class AI:
    def __init__(self, name, color, alpha, gamma, eps):
        self.name = name
        self.color = color
        self.alpha = alpha
        self.gamma = gamma
        self.initial_eps = eps
        self.eps = eps
        self.q = defaultdict(lambda: np.zeros(4))
        self.total_rewards = 0
        self.positive_rewards = 0  # Track positive rewards separately
        self.goals_reached = 0
        self.cells_explored = set()
        self.train_steps_done = 0
        self.path = []
        self.reset()
    
    def reset(self):
        self.pos = START
        self.vis = set()
        self.path = []
        self.steps = None
    
    def train_step(self):
        self.train_steps_done += 1
        decay = max(0.01, 1.0 - (self.train_steps_done / TRAIN_TIME) * 0.99)
        current_eps = self.initial_eps * decay
        
        if random.random() < current_eps:
            a = random.randint(0, 3)
        else:
            a = int(np.argmax(self.q[self.pos]))
        
        npos = move(self.pos, a)
        
        r = -1
        if npos == self.pos:
            r = -20
        elif npos in self.vis:
            r = -8
        
        if npos == GOAL:
            r = 200
            self.goals_reached += 1
            self.positive_rewards += 200
        else:
            # Strong distance-based reward shaping
            old_dist = abs(self.pos[0] - GOAL[0]) + abs(self.pos[1] - GOAL[1])
            new_dist = abs(npos[0] - GOAL[0]) + abs(npos[1] - GOAL[1])
            distance_reward = (old_dist - new_dist) * 5
            r += distance_reward
            if distance_reward > 0:
                self.positive_rewards += distance_reward
        
        self.total_rewards += r
        self.q[self.pos][a] += self.alpha * (r + self.gamma * np.max(self.q[npos]) - self.q[self.pos][a])
        self.pos = npos
        self.vis.add(self.pos)
        self.cells_explored.add(self.pos)
        
        if self.pos == GOAL:
            self.pos = START
            self.vis.clear()
    
    def race(self):
        self.reset()
        self.path = [START]
        visited_count = defaultdict(int)
        visited_count[START] = 1
        
        for step in range(MAX_RACE_STEPS):
            # Pure exploitation - use only learned Q-values
            q_values = self.q[self.pos].copy()
            
            # Strong penalty for revisiting cells
            for action in range(4):
                next_pos = move(self.pos, action)
                if visited_count[next_pos] > 0:
                    q_values[action] -= 100 * visited_count[next_pos]
            
            # Choose best action deterministically
            a = int(np.argmax(q_values))
            
            self.pos = move(self.pos, a)
            self.path.append(self.pos)
            visited_count[self.pos] += 1
            
            if self.pos == GOAL:
                self.steps = len(self.path)
                return
        
        self.steps = None
    
    def get_efficiency(self):
        if self.steps and OPTIMAL_PATH_LENGTH > 0:
            return min(100, int((OPTIMAL_PATH_LENGTH / self.steps) * 100))
        return 0

ais = [AI(*cfg) for cfg in AI_CONFIGS]

def draw_maze():
    # Draw floor with subtle pattern
    for y in range(ROWS):
        for x in range(COLS):
            rect = (x * CELL, y * CELL, CELL, CELL)
            if maze[y][x]:
                # Walls with depth
                pygame.draw.rect(screen, COLORS['wall'], rect)
                pygame.draw.rect(screen, COLORS['wall_border'], rect, 1)
                # Add subtle highlight
                highlight = (x * CELL, y * CELL, CELL, 2)
                pygame.draw.rect(screen, tuple(min(255, c + 8) for c in COLORS['wall']), highlight)
            else:
                pygame.draw.rect(screen, COLORS['floor'], rect)
                # Checkerboard pattern
                if (x + y) % 2 == 0:
                    pygame.draw.rect(screen, tuple(min(255, c + 2) for c in COLORS['floor']), rect)
    
    # Start with glow
    start_rect = (START[0] * CELL + 2, START[1] * CELL + 2, CELL - 4, CELL - 4)
    draw_glow_rect(screen, COLORS['start'], start_rect, 4)
    pygame.draw.rect(screen, COLORS['start'], start_rect, border_radius=4)
    pygame.draw.rect(screen, COLORS['glow'], start_rect, 2, border_radius=4)
    
    # Goal with animated glow
    goal_rect = (GOAL[0] * CELL + 2, GOAL[1] * CELL + 2, CELL - 4, CELL - 4)
    draw_glow_rect(screen, COLORS['goal'], goal_rect, 6)
    pygame.draw.rect(screen, COLORS['goal'], goal_rect, border_radius=4)
    pygame.draw.rect(screen, (255, 255, 255), goal_rect, 2, border_radius=4)

def draw_trail(path, color, current_frame, trail_length=TRAIL_LENGTH):
    if not path:
        return
    
    start_idx = max(0, current_frame - trail_length)
    end_idx = min(len(path), current_frame + 1)
    
    for i in range(start_idx, end_idx):
        if i >= len(path):
            break
        
        alpha = (i - start_idx) / max(1, (end_idx - start_idx))
        fade_color = lerp_color(COLORS['floor'], color, alpha * 0.7)
        
        x, y = path[i]
        size = int(CELL * (0.25 + 0.45 * alpha))
        offset = (CELL - size) // 2
        
        trail_rect = (x * CELL + offset, y * CELL + offset, size, size)
        pygame.draw.rect(screen, fade_color, trail_rect, border_radius=3)
    
    # Current position with glow
    if current_frame < len(path):
        x, y = path[current_frame]
        agent_rect = (x * CELL + 3, y * CELL + 3, CELL - 6, CELL - 6)
        
        # Glow effect
        s = pygame.Surface((CELL + 4, CELL + 4), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, 60), (0, 0, CELL + 4, CELL + 4), border_radius=8)
        screen.blit(s, (x * CELL - 2, y * CELL - 2))
        
        pygame.draw.rect(screen, color, agent_rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), agent_rect, 2, border_radius=5)

def draw_sidebar_training(train_step):
    global SIDEBAR_CONTENT_HEIGHT, MAX_SCROLL
    
    # Draw background first
    pygame.draw.rect(screen, COLORS['sidebar_bg'], (MAZE_W, 0, SIDEBAR, HEIGHT))
    
    # Calculate content height
    content_height = 150 + len(ais) * 140 + 100
    SIDEBAR_CONTENT_HEIGHT = content_height
    MAX_SCROLL = max(0, content_height - HEIGHT)
    
    # Create a surface for sidebar content that can be scrolled
    sidebar_surface = pygame.Surface((SIDEBAR - 20, content_height), pygame.SRCALPHA)
    sidebar_surface.fill((0, 0, 0, 0))  # Transparent
    
    y_offset = 20
    
    title = font_title.render("TRAINING", True, COLORS['progress_fill'])
    sidebar_surface.blit(title, (15, y_offset))
    y_offset += 50
    
    progress = train_step / TRAIN_TIME
    bar_width = SIDEBAR - 70
    bar_height = 28
    bar_x = 15
    
    draw_rounded_rect(sidebar_surface, COLORS['progress_bg'], (bar_x, y_offset, bar_width, bar_height), 10)
    if progress > 0:
        fill_width = int(bar_width * progress)
        grad_color = lerp_color((100, 150, 200), COLORS['progress_fill'], progress)
        draw_rounded_rect(sidebar_surface, grad_color, (bar_x, y_offset, fill_width, bar_height), 10)
    
    pct_text = font_medium.render(f"{int(progress * 100)}%", True, COLORS['text'])
    sidebar_surface.blit(pct_text, (bar_x + bar_width // 2 - pct_text.get_width() // 2, y_offset + 4))
    y_offset += 35
    
    step_text = font_small.render(f"{train_step:,} / {TRAIN_TIME:,} steps", True, COLORS['text_dim'])
    sidebar_surface.blit(step_text, (bar_x, y_offset))
    y_offset += 40
    
    header = font_large.render("AI AGENTS", True, COLORS['text'])
    sidebar_surface.blit(header, (15, y_offset))
    y_offset += 45
    
    for ai in ais:
        panel_h = 130
        draw_rounded_rect(sidebar_surface, (25, 27, 38), (10, y_offset - 5, SIDEBAR - 50, panel_h), 10, ai.color, 1)
        
        pygame.draw.circle(sidebar_surface, ai.color, (30, y_offset + 20), 10)
        pygame.draw.circle(sidebar_surface, (255, 255, 255), (30, y_offset + 20), 10, 2)
        
        name_text = font_medium.render(ai.name, True, COLORS['text'])
        sidebar_surface.blit(name_text, (50, y_offset + 10))
        
        # Reward bar visualization
        reward_label_y = y_offset + 40
        bar_y = y_offset + 60
        bar_w = SIDEBAR - 90
        bar_h = 18
        bar_x = 25
        
        reward_text = font_small.render(f"Rewards: +{ai.positive_rewards:,.0f}", True, COLORS['progress_fill'])
        sidebar_surface.blit(reward_text, (bar_x, reward_label_y))
        
        draw_rounded_rect(sidebar_surface, COLORS['progress_bg'], (bar_x, bar_y, bar_w, bar_h), 6)
        
        # Calculate reward percentage
        max_possible_reward = ai.goals_reached * 200 + ai.train_steps_done * 5
        if max_possible_reward > 0:
            reward_pct = min(1.0, ai.positive_rewards / max_possible_reward)
            fill_w = int(bar_w * reward_pct)
            reward_color = lerp_color((255, 150, 80), COLORS['progress_fill'], reward_pct)
            draw_rounded_rect(sidebar_surface, reward_color, (bar_x, bar_y, fill_w, bar_h), 6)
        
        stats = [
            f"Goals: {ai.goals_reached}",
            f"Explored: {len(ai.cells_explored)} cells",
        ]
        stats_y = y_offset + 85
        for i, stat in enumerate(stats):
            stat_text = font_small.render(stat, True, COLORS['text_dim'])
            sidebar_surface.blit(stat_text, (25, stats_y + i * 20))
        
        y_offset += panel_h + 12
    
    y_offset += 20
    hint = font_small.render("Learning optimal maze navigation...", True, COLORS['text_dim'])
    sidebar_surface.blit(hint, (15, y_offset))
    y_offset += 25
    
    controls = font_small.render("Scroll: Mouse wheel or ↑↓", True, COLORS['text_dim'])
    sidebar_surface.blit(controls, (15, y_offset))
    
    # Draw the scrollable content with clipping
    screen.blit(sidebar_surface, (MAZE_W + 10, 0), (0, int(scroll_offset), SIDEBAR - 20, HEIGHT))
    
    # Draw scrollbar if content is larger than viewport
    if MAX_SCROLL > 0:
        scrollbar_height = max(30, int(HEIGHT * (HEIGHT / content_height)))
        scrollbar_y = int((scroll_offset / MAX_SCROLL) * (HEIGHT - scrollbar_height))
        scrollbar_x = WIDTH - 10
        pygame.draw.rect(screen, (80, 85, 100), (scrollbar_x, scrollbar_y, 6, scrollbar_height), border_radius=3)
    
    # Draw border
    pygame.draw.line(screen, COLORS['panel_border'], (MAZE_W, 0), (MAZE_W, HEIGHT), 3)

def draw_sidebar_race(frame):
    sidebar_x = MAZE_W + 10
    pygame.draw.rect(screen, COLORS['sidebar_bg'], (MAZE_W, 0, SIDEBAR, HEIGHT))
    pygame.draw.line(screen, COLORS['panel_border'], (MAZE_W, 0), (MAZE_W, HEIGHT), 3)
    
    title = font_title.render("LEADERBOARD", True, COLORS['goal'])
    screen.blit(title, (sidebar_x + 15, 20))
    
    # Speed control display
    speed_text = font_medium.render(f"Speed: {RACE_SPEED}x", True, COLORS['text'])
    screen.blit(speed_text, (sidebar_x + SIDEBAR - 130, 25))
    
    speed_hint = font_small.render("[↑/↓]", True, COLORS['text_dim'])
    screen.blit(speed_hint, (sidebar_x + SIDEBAR - 130, 50))
    
    y_offset = 75
    
    medals = ["🥇", "🥈", "🥉", "4th"]
    medal_colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50), (120, 120, 140)]
    
    for i, ai in enumerate(ais):
        panel_h = 82
        is_winner = i == 0 and ai.steps
        panel_color = (35, 40, 52) if is_winner else (25, 27, 38)
        border_w = 2 if is_winner else 1
        
        draw_rounded_rect(screen, panel_color, (sidebar_x + 10, y_offset - 5, SIDEBAR - 40, panel_h), 10, ai.color, border_w)
        
        medal_text = font_large.render(medals[i], True, medal_colors[i])
        screen.blit(medal_text, (sidebar_x + 20, y_offset + 5))
        
        name_text = font_medium.render(ai.name, True, ai.color)
        screen.blit(name_text, (sidebar_x + 65, y_offset + 10))
        
        if ai.steps:
            steps_text = font_medium.render(f"{ai.steps} steps", True, COLORS['text'])
            efficiency = ai.get_efficiency()
            eff_color = COLORS['progress_fill'] if efficiency > 60 else (255, 180, 80) if efficiency > 40 else (255, 120, 120)
            eff_text = font_small.render(f"Efficiency: {efficiency}%", True, eff_color)
        else:
            steps_text = font_medium.render("FAILED", True, (255, 90, 90))
            eff_text = font_small.render("Did not reach goal", True, COLORS['text_dim'])
        
        screen.blit(steps_text, (sidebar_x + 65, y_offset + 35))
        screen.blit(eff_text, (sidebar_x + 65, y_offset + 55))
        
        y_offset += panel_h + 8
    
    y_offset += 15
    stats_title = font_large.render("RACE INFO", True, COLORS['text'])
    screen.blit(stats_title, (sidebar_x + 15, y_offset))
    y_offset += 38
    
    max_path = max((len(ai.path) for ai in ais), default=1)
    progress = min(frame / max_path, 1.0) if max_path > 0 else 0
    
    draw_rounded_rect(screen, (25, 27, 38), (sidebar_x + 10, y_offset - 5, SIDEBAR - 40, 85), 10, COLORS['panel_border'])
    
    race_stats = [
        f"Frame: {frame} / {max_path}",
        f"Progress: {int(progress * 100)}%",
        f"Finishers: {sum(1 for ai in ais if ai.steps and frame >= ai.steps)} / {len(ais)}",
        f"Optimal: {OPTIMAL_PATH_LENGTH} steps",
    ]
    for stat in race_stats:
        stat_text = font_small.render(stat, True, COLORS['text_dim'])
        screen.blit(stat_text, (sidebar_x + 25, y_offset + 5))
        y_offset += 20
    
    hint1 = font_small.render("R: restart • N: new maze • ↑↓: speed", True, COLORS['text_dim'])
    screen.blit(hint1, (sidebar_x + 15, HEIGHT - 35))

TRAINING = True
train_step = 0
frame = 0
RACE_SPEED = 1  # Make it adjustable (start at 1x)
TRAINING_SPEED = 1  # Start at 1x for training too

# Create a surface to draw on at native resolution
game_surface = pygame.Surface((WIDTH, HEIGHT))

running = True
while running:
    clock.tick(FPS)
    
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((e.w, e.h), pygame.RESIZABLE)
            scale_x = e.w / WIDTH
            scale_y = e.h / HEIGHT
        if e.type == pygame.MOUSEWHEEL:
            # Smooth scrolling
            scroll_velocity -= e.y * 30
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_r and not TRAINING:
                for ai in ais:
                    ai.race()
                ais.sort(key=lambda a: a.steps if a.steps else 999999)
                frame = 0
                scroll_offset = 0
                scroll_velocity = 0
            if e.key == pygame.K_n:
                maze = generate_maze(ROWS, COLS)
                OPTIMAL_PATH_LENGTH = bfs_distance(START, GOAL)
                ais = [AI(*cfg) for cfg in AI_CONFIGS]
                TRAINING = True
                train_step = 0
                frame = 0
                scroll_offset = 0
                scroll_velocity = 0
            if e.key == pygame.K_RIGHT and not TRAINING:
                RACE_SPEED = min(10, RACE_SPEED + 1)
            if e.key == pygame.K_LEFT and not TRAINING:
                RACE_SPEED = max(1, RACE_SPEED - 1)
            if e.key == pygame.K_UP:
                scroll_velocity -= 50
            if e.key == pygame.K_DOWN:
                scroll_velocity += 50
    
    # Update smooth scrolling
    scroll_offset += scroll_velocity * 0.16
    scroll_velocity *= 0.85  # Friction
    
    # Clamp scroll offset
    scroll_offset = max(0, min(MAX_SCROLL, scroll_offset))
    
    # Draw everything to the game surface at native resolution
    game_surface.fill(COLORS['bg'])
    
    # Temporarily set screen to game_surface for drawing
    temp_screen = screen
    screen = game_surface
    
    draw_maze()
    
    if TRAINING:
        for _ in range(TRAINING_SPEED):
            for ai in ais:
                ai.train_step()
        
        # Draw AI positions
        for ai in ais:
            agent_rect = (ai.pos[0] * CELL + 5, ai.pos[1] * CELL + 5, CELL - 10, CELL - 10)
            pygame.draw.rect(screen, ai.color, agent_rect, border_radius=4)
            pygame.draw.rect(screen, (255, 255, 255), agent_rect, 1, border_radius=4)
        
        train_step += TRAINING_SPEED
        if train_step >= TRAIN_TIME:
            TRAINING = False
            for ai in ais:
                ai.race()
            ais.sort(key=lambda a: a.steps if a.steps else 999999)
        
        draw_sidebar_training(train_step)
    else:
        for ai in ais:
            draw_trail(ai.path, ai.color, frame)
        
        max_path = max((len(ai.path) for ai in ais), default=1)
        if frame < max_path:
            frame += RACE_SPEED  # Advance by RACE_SPEED frames
        
        draw_sidebar_race(frame)
    
    # Restore original screen
    screen = temp_screen
    
    # Scale and blit the game surface to the actual screen
    if scale_x != 1.0 or scale_y != 1.0:
        scaled_w = int(WIDTH * scale_x)
        scaled_h = int(HEIGHT * scale_y)
        scaled_surface = pygame.transform.smoothscale(game_surface, (scaled_w, scaled_h))
        screen.blit(scaled_surface, (0, 0))
    else:
        screen.blit(game_surface, (0, 0))
    
    pygame.display.flip()

pygame.quit()