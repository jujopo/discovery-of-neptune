import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Constants ---
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2) -> scaled for AU/days
G_SCALED = 2.959e-4  # G in AU^3 / (M_sun * day^2)
M_SUN = 1.989e30 # kg

# --- Planet Data ---
# Data format: [mass (in solar masses), semi-major axis (AU), eccentricity, orbital period (days), color]
# Orbital data is approximate for this toy model.
PLANETS_A = {
    'Sun': [1.0, 0, 0, 0, 'yellow'],
    'Jupiter': [9.547e-4, 5.203, 0.048, 4332, 'orange'],
    'Saturn': [2.858e-4, 9.537, 0.054, 10759, 'tan'],
    'Uranus': [4.366e-5, 19.191, 0.047, 30688, 'cyan']
}

PLANETS_B = {
    'Sun': [1.0, 0, 0, 0, 'yellow'],
    'Jupiter': [9.547e-4, 5.203, 0.048, 4332, 'orange'],
    'Saturn': [2.858e-4, 9.537, 0.054, 10759, 'tan'],
    'Uranus': [4.366e-5, 19.191, 0.047, 30688, 'cyan'],
    'Neptune': [5.151e-5, 30.069, 0.008, 60182, 'blue']
}

class NBodySimulation:
    """
    A class to handle the N-body simulation using velocity-Verlet integration.
    """
    def __init__(self, bodies_data):
        self.bodies_data = bodies_data
        self.num_bodies = len(bodies_data)
        self.names = list(bodies_data.keys())
        self.masses = np.array([data[0] for data in bodies_data.values()]).reshape(self.num_bodies, 1)
        self.colors = [data[4] for data in bodies_data.values()]

        # Initialize positions and velocities
        self.positions = np.zeros((self.num_bodies, 2))
        self.velocities = np.zeros((self.num_bodies, 2))

        for i, (name, data) in enumerate(bodies_data.items()):
            if name == 'Sun':
                continue
            a = data[1] # semi-major axis
            e = data[2] # eccentricity
            # Start at perihelion for simplicity
            self.positions[i, 0] = a * (1 - e)
            # Velocity at perihelion: v = sqrt( (G*M_sun/a) * (1+e)/(1-e) )
            self.velocities[i, 1] = np.sqrt(G_SCALED * self.masses[0] * (1 + e) / (a * (1 - e)))

    def calculate_accelerations(self):
        """Calculates acceleration on each body due to gravitational forces from all others."""
        accelerations = np.zeros_like(self.positions)
        for i in range(self.num_bodies):
            for j in range(self.num_bodies):
                if i == j:
                    continue
                # Vector from body i to body j
                r_ij = self.positions[j] - self.positions[i]
                # Distance squared
                dist_sq = np.sum(r_ij**2)
                # Magnitude of distance
                dist = np.sqrt(dist_sq)
                # Gravitational force formula component
                force_mag = G_SCALED * self.masses[j] / (dist_sq * dist)
                # Accumulate acceleration
                accelerations[i] += force_mag * r_ij
        return accelerations

    def update(self, dt):
        """
        Updates positions and velocities using the velocity-Verlet algorithm.
        """
        # Initial half-step for velocity
        accelerations = self.calculate_accelerations()
        self.velocities += 0.5 * accelerations * dt

        # Full-step for position
        self.positions += self.velocities * dt

        # Calculate new accelerations at the new position
        new_accelerations = self.calculate_accelerations()

        # Final half-step for velocity
        self.velocities += 0.5 * new_accelerations * dt

def run_simulation():
    """
    Runs the full simulation for both scenarios and prepares data for plotting.
    """
    # Simulation parameters
    dt = 50.0  # Time step in days
    years = 250
    steps = int(years * 365.25 / dt)

    # Initialize simulations
    sim_A = NBodySimulation(PLANETS_A) # Without Neptune
    sim_B = NBodySimulation(PLANETS_B) # With Neptune

    # Store history for plotting
    history_A = np.zeros((steps, sim_A.num_bodies, 2))
    history_B = np.zeros((steps, sim_B.num_bodies, 2))

    # Run integration loop
    for i in range(steps):
        history_A[i] = sim_A.positions
        history_B[i] = sim_B.positions
        sim_A.update(dt)
        sim_B.update(dt)

    return history_A, history_B, dt, years

# --- Run the simulation to get data ---
history_A, history_B, dt, total_years = run_simulation()

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("The Influence of Neptune on Uranus's Orbit", fontsize=16)

# --- Panel 1: Orbit Animation ---
ax1.set_aspect('equal')
ax1.set_xlim(-35, 35)
ax1.set_ylim(-35, 35)
ax1.set_facecolor('black')
ax1.set_xlabel('X (AU)')
ax1.set_ylabel('Y (AU)')
ax1.set_title('Planetary Orbits')
ax1.grid(True, linestyle='--', alpha=0.3)

# Placeholders for animated elements
paths_A = [ax1.plot([], [], '-', color=c, linewidth=0.5, alpha=0.7)[0] for c in NBodySimulation(PLANETS_A).colors]
planets_A = [ax1.plot([], [], 'o', color=c, markersize=ms)[0] for c, ms in zip(NBodySimulation(PLANETS_A).colors, [10, 5, 4, 3])]
planets_B = [ax1.plot([], [], 'o', color=c, markersize=ms, markerfacecolor='none', markeredgewidth=1.0)[0] for c, ms in zip(NBodySimulation(PLANETS_B).colors, [10, 5, 4, 3, 3])]
legend_labels = list(PLANETS_B.keys())
ax1.legend(planets_B, [f"{name} (with Neptune)" for name in legend_labels], loc='upper right', fontsize='small')

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', fontsize=12)

# --- Panel 2: Longitude Residual Plot ---
uranus_index_A = list(PLANETS_A.keys()).index('Uranus')
uranus_index_B = list(PLANETS_B.keys()).index('Uranus')

# Calculate angular position (longitude) for Uranus in both simulations
lon_A = np.arctan2(history_A[:, uranus_index_A, 1], history_A[:, uranus_index_A, 0])
lon_B = np.arctan2(history_B[:, uranus_index_B, 1], history_B[:, uranus_index_B, 0])

# Unwrap angle to handle the jump from +pi to -pi
lon_A = np.unwrap(lon_A)
lon_B = np.unwrap(lon_B)

# Calculate the residual in arcseconds (1 radian = 206265 arcseconds)
residuals = (lon_B - lon_A) * 206265
time_years = np.arange(0, total_years, dt / 365.25)

ax2.set_xlim(0, total_years)
ax2.set_ylim(min(residuals) * 1.1, max(residuals) * 1.1)
ax2.set_xlabel('Time (Years)')
ax2.set_ylabel('Uranus Longitude Residual (arcseconds)')
ax2.set_title('Uranus Discrepancy: With vs. Without Neptune')
ax2.grid(True, linestyle='--', alpha=0.5)
residual_line, = ax2.plot([], [], 'r-')
residual_point, = ax2.plot([], [], 'ro') # To mark the current time


def animate(i):
    """Animation function, called for each frame."""
    frame_skip = 10 # Speed up animation
    frame_idx = i * frame_skip
    if frame_idx >= len(history_A):
        # When using blit=True, it's important to return an iterable of artists.
        # An empty list is fine if no updates are made.
        return []

    # Update orbits plot
    for j in range(len(paths_A)):
        paths_A[j].set_data(history_A[:frame_idx, j, 0], history_A[:frame_idx, j, 1])
        # For single points, set_data expects a sequence (like a list) for x and y
        planets_A[j].set_data([history_A[frame_idx, j, 0]], [history_A[frame_idx, j, 1]])

    for j in range(len(planets_B)):
        planets_B[j].set_data([history_B[frame_idx, j, 0]], [history_B[frame_idx, j, 1]])

    year = time_years[frame_idx]
    time_text.set_text(f'Year: {year:.1f}')

    # Update residuals plot
    residual_line.set_data(time_years[:frame_idx], residuals[:frame_idx])
    residual_point.set_data([time_years[frame_idx]], [residuals[frame_idx]])

    return paths_A + planets_A + planets_B + [time_text, residual_line, residual_point]


# Create the animation
num_frames = len(history_A) // 10
ani = FuncAnimation(fig, animate, frames=num_frames, interval=20, blit=True, repeat=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
