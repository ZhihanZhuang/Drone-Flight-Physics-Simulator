import math

class DronePhysics:
    """
    Core physics engine for 1D vertical drone flight simulation.
    Handles Newtonian mechanics, thrust mapping, and battery depletion.
    """
    def __init__(self, mass, motor_k, battery_capacity_mah, max_rpm=15000, voltage=11.1):
        # Configuration
        self.mass = mass
        self.motor_k = motor_k
        self.max_rpm = max_rpm
        self.voltage = voltage
        self.battery_capacity = battery_capacity_mah
        
        # Energy Conversion: mAh to Joules (Amp-hours * Voltage * 3600 seconds)
        self.energy = (battery_capacity_mah / 1000.0) * voltage * 3600.0
        self.initial_energy = self.energy
        
        # State variables
        self.h = 0.0          # Height (m)
        self.v = 0.0          # Velocity (m/s)
        self.time = 0.0       # Time (s)
        self.dt = 0.1         # Timestep
        self.g = 9.81         # Gravity
        
        # Max theoretical thrust based on motor constant and max RPM
        self.max_thrust = self.motor_k * (self.max_rpm ** 2)
        
        # Binary feasibility: Can the drone lift its own weight?
        self.feasible = self.max_thrust > (self.mass * self.g)

    def step(self, target_h=15.0):
        """
        Advances the simulation by one timestep dt.
        Returns False if the simulation should stop (crashed or battery dead).
        """
        if not self.feasible or self.energy <= 0:
            return False

        # --- Control System (Simple PD Controller for Altitude Hover) ---
        error = target_h - self.h
        # Calculate desired acceleration to reach target height
        desired_accel = 0.8 * error - 0.4 * self.v 

        # Determine required thrust to achieve desired acceleration
        desired_thrust = self.mass * (self.g + desired_accel)
        
        # Clamp thrust to motor physical limits
        actual_thrust = max(0, min(desired_thrust, self.max_thrust))

        # --- Power Model ---
        # RPM = sqrt(Thrust / k)
        # Power (aerodynamic approximation) P ∝ Thrust^1.5
        power_draw = 0.08 * (actual_thrust ** 1.5)

        # --- Newtonian Physics Update ---
        f_net = actual_thrust - (self.mass * self.g)
        a = f_net / self.mass

        self.v += a * self.dt
        self.h += self.v * self.dt
        self.energy -= power_draw * self.dt
        self.time += self.dt

        # --- Stop Conditions ---
        if self.h < -0.1 and self.time > 2.0: 
            return False # Crashed back to ground
        if self.energy <= 0:
            return False # Battery Depleted
        if self.time > 7200: 
            return False # Max simulation time cap (2 hours)

        return True