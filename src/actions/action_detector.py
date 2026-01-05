"""
Action Detection System for Pen/Stylus Movement
Detects writing, thinking, tilt, and drift from IMU sensor data
"""

import numpy as np
from collections import deque
from datetime import datetime


class ActionDetector:
    """Detects user actions based on sensor data"""

    def __init__(self, buffer_size=30):
        """
        Initialize action detector

        Args:
            buffer_size: Number of samples to use for action detection
        """
        self.buffer_size = buffer_size

        # Buffers for moving window analysis
        self.accel_buffer = deque(maxlen=buffer_size)
        self.gyro_buffer = deque(maxlen=buffer_size)

        # Thresholds for action detection
        self.accel_motion_threshold = 500  # m/s² - movement threshold (lowered from 800)
        self.gyro_motion_threshold = 150  # °/s - rotation threshold (lowered from 300)
        self.tilt_threshold = 30  # degrees - significant tilt (raised)
        self.drift_threshold = 30  # degrees - gyro drift (raised)
        self.pen_up_threshold = 11000  # m/s² - pen lifted (high accel)
        self.pen_down_threshold = 2000  # m/s² - pen touching surface (very low accel)
        self.firmly_hold_threshold = 100  # Very low motion for firm hold
        self.force_threshold = 1000  # Force sensor value indicating grip (lowered from 1500)

        # State tracking
        self.current_action = "idle"
        self.action_start_time = None
        self.action_confidence = 0.0
        self.tilt_angle = 0.0
        self.drift_rate = 0.0

        # For drift calculation (integrate gyro)
        self.last_gyro = [0, 0, 0]
        self.integrated_gyro = [0, 0, 0]
        self.dt = 1 / 208.0  # 208 Hz sampling rate

        # Force sensor buffer
        self.force_buffer = deque(maxlen=buffer_size)
        self.has_valid_data = False  # Track if we've received real data

    def update(self, top_accel, rear_accel, top_gyro, rear_gyro, mag, force=None):
        """
        Update action detection with latest sensor data

        Args:
            top_accel: dict with 'x', 'y', 'z' accelerometer from top IMU
            rear_accel: dict with 'x', 'y', 'z' accelerometer from rear IMU
            top_gyro: dict with 'x', 'y', 'z' gyroscope from top IMU
            rear_gyro: dict with 'x', 'y', 'z' gyroscope from rear IMU
            mag: dict with 'x', 'y', 'z' magnetometer
            force: dict with 'x', 'y', 'z' force sensor values
        """
        # Store in buffers
        self.accel_buffer.append(top_accel)
        self.gyro_buffer.append(top_gyro)

        if force is not None:
            self.force_buffer.append(force)

        # Mark that we have received data
        if not self.has_valid_data:
            accel_sum = abs(top_accel.get("x", 0)) + abs(top_accel.get("y", 0)) + abs(top_accel.get("z", 0))
            if accel_sum > 100:  # Real data, not zeros
                self.has_valid_data = True

        # Update drift (integrate gyro)
        if len(self.gyro_buffer) > 0:
            self._update_drift(top_gyro)

        # Calculate tilt from accelerometer
        self._update_tilt(top_accel, mag)

        # Detect action based on motion patterns
        self._detect_action()

    def _update_drift(self, gyro):
        """Calculate gyro drift through integration"""
        gx = (gyro.get("x", 0) + self.last_gyro[0]) / 2.0
        gy = (gyro.get("y", 0) + self.last_gyro[1]) / 2.0
        gz = (gyro.get("z", 0) + self.last_gyro[2]) / 2.0

        # Integrate using trapezoidal rule
        self.integrated_gyro[0] += gx * self.dt
        self.integrated_gyro[1] += gy * self.dt
        self.integrated_gyro[2] += gz * self.dt

        # Reset drift if it gets too large (prevent infinite accumulation)
        for i in range(3):
            if abs(self.integrated_gyro[i]) > 360:
                self.integrated_gyro[i] = 0  # Reset to prevent runaway

        self.last_gyro = [gyro.get("x", 0), gyro.get("y", 0), gyro.get("z", 0)]

        # Drift rate is magnitude of integrated gyro
        self.drift_rate = np.sqrt(sum(x**2 for x in self.integrated_gyro))

    def _update_tilt(self, accel, mag):
        """Calculate tilt angle from accelerometer and magnetometer"""
        ax = accel.get("x", 0)
        ay = accel.get("y", 0)
        az = accel.get("z", 0)

        # Calculate magnitude to normalize
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)

        if accel_mag < 1000:  # Too low, likely noise
            self.tilt_angle = 0
            return

        # Normalize to remove gravity magnitude effect
        ax_norm = ax / accel_mag
        ay_norm = ay / accel_mag
        az_norm = az / accel_mag

        # Tilt is deviation from vertical (z-axis)
        # When vertical, az_norm should be close to ±1
        tilt_from_vertical = np.degrees(np.arccos(min(1.0, abs(az_norm))))
        self.tilt_angle = tilt_from_vertical

    def _detect_action(self):
        """Detect current action based on motion patterns"""
        # Check if we have valid data
        if not self.has_valid_data or len(self.accel_buffer) < 5:
            self.current_action = "idle"
            self.action_confidence = 0.0
            return

        # Calculate motion metrics
        accel_motion = self._calculate_motion(self.accel_buffer)
        gyro_motion = self._calculate_motion(self.gyro_buffer)

        # Get current accelerometer magnitude
        latest_accel = list(self.accel_buffer)[-1]
        accel_magnitude = np.sqrt(
            latest_accel.get("x", 0) ** 2 + latest_accel.get("y", 0) ** 2 + latest_accel.get("z", 0) ** 2
        )

        # Get force sensor data if available
        force_magnitude = 0
        has_force_data = len(self.force_buffer) > 0
        if has_force_data:
            latest_force = list(self.force_buffer)[-1]
            force_magnitude = np.sqrt(
                latest_force.get("x", 0) ** 2 + latest_force.get("y", 0) ** 2 + latest_force.get("z", 0) ** 2
            )

        # Motion score (0-100)
        accel_score = min(100, (accel_motion / self.accel_motion_threshold) * 50)
        gyro_score = min(100, (gyro_motion / self.gyro_motion_threshold) * 50)
        motion_score = accel_score + gyro_score

        # Debug: Show motion scores every 50 updates to diagnose why writing isn't triggering
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1

        if self._debug_counter % 50 == 0 and has_force_data:
            print(
                f"🔍 Motion Scores | accel={accel_score:.1f} gyro={gyro_score:.1f} total={motion_score:.1f} | force={force_magnitude:.0f} | accel_mag={accel_magnitude:.0f}"
            )

        # Debug: Show action transitions
        if not hasattr(self, "_prev_action"):
            self._prev_action = None

        # Pre-evaluate what action will be chosen
        if accel_magnitude > self.pen_up_threshold and motion_score > 30:
            potential_action = "pen_up"
        elif (motion_score > 50 and gyro_score > 10) or (
            has_force_data and force_magnitude > 1000 and motion_score > 5
        ):
            potential_action = "writing"
        elif has_force_data and force_magnitude > 1200 and motion_score < 40 and accel_magnitude > 2500:
            potential_action = "pen_down"
        elif (
            has_force_data
            and force_magnitude > self.force_threshold
            and accel_magnitude >= 3000
            and motion_score < 50
            and motion_score > 5
        ):
            potential_action = "firmly_held"
        elif motion_score < 20 and accel_magnitude > 1000:
            potential_action = "thinking"
        elif self.tilt_angle > self.tilt_threshold and accel_magnitude > 1000:
            potential_action = "tilted"
        elif self.drift_rate > self.drift_threshold and accel_magnitude > 1000:
            potential_action = "drifting"
        else:
            potential_action = "idle"

        if potential_action != self._prev_action:
            print(
                f"   [ACTION TRANSITION] {self._prev_action} → {potential_action} (motion={motion_score:.0f}, force={force_magnitude:.0f})"
            )
            self._prev_action = potential_action
        elif potential_action == "writing" and self._debug_counter % 100 == 0:
            # Show why still in writing every 100 samples
            print(
                f"   📝 Still WRITING: motion={motion_score:.0f} force={force_magnitude:.0f} (need motion<30 to exit to firmly_held)"
            )

        # Decision logic - check in priority order
        # 1. Check for extreme acceleration (pen up - only if very high AND motion detected)
        if accel_magnitude > self.pen_up_threshold and motion_score > 30:
            self.current_action = "pen_up"
            self.action_confidence = min(100, (accel_magnitude / 15000) * 100)
        # 2. WRITING with force applied - active motion OR force sensor + accel
        elif (motion_score > 50 and gyro_score > 10) or (
            has_force_data and force_magnitude > 1000 and motion_score > 30
        ):
            self.current_action = "writing"
            self.action_confidence = min(100, motion_score if motion_score > 50 else (force_magnitude / 2500) * 100)
            if self._debug_counter % 20 == 0:
                print(
                    f"✍️ WRITING! Condition: motion={motion_score:.0f}>{50} gyro={gyro_score:.0f}>{10} OR (force={force_magnitude:.0f}>{1000} AND motion={motion_score:.0f}>{30})"
                )
        # 3. Pen down with force - force applied + minimal motion
        elif has_force_data and force_magnitude > 1200 and motion_score < 40 and accel_magnitude > 2500:
            self.current_action = "pen_down"
            self.action_confidence = min(100, (force_magnitude / 2500) * 100)
        # 4. Firmly held - based on FORCE SENSOR (high force + normal accel + low motion)
        elif (
            has_force_data
            and force_magnitude > self.force_threshold
            and accel_magnitude >= 3000
            and motion_score < 50
            and motion_score > 5
        ):
            self.current_action = "firmly_held"
            self.action_confidence = min(100, (force_magnitude / 2500) * 100)
            if self._debug_counter % 30 == 0:
                print(
                    f"🤚 FIRMLY_HELD (blocking WRITING!) | force={force_magnitude:.0f}>{self.force_threshold} accel={accel_magnitude:.0f}>=3000 motion={motion_score:.0f} (5-50 range)"
                )
        # 5. Thinking - low motion, pen steady, but with some minimal sensor activity
        elif motion_score < 20 and accel_magnitude > 1000:
            self.current_action = "thinking"
            self.action_confidence = 100 - motion_score
        # 6. Significant tilt - only if pronounced
        elif self.tilt_angle > self.tilt_threshold and accel_magnitude > 1000:
            self.current_action = "tilted"
            self.action_confidence = min(100, (self.tilt_angle / 60.0) * 100)
        # 7. Significant drift - only if pronounced
        elif self.drift_rate > self.drift_threshold and accel_magnitude > 1000:
            self.current_action = "drifting"
            self.action_confidence = min(100, (self.drift_rate / 90.0) * 100)
        # 8. Default to idle (includes very low sensor readings)
        else:
            self.current_action = "idle"
            self.action_confidence = 0.0
            self.action_confidence = 0.0

    def _calculate_motion(self, buffer):
        """Calculate motion variance from sensor buffer"""
        if len(buffer) < 2:
            return 0

        values = []
        for sample in buffer:
            x = sample.get("x", 0)
            y = sample.get("y", 0)
            z = sample.get("z", 0)
            magnitude = np.sqrt(x**2 + y**2 + z**2)
            values.append(magnitude)

        # Calculate variance as indicator of motion
        variance = np.var(values)
        # Also consider mean magnitude
        mean_magnitude = np.mean(values)

        # Motion score combines variance and magnitude
        motion = np.sqrt(variance) + (mean_magnitude * 0.1)
        return motion

    def get_action(self):
        """Get current detected action"""
        return self.current_action

    def get_details(self):
        """Get detailed action information"""
        return {
            "action": self.current_action,
            "confidence": self.action_confidence,
            "tilt": self.tilt_angle,
            "drift": self.drift_rate,
        }

    def get_display_text(self):
        """Get formatted text for display"""
        action_icons = {
            "writing": "✏️",
            "thinking": "💭",
            "firmly_held": "✊",
            "pen_up": "⬆️",
            "pen_down": "⬇️",
            "tilted": "⚠️",
            "drifting": "↗️",
            "idle": "⊘",
        }

        icon = action_icons.get(self.current_action, "?")

        # Larger, clearer display format
        text = f"""
{icon} {self.current_action.upper().replace('_', ' ')}

Confidence:
{self.action_confidence:.0f}%

Tilt: {self.tilt_angle:.1f}°
Drift: {self.drift_rate:.1f}°

"""

        if self.current_action == "writing":
            text += "Pen is\nwriting"
        elif self.current_action == "thinking":
            text += "Pen held\nstill"
        elif self.current_action == "firmly_held":
            text += "Pen firmly\nheld"
        elif self.current_action == "pen_up":
            text += "Pen lifted\nup"
        elif self.current_action == "pen_down":
            text += "Pen on\nsurface"
        elif self.current_action == "tilted":
            text += f"High tilt\ndetected!"
        elif self.current_action == "drifting":
            text += f"Gyro drift\ndetected!"
        else:
            text += "Waiting\nfor input..."

        return text

    def reset(self):
        """Reset action detector"""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.force_buffer.clear()
        self.current_action = "idle"
        self.action_confidence = 0.0
        self.tilt_angle = 0.0
        self.drift_rate = 0.0
        self.integrated_gyro = [0, 0, 0]
        self.last_gyro = [0, 0, 0]
        self.has_valid_data = False
