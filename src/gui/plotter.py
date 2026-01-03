"""Matplotlib visualization for sensor dashboard"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import numpy as np


class DashboardPlotter:
    """Create and update matplotlib dashboard"""
    
    def __init__(self, config, buffers):
        self.config = config
        self.buffers = buffers
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.stats_text = None
        self.start_time = datetime.now()
        self.frame_count = 0
    
    def setup_figure(self, title):
        """Setup complete matplotlib figure with all subplots"""
        fig_size = self.config.get('ui', 'figure_size', default=[18, 14])
        self.fig = plt.figure(figsize=fig_size)
        self.fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create 4x3 grid
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # Row 1: Top IMU
        self._setup_row1_top_imu(gs)
        
        # Row 2: Magnetometer (full width)
        self._setup_row2_magnetometer(gs)
        
        # Row 3: Rear IMU
        self._setup_row3_rear_imu(gs)
        
        # Row 4: Force & Status
        self._setup_row4_force_status(gs)
        
        return self.fig
    
    def _setup_row1_top_imu(self, gs):
        """Setup Row 1: Top IMU plots"""
        ax_accel = self.fig.add_subplot(gs[0, 0])
        ax_gyro = self.fig.add_subplot(gs[0, 1])
        ax_mag = self.fig.add_subplot(gs[0, 2])
        
        # Accelerometer
        ax_accel.set_title('TOP IMU: Accelerometer (LSM6DSO)', fontweight='bold', fontsize=10)
        self.lines['top_ax'] = ax_accel.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)[0]
        self.lines['top_ay'] = ax_accel.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)[0]
        self.lines['top_az'] = ax_accel.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)[0]
        ax_accel.legend(loc='upper right', fontsize=8)
        ax_accel.set_ylabel('Accel (m/s²)', fontsize=9)
        ax_accel.grid(True, alpha=0.3)
        ax_accel.set_ylim(-8000, 8000)
        
        # Gyroscope
        ax_gyro.set_title('TOP IMU: Gyroscope (LSM6DSO)', fontweight='bold', fontsize=10)
        self.lines['top_gx'] = ax_gyro.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)[0]
        self.lines['top_gy'] = ax_gyro.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)[0]
        self.lines['top_gz'] = ax_gyro.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)[0]
        ax_gyro.legend(loc='upper right', fontsize=8)
        ax_gyro.set_ylabel('Gyro (°/s)', fontsize=9)
        ax_gyro.grid(True, alpha=0.3)
        ax_gyro.set_ylim(-2000, 2000)
        
        # Magnitude
        ax_mag.set_title('TOP: Accel Magnitude', fontweight='bold', fontsize=10)
        self.lines['top_mag'] = ax_mag.plot([], [], 'purple', linewidth=2)[0]
        ax_mag.set_ylabel('|a| (m/s²)', fontsize=9)
        ax_mag.grid(True, alpha=0.3)
        ax_mag.set_ylim(0, 10000)
        
        self.axes['top_accel'] = ax_accel
        self.axes['top_gyro'] = ax_gyro
        self.axes['top_mag'] = ax_mag
    
    def _setup_row2_magnetometer(self, gs):
        """Setup Row 2: Magnetometer (full width)"""
        ax_mag = self.fig.add_subplot(gs[1, :])
        ax_mag.set_title('MAGNETOMETER: Magnetic Field (ST)', fontweight='bold', fontsize=10)
        self.lines['mag_x'] = ax_mag.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)[0]
        self.lines['mag_y'] = ax_mag.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)[0]
        self.lines['mag_z'] = ax_mag.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)[0]
        ax_mag.legend(loc='upper right', fontsize=9)
        ax_mag.set_ylabel('Mag Field (μT)', fontsize=9)
        ax_mag.grid(True, alpha=0.3)
        ax_mag.set_ylim(-10000, 10000)
        self.axes['mag'] = ax_mag
    
    def _setup_row3_rear_imu(self, gs):
        """Setup Row 3: Rear IMU plots"""
        ax_accel = self.fig.add_subplot(gs[2, 0])
        ax_gyro = self.fig.add_subplot(gs[2, 1])
        ax_mag = self.fig.add_subplot(gs[2, 2])
        
        # Accelerometer
        ax_accel.set_title('REAR IMU: Accelerometer (LSM6DSM)', fontweight='bold', fontsize=10)
        self.lines['rear_ax'] = ax_accel.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)[0]
        self.lines['rear_ay'] = ax_accel.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)[0]
        self.lines['rear_az'] = ax_accel.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)[0]
        ax_accel.legend(loc='upper right', fontsize=8)
        ax_accel.set_ylabel('Accel (m/s²)', fontsize=9)
        ax_accel.grid(True, alpha=0.3)
        ax_accel.set_ylim(-8000, 8000)
        
        # Gyroscope
        ax_gyro.set_title('REAR IMU: Gyroscope (LSM6DSM)', fontweight='bold', fontsize=10)
        self.lines['rear_gx'] = ax_gyro.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)[0]
        self.lines['rear_gy'] = ax_gyro.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)[0]
        self.lines['rear_gz'] = ax_gyro.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)[0]
        ax_gyro.legend(loc='upper right', fontsize=8)
        ax_gyro.set_ylabel('Gyro (°/s)', fontsize=9)
        ax_gyro.grid(True, alpha=0.3)
        ax_gyro.set_ylim(-2000, 2000)
        
        # Magnitude
        ax_mag.set_title('REAR: Accel Magnitude', fontweight='bold', fontsize=10)
        self.lines['rear_mag'] = ax_mag.plot([], [], 'orange', linewidth=2)[0]
        ax_mag.set_ylabel('|a| (m/s²)', fontsize=9)
        ax_mag.grid(True, alpha=0.3)
        ax_mag.set_ylim(0, 10000)
        
        self.axes['rear_accel'] = ax_accel
        self.axes['rear_gyro'] = ax_gyro
        self.axes['rear_mag'] = ax_mag
    
    def _setup_row4_force_status(self, gs):
        """Setup Row 4: Force sensor and status panel"""
        ax_force = self.fig.add_subplot(gs[3, 0:2])
        ax_stats = self.fig.add_subplot(gs[3, 2])
        
        # Force sensor
        ax_force.set_title('FORCE SENSOR: 3-Axis (HLP A04)', fontweight='bold', fontsize=10)
        self.lines['force_x'] = ax_force.plot([], [], 'r-', linewidth=1.5, label='Fx', alpha=0.8)[0]
        self.lines['force_y'] = ax_force.plot([], [], 'g-', linewidth=1.5, label='Fy', alpha=0.8)[0]
        self.lines['force_z'] = ax_force.plot([], [], 'b-', linewidth=1.5, label='Fz', alpha=0.8)[0]
        ax_force.legend(loc='upper right', fontsize=9)
        ax_force.set_xlabel('Sample Index', fontsize=9)
        ax_force.set_ylabel('Force (ADC)', fontsize=9)
        ax_force.grid(True, alpha=0.3)
        ax_force.set_ylim(0, 4096)
        
        # Status panel
        ax_stats.set_title('STATUS', fontweight='bold', fontsize=10)
        ax_stats.axis('off')
        self.stats_text = ax_stats.text(0.05, 0.95, '', transform=ax_stats.transAxes,
                                       verticalalignment='top', fontfamily='monospace',
                                       fontsize=8, bbox=dict(boxstyle='round', 
                                       facecolor='wheat', alpha=0.5))
        
        self.axes['force'] = ax_force
        self.axes['stats'] = ax_stats
    
    def update(self, frame, data_source=''):
        """Update all plots with current buffer data"""
        if len(self.buffers.timestamps) < 2:
            return
        
        # Get thread-safe snapshot as lists
        data = self.buffers.get_data_arrays()
        n_samples = len(data['x_data'])
        
        if n_samples == 0:
            return
        
        x_indices = np.arange(n_samples)  # Use numpy array
        buffer_size = self.buffers.buffer_size
        x_min = max(0, n_samples - buffer_size)
        
        # Update Top IMU
        self.lines['top_ax'].set_data(x_indices, data['top_accel'][0])
        self.lines['top_ay'].set_data(x_indices, data['top_accel'][1])
        self.lines['top_az'].set_data(x_indices, data['top_accel'][2])
        self.axes['top_accel'].set_xlim(x_min, n_samples)
        
        self.lines['top_gx'].set_data(x_indices, data['top_gyro'][0])
        self.lines['top_gy'].set_data(x_indices, data['top_gyro'][1])
        self.lines['top_gz'].set_data(x_indices, data['top_gyro'][2])
        self.axes['top_gyro'].set_xlim(x_min, n_samples)
        
        # Use numpy for magnitude
        top_accel_arr = np.column_stack([data['top_accel'][0], data['top_accel'][1], data['top_accel'][2]])
        top_mag = np.linalg.norm(top_accel_arr, axis=1)
        self.lines['top_mag'].set_data(x_indices, top_mag)
        self.axes['top_mag'].set_xlim(x_min, n_samples)
        
        # Update Magnetometer
        self.lines['mag_x'].set_data(x_indices, data['mag'][0])
        self.lines['mag_y'].set_data(x_indices, data['mag'][1])
        self.lines['mag_z'].set_data(x_indices, data['mag'][2])
        self.axes['mag'].set_xlim(x_min, n_samples)
        
        # Update Rear IMU
        self.lines['rear_ax'].set_data(x_indices, data['rear_accel'][0])
        self.lines['rear_ay'].set_data(x_indices, data['rear_accel'][1])
        self.lines['rear_az'].set_data(x_indices, data['rear_accel'][2])
        self.axes['rear_accel'].set_xlim(x_min, n_samples)
        
        self.lines['rear_gx'].set_data(x_indices, data['rear_gyro'][0])
        self.lines['rear_gy'].set_data(x_indices, data['rear_gyro'][1])
        self.lines['rear_gz'].set_data(x_indices, data['rear_gyro'][2])
        self.axes['rear_gyro'].set_xlim(x_min, n_samples)
        
        rear_accel_arr = np.column_stack([data['rear_accel'][0], data['rear_accel'][1], data['rear_accel'][2]])
        rear_mag = np.linalg.norm(rear_accel_arr, axis=1)
        self.lines['rear_mag'].set_data(x_indices, rear_mag)
        self.axes['rear_mag'].set_xlim(x_min, n_samples)
        
        # Update Force
        self.lines['force_x'].set_data(x_indices, data['force'][0])
        self.lines['force_y'].set_data(x_indices, data['force'][1])
        self.lines['force_z'].set_data(x_indices, data['force'][2])
        self.axes['force'].set_xlim(x_min, n_samples)
        
        # Update status text
        self._update_status_text(data, top_mag, rear_mag, data_source)
        
        self.frame_count += 1
    
    def _update_status_text(self, data, top_mag, rear_mag, data_source):
        """Update status panel text"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.buffers.sample_count / elapsed if elapsed > 0 else 0
        
        # Get latest values safely - direct index access, no list conversion
        def safe_last(arr, default=0):
            return arr[-1] if len(arr) > 0 else default
        
        stats = f"""DATA SOURCE
{data_source}

SAMPLES: {self.buffers.sample_count}
ELAPSED: {elapsed:.1f}s
RATE: {rate:.1f} Hz
TARGET: 104 Hz

TOP ACCEL
X: {safe_last(data['top_accel'][0]):.0f}
Y: {safe_last(data['top_accel'][1]):.0f}
Z: {safe_last(data['top_accel'][2]):.0f}
|a|: {safe_last(top_mag):.0f}

REAR ACCEL
X: {safe_last(data['rear_accel'][0]):.0f}
Y: {safe_last(data['rear_accel'][1]):.0f}
Z: {safe_last(data['rear_accel'][2]):.0f}
|a|: {safe_last(rear_mag):.0f}

FORCE
X: {safe_last(data['force'][0]):.0f}
Y: {safe_last(data['force'][1]):.0f}
Z: {safe_last(data['force'][2]):.0f}

STATUS: ✓ LIVE
"""
        self.stats_text.set_text(stats)
