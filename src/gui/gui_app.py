#!/usr/bin/env python3
"""
Sensor Fusion Dashboard GUI Application
Professional multi-sensor real-time visualization with GUI controls
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import threading
import serial
import csv
import time
import os
from datetime import datetime

from core.serial_config import SerialConfig
from utils.sensor_buffers import SensorBuffers
from core.sensor_parser import SensorParser
from core.data_reader import CSVReader, SerialReader
from utils.sensor_fusion_filters import SensorFusionManager
from actions.action_detector import ActionDetector
from gui.character_recognition_integration import CharacterRecognitionIntegration


class SensorDashboardGUI:
    """Main GUI Application for Sensor Dashboard"""
    
    def __init__(self, root):
        """Initialize the GUI application"""
        self.root = root
        self.root.title("Sensor Fusion Dashboard (104Hz)")
        self.root.geometry("1600x900")
        
        # Application state
        self.running = False
        self.data_reader = None
        self.buffers = SensorBuffers(buffer_size=256)
        self.sample_count = 0
        self.start_time = None
        self.csv_file = None
        self.csv_reader = None
        
        # Sensor fusion filter
        self.filter_manager = SensorFusionManager(filter_type='none')
        self.apply_filter = False
        
        # Action detector
        self.action_detector = ActionDetector(buffer_size=30)
        
        # Character recognition integration (load model if it exists)
        model_path = 'src/ai/models/character_model.h5'
        if not os.path.exists(model_path):
            model_path = None  # Don't try to load if model doesn't exist
        
        self.char_recognition = CharacterRecognitionIntegration(
            model_path=model_path,
            buffer_size=512
        )
        
        # Track pen state for character recognition triggers
        self.prev_pen_action = "idle"
        
        # Create UI
        self._create_control_panel()
        self._create_plot_area()
        
        # Start animation loop
        self.update_plot()
    
    def _create_control_panel(self):
        """Create the top control panel with COM port and baudrate selection"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Left section - Status
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(left_frame, text="Status:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(left_frame, text="⊘ Disconnected", foreground="red", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.sample_label = ttk.Label(left_frame, text="Samples: 0", font=("Arial", 9))
        self.sample_label.pack(side=tk.LEFT, padx=20)
        
        self.rate_label = ttk.Label(left_frame, text="Rate: 0.0 Hz", font=("Arial", 9))
        self.rate_label.pack(side=tk.LEFT, padx=5)
        
        # Right section - Controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X)
        
        # Data Source Selection
        ttk.Label(right_frame, text="Source:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.source_var = tk.StringVar(value="serial")
        self.source_combo = ttk.Combobox(
            right_frame,
            textvariable=self.source_var,
            values=["serial", "csv"],
            state="readonly",
            width=10
        )
        self.source_combo.pack(side=tk.LEFT, padx=5)
        self.source_combo.bind('<<ComboboxSelected>>', self._on_source_changed)
        
        # CSV File Selection
        ttk.Label(right_frame, text="CSV:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.csv_entry = ttk.Entry(right_frame, width=20)
        self.csv_entry.pack(side=tk.LEFT, padx=5)
        self.csv_entry.insert(0, "../data/sample_sensor_data.csv")
        
        # COM Port Selection
        ttk.Label(right_frame, text="COM Port:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(
            right_frame,
            textvariable=self.port_var,
            state="readonly",
            width=12
        )
        self.port_combo.pack(side=tk.LEFT, padx=5)
        self._refresh_ports()
        
        # Refresh Ports Button
        ttk.Button(right_frame, text="↻", width=3, command=self._refresh_ports).pack(side=tk.LEFT, padx=2)
        
        # Baudrate Selection
        ttk.Label(right_frame, text="Baud:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.baudrate_var = tk.StringVar(value="115200")
        self.baudrate_combo = ttk.Combobox(
            right_frame,
            textvariable=self.baudrate_var,
            values=[str(b) for b in SerialConfig.get_standard_baudrates()],
            state="readonly",
            width=10
        )
        self.baudrate_combo.pack(side=tk.LEFT, padx=5)
        
        # Sensor Fusion Filter Selection
        ttk.Label(right_frame, text="Filter:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.filter_var = tk.StringVar(value="none")
        self.filter_combo = ttk.Combobox(
            right_frame,
            textvariable=self.filter_var,
            values=["none", "kf", "ekf", "cf"],
            state="readonly",
            width=10
        )
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind('<<ComboboxSelected>>', self._on_filter_changed)
        
        # Control Buttons
        self.connect_btn = ttk.Button(right_frame, text="Connect", command=self._connect)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.disconnect_btn = ttk.Button(right_frame, text="Disconnect", command=self._disconnect, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=2)
        
        # Hide CSV entry initially
        self.csv_entry.pack_forget()
    
    def _on_filter_changed(self, event=None):
        """Handle filter type change"""
        filter_type = self.filter_var.get()
        if filter_type == "none":
            self.apply_filter = False
            self.filter_manager.set_filter_type('none')
        else:
            self.apply_filter = True
            self.filter_manager.set_filter_type(filter_type)
            self.filter_manager.reset()
    
    def _on_source_changed(self, event=None):
        """Handle data source change"""
        source = self.source_var.get()
        
        if source == "csv":
            self.csv_entry.pack(side=tk.LEFT, padx=5)
            self.port_combo.pack_forget()
            self.port_combo.master.children[list(self.port_combo.master.children.keys())[0]].pack_forget()
        else:
            self.csv_entry.pack_forget()
            self.port_combo.pack(side=tk.LEFT, padx=5)
    
    def _create_plot_area(self):
        """Create the matplotlib plot area with action box"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right side - action insights (split into two sections)
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # AI Character Recognition Section
        ai_recognition_frame = ttk.LabelFrame(action_frame, text="AI CHARACTER RECOGNITION", padding=10)
        ai_recognition_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self.ai_recognition_canvas = tk.Canvas(ai_recognition_frame, bg="#e8f4f8", width=200, height=200, 
                                                highlightthickness=1, highlightbackground="#0066cc")
        self.ai_recognition_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.ai_recognition_text = self.ai_recognition_canvas.create_text(
            100, 100, text="Ready\n\nModel: Loaded" if self.char_recognition.model else "No Model",
            font=("Arial", 12, "bold"),
            fill="#0066cc",
            anchor="center",
            width=180
        )
        
        # Pen Action Insights Section
        pen_action_frame = ttk.LabelFrame(action_frame, text="PEN ACTION INSIGHTS", padding=10)
        pen_action_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.action_canvas = tk.Canvas(pen_action_frame, bg="white", width=200, height=200, 
                                       highlightthickness=1, highlightbackground="black")
        self.action_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.action_text = self.action_canvas.create_text(
            100, 100, text="Initializing...",
            font=("Arial", 12, "bold"),
            fill="black",
            anchor="center",
            width=180
        )
        
        # Create figure for plots
        self.fig = Figure(figsize=(11, 8), dpi=100)
        self.fig.suptitle('Multi-Sensor Real-Time Dashboard (104Hz)', fontsize=14, fontweight='bold')
        
        # Create grid layout
        gs = GridSpec(4, 3, figure=self.fig, hspace=0.35, wspace=0.3)
        
        # ROW 1: Top IMU
        self.ax1_accel = self.fig.add_subplot(gs[0, 0])
        self.ax1_gyro = self.fig.add_subplot(gs[0, 1])
        self.ax1_mag = self.fig.add_subplot(gs[0, 2])
        
        self._setup_accel_plot(self.ax1_accel, "TOP IMU: Accelerometer (LSM6DSO)")
        self._setup_gyro_plot(self.ax1_gyro, "TOP IMU: Gyroscope (LSM6DSO)")
        self._setup_mag_plot(self.ax1_mag, "TOP IMU: Accel Magnitude")
        
        # ROW 2: Magnetometer
        self.ax2_mag = self.fig.add_subplot(gs[1, :])
        self._setup_mag_field_plot(self.ax2_mag, "MAGNETOMETER: Magnetic Field (ST)")
        
        # ROW 3: Rear IMU
        self.ax3_accel = self.fig.add_subplot(gs[2, 0])
        self.ax3_gyro = self.fig.add_subplot(gs[2, 1])
        self.ax3_mag = self.fig.add_subplot(gs[2, 2])
        
        self._setup_accel_plot(self.ax3_accel, "REAR IMU: Accelerometer (LSM6DSM)")
        self._setup_gyro_plot(self.ax3_gyro, "REAR IMU: Gyroscope (LSM6DSM)")
        self._setup_mag_plot(self.ax3_mag, "REAR IMU: Accel Magnitude")
        
        # ROW 4: Force Sensor & Stats
        self.ax4_force = self.fig.add_subplot(gs[3, 0:2])
        self.ax4_stats = self.fig.add_subplot(gs[3, 2])
        
        self._setup_force_plot(self.ax4_force, "FORCE SENSOR: 3-Axis Force (HLP A04)")
        self.ax4_stats.axis('off')
        self.stats_text = self.ax4_stats.text(0.05, 0.95, '', transform=self.ax4_stats.transAxes,
                                             verticalalignment='top', fontfamily='monospace',
                                             fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Store main frame for later reference
        self.plot_frame = plot_frame
        self.action_frame = action_frame
        
        # Store line objects for updates
        self._create_line_objects()
    
    def _setup_accel_plot(self, ax, title):
        """Setup accelerometer plot"""
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel('m/s²', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-8000, 8000)
    
    def _setup_gyro_plot(self, ax, title):
        """Setup gyroscope plot"""
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel('°/s', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2000, 2000)
    
    def _setup_mag_plot(self, ax, title):
        """Setup magnitude plot"""
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel('|a| (m/s²)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10000)
    
    def _setup_mag_field_plot(self, ax, title):
        """Setup magnetic field plot"""
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel('μT', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-10000, 10000)
    
    def _setup_force_plot(self, ax, title):
        """Setup force sensor plot"""
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Force (ADC)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 4096)
    
    def _create_line_objects(self):
        """Create line objects for all plots"""
        # Top IMU
        self.line_top_ax, = self.ax1_accel.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)
        self.line_top_ay, = self.ax1_accel.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)
        self.line_top_az, = self.ax1_accel.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)
        self.ax1_accel.legend(loc='upper right', fontsize=8)
        
        self.line_top_gx, = self.ax1_gyro.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)
        self.line_top_gy, = self.ax1_gyro.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)
        self.line_top_gz, = self.ax1_gyro.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)
        self.ax1_gyro.legend(loc='upper right', fontsize=8)
        
        self.line_top_mag, = self.ax1_mag.plot([], [], 'purple', linewidth=2)
        
        # Magnetometer
        self.line_mag_x, = self.ax2_mag.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)
        self.line_mag_y, = self.ax2_mag.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)
        self.line_mag_z, = self.ax2_mag.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)
        self.ax2_mag.legend(loc='upper right', fontsize=9)
        
        # Rear IMU
        self.line_rear_ax, = self.ax3_accel.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)
        self.line_rear_ay, = self.ax3_accel.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)
        self.line_rear_az, = self.ax3_accel.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)
        self.ax3_accel.legend(loc='upper right', fontsize=8)
        
        self.line_rear_gx, = self.ax3_gyro.plot([], [], 'r-', linewidth=1.5, label='X', alpha=0.8)
        self.line_rear_gy, = self.ax3_gyro.plot([], [], 'g-', linewidth=1.5, label='Y', alpha=0.8)
        self.line_rear_gz, = self.ax3_gyro.plot([], [], 'b-', linewidth=1.5, label='Z', alpha=0.8)
        self.ax3_gyro.legend(loc='upper right', fontsize=8)
        
        self.line_rear_mag, = self.ax3_mag.plot([], [], 'orange', linewidth=2)
        
        # Force Sensor
        self.line_force_x, = self.ax4_force.plot([], [], 'r-', linewidth=1.5, label='Fx', alpha=0.8)
        self.line_force_y, = self.ax4_force.plot([], [], 'g-', linewidth=1.5, label='Fy', alpha=0.8)
        self.line_force_z, = self.ax4_force.plot([], [], 'b-', linewidth=1.5, label='Fz', alpha=0.8)
        self.ax4_force.legend(loc='upper right', fontsize=9)
    
    def _refresh_ports(self):
        """Refresh available COM ports"""
        ports = SerialConfig.get_available_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)
    
    def _connect(self):
        """Connect to data source"""
        source = self.source_var.get()
        
        if source == "serial":
            self._connect_serial()
        else:
            self._connect_csv()
    
    def _connect_serial(self):
        """Connect to serial port"""
        port = self.port_var.get()
        baudrate = int(self.baudrate_var.get())
        
        if not port:
            messagebox.showerror("Error", "Please select a COM port")
            return
        
        # Validate connection
        success, msg = SerialConfig.validate_connection(port, baudrate)
        if not success:
            messagebox.showerror("Connection Error", msg)
            return
        
        # Create serial reader
        self.data_reader = SerialReader(port, baudrate, self._on_data_received)
        if not self.data_reader.connect():
            messagebox.showerror("Error", "Failed to connect to serial port")
            return
        
        self.data_reader.start()
        self.running = True
        self.start_time = datetime.now()
        self.sample_count = 0
        
        self._update_connection_ui(True)
    
    def _connect_csv(self):
        """Connect to CSV file"""
        csv_path = self.csv_entry.get()
        
        if not csv_path:
            messagebox.showerror("Error", "Please enter CSV file path")
            return
        
        # Create CSV reader
        self.data_reader = CSVReader(csv_path, self._on_data_received, has_header=False)
        if not self.data_reader.connect():
            messagebox.showerror("Error", f"Failed to open CSV file: {csv_path}")
            return
        
        self.data_reader.start()
        self.running = True
        self.start_time = datetime.now()
        self.sample_count = 0
        
        self._update_connection_ui(True)
    
    def _disconnect(self):
        """Disconnect from data source"""
        if self.data_reader:
            self.data_reader.stop()
            self.data_reader.close()
            self.data_reader = None
        
        # Reset filter and action detector
        self.filter_manager.reset()
        self.action_detector.reset()
        self.prev_pen_action = "idle"
        
        self.running = False
        self._update_connection_ui(False)
    
    def _on_data_received(self, data):
        """Callback when data is received"""
        # Data comes as a string line from DataReader
        line = data if isinstance(data, str) else data.get('raw_line', '')
        if not line:
            return
        
        parsed = SensorParser.parse_line(line)
        if not parsed:
            return
        
        # Apply sensor fusion filters if enabled
        if self.apply_filter:
            # Filter top IMU (accelerometer and gyroscope)
            parsed['top_accel_x'] = self.filter_manager.filter_accel({'x': parsed['top_accel_x'], 'y': 0, 'z': 0})['x']
            parsed['top_accel_y'] = self.filter_manager.filter_accel({'x': 0, 'y': parsed['top_accel_y'], 'z': 0})['y']
            parsed['top_accel_z'] = self.filter_manager.filter_accel({'x': 0, 'y': 0, 'z': parsed['top_accel_z']})['z']
            
            parsed['top_gyro_x'] = self.filter_manager.filter_gyro({'x': parsed['top_gyro_x'], 'y': 0, 'z': 0})['x']
            parsed['top_gyro_y'] = self.filter_manager.filter_gyro({'x': 0, 'y': parsed['top_gyro_y'], 'z': 0})['y']
            parsed['top_gyro_z'] = self.filter_manager.filter_gyro({'x': 0, 'y': 0, 'z': parsed['top_gyro_z']})['z']
            
            # Filter rear IMU (accelerometer and gyroscope)
            parsed['rear_accel_x'] = self.filter_manager.filter_accel({'x': parsed['rear_accel_x'], 'y': 0, 'z': 0})['x']
            parsed['rear_accel_y'] = self.filter_manager.filter_accel({'x': 0, 'y': parsed['rear_accel_y'], 'z': 0})['y']
            parsed['rear_accel_z'] = self.filter_manager.filter_accel({'x': 0, 'y': 0, 'z': parsed['rear_accel_z']})['z']
            
            parsed['rear_gyro_x'] = self.filter_manager.filter_gyro({'x': parsed['rear_gyro_x'], 'y': 0, 'z': 0})['x']
            parsed['rear_gyro_y'] = self.filter_manager.filter_gyro({'x': 0, 'y': parsed['rear_gyro_y'], 'z': 0})['y']
            parsed['rear_gyro_z'] = self.filter_manager.filter_gyro({'x': 0, 'y': 0, 'z': parsed['rear_gyro_z']})['z']
        
        # Update action detector
        self.action_detector.update(
            top_accel={'x': parsed['top_accel_x'], 'y': parsed['top_accel_y'], 'z': parsed['top_accel_z']},
            rear_accel={'x': parsed['rear_accel_x'], 'y': parsed['rear_accel_y'], 'z': parsed['rear_accel_z']},
            top_gyro={'x': parsed['top_gyro_x'], 'y': parsed['top_gyro_y'], 'z': parsed['top_gyro_z']},
            rear_gyro={'x': parsed['rear_gyro_x'], 'y': parsed['rear_gyro_y'], 'z': parsed['rear_gyro_z']},
            mag={'x': parsed['mag_x'], 'y': parsed['mag_y'], 'z': parsed['mag_z']},
            force={'x': parsed['force_x'], 'y': parsed['force_y'], 'z': parsed['force_z']}
        )
        
        # Detect pen state transitions for character recognition
        current_action = self.action_detector.current_action
        
        # Debug: print action transitions
        if current_action != self.prev_pen_action:
            print(f"🔄 Action changed: {self.prev_pen_action} → {current_action}")
        
        # Start writing when pen touches down or writing action detected
        if current_action in ['pen_down', 'writing'] and self.prev_pen_action not in ['pen_down', 'writing']:
            # Clear previous recognition when starting new character
            self.char_recognition.recognized_char = None
            self.char_recognition.recognized_confidence = 0.0
            self.char_recognition.start_writing()
            print("✏️  Started collecting character data (pen down/writing detected)")
        
        # Stop writing when pen lifts up OR goes to any non-writing state
        elif self.prev_pen_action in ['pen_down', 'writing'] and current_action not in ['pen_down', 'writing']:
            print(f"⏸️  Pen stopped writing (transitioned to: {current_action})")
            self.char_recognition.stop_writing()
            print(f"⏳ Waiting for {self.char_recognition.pause_threshold:.1f}s pause to recognize character...")
        
        # Check if pause is complete - trigger recognition
        if self.char_recognition.check_pause_complete():
            print(f"✅ Pause detected! Recognizing character...")
            recognized_char, confidence = self.char_recognition.end_writing()
            if recognized_char:
                print(f"🎯 Recognized character: {recognized_char} (confidence: {confidence:.1%})")
            else:
                if self.char_recognition.model:
                    print(f"⚠️  Low confidence recognition (below 80% threshold, got {confidence:.1%})")
                else:
                    print("ℹ️  Collected character data, but no model loaded to recognize")
        
        self.prev_pen_action = current_action
        
        # Update character recognition with sensor data
        if self.char_recognition.is_writing:
            self.char_recognition.add_sensor_data(parsed)
        
        # Add to buffers
        self.buffers.add_sample(self.sample_count, parsed)
        self.sample_count += 1
    
    def _update_connection_ui(self, connected):
        """Update UI based on connection state"""
        if connected:
            self.status_label.config(text="✓ Connected", foreground="green")
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            # Disable all config controls while connected
            self.source_combo.config(state="disabled")
            self.port_combo.config(state="disabled")
            self.baudrate_combo.config(state="disabled")
            self.csv_entry.config(state="disabled")
            # Keep filter combo enabled so user can change filters during operation
            self.filter_combo.config(state="readonly")
        else:
            self.status_label.config(text="⊘ Disconnected", foreground="red")
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            # Re-enable all config controls
            self.source_combo.config(state="readonly")
            self.port_combo.config(state="readonly")
            self.baudrate_combo.config(state="readonly")
            self.csv_entry.config(state="normal")
            self.filter_combo.config(state="readonly")
            self.buffers.clear()
            self.sample_count = 0
    
    def update_plot(self):
        """Update plots with latest data"""
        if self.running and len(self.buffers.timestamps) > 0:
            # Get thread-safe snapshot of all data as lists
            data = self.buffers.get_all()
            n_samples = len(data['timestamps'])
            
            if n_samples == 0:
                self.root.after(100, self.update_plot)
                return
            
            # Use numpy arrays
            x_indices = np.arange(n_samples)
            x_min = max(0, n_samples - 256)
            
            # Update Top IMU
            self.line_top_ax.set_data(x_indices, data['top_accel_x'])
            self.line_top_ay.set_data(x_indices, data['top_accel_y'])
            self.line_top_az.set_data(x_indices, data['top_accel_z'])
            self.ax1_accel.set_xlim(x_min, n_samples)
            
            self.line_top_gx.set_data(x_indices, data['top_gyro_x'])
            self.line_top_gy.set_data(x_indices, data['top_gyro_y'])
            self.line_top_gz.set_data(x_indices, data['top_gyro_z'])
            self.ax1_gyro.set_xlim(x_min, n_samples)
            
            # Magnitude - use numpy for vectorized computation on lists
            top_accel_arr = np.column_stack([data['top_accel_x'], 
                                             data['top_accel_y'], 
                                             data['top_accel_z']])
            top_mag = np.linalg.norm(top_accel_arr, axis=1)
            self.line_top_mag.set_data(x_indices, top_mag)
            self.ax1_mag.set_xlim(x_min, n_samples)
            
            # Magnetometer
            self.line_mag_x.set_data(x_indices, data['mag_x'])
            self.line_mag_y.set_data(x_indices, data['mag_y'])
            self.line_mag_z.set_data(x_indices, data['mag_z'])
            self.ax2_mag.set_xlim(x_min, n_samples)
            
            # Rear IMU
            self.line_rear_ax.set_data(x_indices, data['rear_accel_x'])
            self.line_rear_ay.set_data(x_indices, data['rear_accel_y'])
            self.line_rear_az.set_data(x_indices, data['rear_accel_z'])
            self.ax3_accel.set_xlim(x_min, n_samples)
            
            self.line_rear_gx.set_data(x_indices, data['rear_gyro_x'])
            self.line_rear_gy.set_data(x_indices, data['rear_gyro_y'])
            self.line_rear_gz.set_data(x_indices, data['rear_gyro_z'])
            self.ax3_gyro.set_xlim(x_min, n_samples)
            
            rear_accel_arr = np.column_stack([data['rear_accel_x'],
                                              data['rear_accel_y'],
                                              data['rear_accel_z']])
            rear_mag = np.linalg.norm(rear_accel_arr, axis=1)
            self.line_rear_mag.set_data(x_indices, rear_mag)
            self.ax3_mag.set_xlim(x_min, n_samples)
            
            # Force Sensor
            self.line_force_x.set_data(x_indices, data['force_x'])
            self.line_force_y.set_data(x_indices, data['force_y'])
            self.line_force_z.set_data(x_indices, data['force_z'])
            self.ax4_force.set_xlim(x_min, n_samples)
            
            # Update AI Character Recognition display
            if self.char_recognition.model:
                # Model is loaded
                if self.char_recognition.is_writing:
                    buffer_len = len(self.char_recognition.writing_buffer['top_accel_x'])
                    ai_text = f"✏️ Writing...\n\n📊 Collecting Data\n({buffer_len} samples)"
                elif self.char_recognition.recognized_char:
                    # Show last recognized character (persistent)
                    ai_text = f"✓ Recognized:\n\n🔤 {self.char_recognition.recognized_char}\n\n📈 Confidence:\n{self.char_recognition.recognized_confidence:.1%}"
                else:
                    ai_text = "✓ Ready\n\n📦 Model: Loaded\n\n✍️  Write a character\nto recognize"
            else:
                # No model loaded
                if self.char_recognition.is_writing:
                    buffer_len = len(self.char_recognition.writing_buffer['top_accel_x'])
                    ai_text = f"Collecting...\n\n({buffer_len} samples)\n\nNo Model\nto Recognize"
                else:
                    ai_text = "No Model\n\nTrain first using:\ntrain_character_\nmodel.py"
            self.ai_recognition_canvas.itemconfig(self.ai_recognition_text, text=ai_text)
            
            # Update pen action insights box
            action_text = self.action_detector.get_display_text()
            self.action_canvas.itemconfig(self.action_text, text=action_text)
            
            # Update status
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.sample_count / elapsed if elapsed > 0 else 0
            
            self.sample_label.config(text=f"Samples: {self.sample_count}")
            self.rate_label.config(text=f"Rate: {rate:.1f} Hz")
            
            # Stats panel - use snapshot data
            latest_vals = {
                'top_ax': data['top_accel_x'][-1] if n_samples > 0 else 0,
                'top_ay': data['top_accel_y'][-1] if n_samples > 0 else 0,
                'top_az': data['top_accel_z'][-1] if n_samples > 0 else 0,
                'top_mag': top_mag[-1] if len(top_mag) > 0 else 0,
                'rear_ax': data['rear_accel_x'][-1] if n_samples > 0 else 0,
                'rear_ay': data['rear_accel_y'][-1] if n_samples > 0 else 0,
                'rear_az': data['rear_accel_z'][-1] if n_samples > 0 else 0,
                'rear_mag': rear_mag[-1] if len(rear_mag) > 0 else 0,
                'force_x': data['force_x'][-1] if n_samples > 0 else 0,
                'force_y': data['force_y'][-1] if n_samples > 0 else 0,
                'force_z': data['force_z'][-1] if n_samples > 0 else 0,
            }
            
            stats = f"""SYSTEM STATUS
━━━━━━━━━━━━━━━
Samples: {self.sample_count}
Rate: {rate:.1f} Hz
Elapsed: {elapsed:.1f}s

LATEST
━━━━━━━━━━━━━━━
Top Accel:
  X: {latest_vals['top_ax']:.0f}
  Y: {latest_vals['top_ay']:.0f}
  Z: {latest_vals['top_az']:.0f}
  Mag: {latest_vals['top_mag']:.0f}

Rear Accel:
  X: {latest_vals['rear_ax']:.0f}
  Y: {latest_vals['rear_ay']:.0f}
  Z: {latest_vals['rear_az']:.0f}
  Mag: {latest_vals['rear_mag']:.0f}

Force:
  X: {latest_vals['force_x']:.0f}
  Y: {latest_vals['force_y']:.0f}
  Z: {latest_vals['force_z']:.0f}

Status: ✓ RUNNING
"""
            self.stats_text.set_text(stats)
            
            try:
                self.canvas.draw_idle()
                self.canvas.flush_events()  # Process GUI events
            except:
                pass
        
        # Schedule next update
        self.root.after(100, self.update_plot)
    
    def on_closing(self):
        """Handle window closing"""
        self._disconnect()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SensorDashboardGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
