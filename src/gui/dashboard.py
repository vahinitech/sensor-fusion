"""Main dashboard orchestrator"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

from .config import Config
from .data_reader import SerialReader, CSVReader
from .data_parser import SensorDataParser
from .data_buffers import SensorBuffers
from .plotter import DashboardPlotter


class MultiSensorDashboard:
    """Main dashboard controller for multi-sensor visualization"""
    
    def __init__(self, config_path='config.json', data_source='csv', data_path=None):
        """
        Initialize dashboard
        
        Args:
            config_path: Path to JSON config
            data_source: 'serial' or 'csv'
            data_path: Path to CSV file (if csv mode)
        """
        self.config = Config(config_path)
        self.data_source = data_source
        self.data_path = data_path
        
        # Initialize components
        self.parser = SensorDataParser()
        self.buffers = SensorBuffers(
            buffer_size=self.config.get('dashboard', 'buffer_size', default=256)
        )
        self.plotter = DashboardPlotter(self.config, self.buffers)
        
        # Data reader (set up in connect)
        self.reader = None
        self.start_time = None
    
    def _handle_data_line(self, line):
        """Callback for when new data line is received"""
        data = self.parser.parse_line(line)
        if data:
            self.buffers.add_data(data, self.buffers.sample_count)
    
    def connect(self):
        """Establish connection to data source"""
        if self.data_source == 'serial':
            port = self.config.get('data_source', 'serial', 'port', default='COM3')
            baudrate = self.config.get('data_source', 'serial', 'baudrate', default=115200)
            self.reader = SerialReader(port, baudrate, self._handle_data_line)
        else:
            csv_path = self.data_path or self.config.get('data_source', 'csv', 'path', 
                                                         default='sample_sensor_data.csv')
            has_header = self.config.get('data_source', 'csv', 'has_header', default=False)
            self.reader = CSVReader(csv_path, self._handle_data_line, has_header)
        
        return self.reader.connect()
    
    def run(self):
        """Start dashboard visualization"""
        print("\n" + "="*60)
        print("  MULTI-SENSOR REAL-TIME DASHBOARD v2.0")
        print("  104Hz | Modular Architecture")
        print("="*60 + "\n")
        
        if not self.connect():
            print("✗ Connection failed. Exiting...")
            return
        
        # Start data reading
        self.start_time = datetime.now()
        self.reader.start()
        print(f"✓ Data reader started ({self.data_source.upper()} mode)\n")
        
        # Setup visualization
        print("Setting up dashboard...")
        title = f'{self.config.get("dashboard", "title")} | {self.data_source.upper()}'
        fig = self.plotter.setup_figure(title)
        print("✓ Dashboard ready\n")
        
        # Start animation
        interval = self.config.get('ui', 'update_interval_ms', default=100)
        ani = animation.FuncAnimation(
            fig, 
            lambda frame: self.plotter.update(frame, self.data_source.upper()),
            interval=interval, 
            blit=False, 
            cache_frame_data=False
        )
        
        print("Starting visualization...\n")
        plt.tight_layout()
        plt.show()
        
        # Cleanup
        self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n" + "="*60)
        print("  Shutting down...")
        print("="*60)
        
        if self.reader:
            self.reader.stop()
            self.reader.close()
        
        stats = self.parser.get_stats()
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print(f"\nTotal samples: {stats['valid']}")
        print(f"Parse errors: {stats['errors']}")
        if elapsed > 0:
            print(f"Average rate: {stats['valid'] / elapsed:.1f} Hz")
        print("\n✓ Cleanup complete\n")
