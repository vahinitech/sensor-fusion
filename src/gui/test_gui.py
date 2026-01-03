#!/usr/bin/env python3
"""
Test script for GUI app - Validates core functionality and imports
"""

import sys
import os
import csv
import time


def test_imports():
    """Test that all required modules can be imported"""
    print("\n1. Testing Module Imports...")
    print("-" * 50)
    
    try:
        from src.serial_config import SerialConfig
        print("  ✓ SerialConfig imported")
    except Exception as e:
        print(f"  ✗ Failed to import SerialConfig: {e}")
        return False
    
    try:
        from src.sensor_buffers import SensorBuffers
        print("  ✓ SensorBuffers imported")
    except Exception as e:
        print(f"  ✗ Failed to import SensorBuffers: {e}")
        return False
    
    try:
        from src.sensor_parser import SensorParser
        print("  ✓ SensorParser imported")
    except Exception as e:
        print(f"  ✗ Failed to import SensorParser: {e}")
        return False
    
    try:
        from src.data_reader import DataReader
        print("  ✓ DataReader imported")
    except Exception as e:
        print(f"  ✗ Failed to import DataReader: {e}")
        return False
    
    return True


def test_sensor_buffers():
    """Test SensorBuffers functionality"""
    print("\n2. Testing SensorBuffers...")
    print("-" * 50)
    
    try:
        from src.sensor_buffers import SensorBuffers
        
        buffers = SensorBuffers(buffer_size=256)
        
        # Create test data using the correct keys
        test_data = {
            'timestamp': 0,
            'top_accel_x': 100,
            'top_accel_y': 200,
            'top_accel_z': 300,
            'top_gyro_x': 10,
            'top_gyro_y': 20,
            'top_gyro_z': 30,
            'mag_x': 1,
            'mag_y': 2,
            'mag_z': 3,
            'rear_accel_x': 400,
            'rear_accel_y': 500,
            'rear_accel_z': 600,
            'rear_gyro_x': 40,
            'rear_gyro_y': 50,
            'rear_gyro_z': 60,
            'force_x': 100,
            'force_y': 200,
            'force_z': 300
        }
        
        # Add sample
        buffers.add_sample(0, test_data)
        
        # Verify data was added
        assert len(buffers.top_accel_x) == 1, "Buffer should contain one sample"
        assert buffers.top_accel_x[0] == 100, "Accel X value should be 100"
        
        print(f"  ✓ Add sample successful")
        print(f"  ✓ Top accel X value: {buffers.top_accel_x[0]}")
        
        # Add more samples
        for i in range(1, 10):
            test_data['timestamp'] = i
            test_data['top_accel_x'] = 100 + i
            buffers.add_sample(i, test_data)
        
        assert len(buffers.top_accel_x) == 10, "Buffer should contain 10 samples"
        print(f"  ✓ Multiple samples added: {len(buffers.top_accel_x)} entries")
        
        # Test clear
        buffers.clear()
        assert len(buffers.top_accel_x) == 0, "Buffer should be empty after clear"
        print(f"  ✓ Clear buffer works")
        
        return True
    except Exception as e:
        print(f"  ✗ SensorBuffers test failed: {e}")
        return False


def test_sensor_parser():
    """Test SensorParser functionality"""
    print("\n3. Testing SensorParser...")
    print("-" * 50)
    
    try:
        from src.sensor_parser import SensorParser
        
        # Test parsing a sample line
        # Format: TIMESTAMP,TOP_ACCEL_X,TOP_ACCEL_Y,TOP_ACCEL_Z,TOP_GYRO_X,TOP_GYRO_Y,TOP_GYRO_Z,
        #         MAG_X,MAG_Y,MAG_Z,REAR_ACCEL_X,REAR_ACCEL_Y,REAR_ACCEL_Z,REAR_GYRO_X,REAR_GYRO_Y,REAR_GYRO_Z,
        #         FORCE_X,FORCE_Y,FORCE_Z
        
        test_line = "1000,100,200,300,10,20,30,1,2,3,400,500,600,40,50,60,100,200,300"
        result = SensorParser.parse_line(test_line)
        
        if result:
            assert result['top_accel_x'] == 100, "Top accel X should be 100"
            assert result['mag_x'] == 1, "Mag X should be 1"
            assert result['force_z'] == 300, "Force Z should be 300"
            print(f"  ✓ Parse line successful")
            print(f"  ✓ Top accel: ({result['top_accel_x']}, {result['top_accel_y']}, {result['top_accel_z']})")
            return True
        else:
            print(f"  ✗ Failed to parse line")
            return False
    except Exception as e:
        print(f"  ✗ SensorParser test failed: {e}")
        return False


def test_csv_reading():
    """Test CSV file reading"""
    print("\n4. Testing CSV File Reading...")
    print("-" * 50)
    
    try:
        csv_file = "../data/sample_sensor_data.csv"
        
        if not os.path.exists(csv_file):
            print(f"  ✗ CSV file not found: {csv_file}")
            return False
        
        # Read CSV and count lines
        line_count = 0
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    line_count += 1
        
        print(f"  ✓ CSV file exists: {csv_file}")
        print(f"  ✓ Total lines in CSV: {line_count}")
        
        # Test reading with DataReader
        from src.data_reader import CSVReader
        
        samples_read = [0]
        
        def on_data(line):
            samples_read[0] += 1
        
        csv_reader = CSVReader(csv_file, data_callback=on_data, has_header=False)
        if csv_reader.connect():
            csv_reader.start()
            time.sleep(0.5)
            csv_reader.stop()
            csv_reader.close()
            
            print(f"  ✓ CSVReader successfully read {samples_read[0]} samples from CSV")
            return True
        else:
            print(f"  ✗ CSVReader failed to open CSV")
            return False
    except Exception as e:
        print(f"  ✗ CSV reading test failed: {e}")
        return False


def test_serial_config():
    """Test SerialConfig"""
    print("\n5. Testing SerialConfig...")
    print("-" * 50)
    
    try:
        from src.serial_config import SerialConfig
        
        baudrates = SerialConfig.get_standard_baudrates()
        print(f"  ✓ Standard baudrates available: {len(baudrates)} rates")
        print(f"    Rates: {baudrates[:3]}... (showing first 3)")
        
        ports = SerialConfig.get_available_ports()
        print(f"  ✓ Available COM ports: {len(ports)}")
        if ports:
            print(f"    Ports: {ports[:3]}... (showing first 3)")
        else:
            print(f"    No COM ports available (expected in test environment)")
        
        return True
    except Exception as e:
        print(f"  ✗ SerialConfig test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("SENSOR FUSION DASHBOARD - GUI APP VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("SerialConfig", test_serial_config),
        ("SensorBuffers", test_sensor_buffers),
        ("SensorParser", test_sensor_parser),
        ("CSV Reading", test_csv_reading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} - {test_name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL VALIDATIONS PASSED - GUI APP IS READY")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
