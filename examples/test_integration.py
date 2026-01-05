#!/usr/bin/env python3
"""
End-to-End Integration Test
Tests that all modules work correctly with new directory structure
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test that all key modules can be imported"""
    print("=" * 70)
    print("TESTING MODULE IMPORTS")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    # Test core module imports
    try:
        from core.config import Config

        print("✓ core.config.Config")
        tests_passed += 1
    except Exception as e:
        print(f"✗ core.config.Config: {e}")
        tests_failed += 1

    try:
        from core.serial_config import SerialConfig

        print("✓ core.serial_config.SerialConfig")
        tests_passed += 1
    except Exception as e:
        print(f"✗ core.serial_config.SerialConfig: {e}")
        tests_failed += 1

    try:
        from core.data_reader import SerialReader, CSVReader

        print("✓ core.data_reader (SerialReader, CSVReader)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ core.data_reader: {e}")
        tests_failed += 1

    try:
        from core.sensor_parser import SensorParser

        print("✓ core.sensor_parser.SensorParser")
        tests_passed += 1
    except Exception as e:
        print(f"✗ core.sensor_parser.SensorParser: {e}")
        tests_failed += 1

    try:
        from core.data_buffers import SensorBuffers

        print("✓ core.data_buffers.SensorBuffers")
        tests_passed += 1
    except Exception as e:
        print(f"✗ core.data_buffers.SensorBuffers: {e}")
        tests_failed += 1

    # Test GUI module imports
    try:
        from gui.gui_app import SensorDashboardGUI

        print("✓ gui.gui_app.SensorDashboardGUI")
        tests_passed += 1
    except Exception as e:
        print(f"✗ gui.gui_app.SensorDashboardGUI: {e}")
        tests_failed += 1

    try:
        from gui.plotter import DashboardPlotter

        print("✓ gui.plotter.DashboardPlotter")
        tests_passed += 1
    except Exception as e:
        print(f"✗ gui.plotter.DashboardPlotter: {e}")
        tests_failed += 1

    # Test action module imports
    try:
        from actions.action_detector import ActionDetector

        print("✓ actions.action_detector.ActionDetector")
        tests_passed += 1
    except Exception as e:
        print(f"✗ actions.action_detector.ActionDetector: {e}")
        tests_failed += 1

    # Test AI module imports
    try:
        from ai.character_recognition.model import CharacterRecognitionModel

        print("✓ ai.character_recognition.model.CharacterRecognitionModel")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ai.character_recognition.model: {e}")
        tests_failed += 1

    try:
        from ai.character_recognition.preprocessor import SensorDataPreprocessor

        print("✓ ai.character_recognition.preprocessor.SensorDataPreprocessor")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ai.character_recognition.preprocessor: {e}")
        tests_failed += 1

    try:
        from ai.character_recognition.trainer import ModelTrainer

        print("✓ ai.character_recognition.trainer.ModelTrainer")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ai.character_recognition.trainer: {e}")
        tests_failed += 1

    # Test utils module imports
    try:
        from utils.sensor_buffers import SensorBuffers as SB

        print("✓ utils.sensor_buffers.SensorBuffers")
        tests_passed += 1
    except Exception as e:
        print(f"✗ utils.sensor_buffers: {e}")
        tests_failed += 1

    try:
        from utils.sensor_fusion_filters import SensorFusionManager

        print("✓ utils.sensor_fusion_filters.SensorFusionManager")
        tests_passed += 1
    except Exception as e:
        print(f"✗ utils.sensor_fusion_filters: {e}")
        tests_failed += 1

    print()
    return tests_passed, tests_failed


def test_classes():
    """Test that key classes can be instantiated"""
    print("=" * 70)
    print("TESTING CLASS INSTANTIATION")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    try:
        from core.data_buffers import SensorBuffers

        buffers = SensorBuffers(buffer_size=256)
        print(f"✓ SensorBuffers instantiated (buffer_size=256)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ SensorBuffers instantiation failed: {e}")
        tests_failed += 1

    try:
        from actions.action_detector import ActionDetector

        detector = ActionDetector(buffer_size=30)
        print(f"✓ ActionDetector instantiated (buffer_size=30)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ActionDetector instantiation failed: {e}")
        tests_failed += 1

    try:
        from ai.character_recognition.model import CharacterRecognitionModel

        model = CharacterRecognitionModel(timesteps=512, n_features=13, num_classes=26)
        print(f"✓ CharacterRecognitionModel instantiated")
        tests_passed += 1
    except Exception as e:
        print(f"✗ CharacterRecognitionModel instantiation failed: {e}")
        tests_failed += 1

    try:
        from ai.character_recognition.preprocessor import SensorDataPreprocessor

        preprocessor = SensorDataPreprocessor(max_timesteps=512, sampling_rate=104)
        print(f"✓ SensorDataPreprocessor instantiated")
        tests_passed += 1
    except Exception as e:
        print(f"✗ SensorDataPreprocessor instantiation failed: {e}")
        tests_failed += 1

    try:
        from ai.character_recognition.trainer import ModelTrainer
        from ai.character_recognition.model import CharacterRecognitionModel
        from ai.character_recognition.preprocessor import SensorDataPreprocessor

        model = CharacterRecognitionModel()
        preprocessor = SensorDataPreprocessor()
        trainer = ModelTrainer(model, preprocessor)
        print(f"✓ ModelTrainer instantiated")
        tests_passed += 1
    except Exception as e:
        print(f"✗ ModelTrainer instantiation failed: {e}")
        tests_failed += 1

    print()
    return tests_passed, tests_failed


def test_data_flow():
    """Test basic data flow through the system"""
    print("=" * 70)
    print("TESTING DATA FLOW")
    print("=" * 70)

    tests_passed = 0
    tests_failed = 0

    try:
        from utils.sensor_buffers import SensorBuffers

        buffers = SensorBuffers(buffer_size=256)

        # Create sample sensor data
        sample_data = {
            "top_accel_x": 100.0,
            "top_accel_y": 200.0,
            "top_accel_z": 300.0,
            "top_gyro_x": 10.0,
            "top_gyro_y": 20.0,
            "top_gyro_z": 30.0,
            "rear_accel_x": 100.0,
            "rear_accel_y": 200.0,
            "rear_accel_z": 300.0,
            "rear_gyro_x": 10.0,
            "rear_gyro_y": 20.0,
            "rear_gyro_z": 30.0,
            "mag_x": 50.0,
            "mag_y": 60.0,
            "mag_z": 70.0,
            "force_x": 500.0,
        }

        # Add sample to buffers
        buffers.add_sample(0, sample_data)

        # Get all data
        snapshot = buffers.get_all()

        if snapshot and len(snapshot["top_accel_x"]) > 0:
            print(f"✓ Data buffering works (1 sample added)")
            tests_passed += 1
        else:
            print(f"✗ Data buffering failed (snapshot empty)")
            tests_failed += 1

    except Exception as e:
        print(f"✗ Data buffering test failed: {e}")
        tests_failed += 1

    try:
        from actions.action_detector import ActionDetector
        from utils.sensor_buffers import SensorBuffers

        buffers = SensorBuffers(buffer_size=256)
        detector = ActionDetector(buffer_size=30)

        # Add multiple samples
        for i in range(50):
            sample_data = {
                "top_accel_x": 100.0 + i * 10,
                "top_accel_y": 200.0 + i * 10,
                "top_accel_z": 300.0 + i * 10,
                "top_gyro_x": 10.0,
                "top_gyro_y": 20.0,
                "top_gyro_z": 30.0,
                "rear_accel_x": 100.0,
                "rear_accel_y": 200.0,
                "rear_accel_z": 300.0,
                "rear_gyro_x": 10.0,
                "rear_gyro_y": 20.0,
                "rear_gyro_z": 30.0,
                "mag_x": 50.0,
                "mag_y": 60.0,
                "mag_z": 70.0,
                "force_x": 500.0,
            }
            buffers.add_sample(i, sample_data)

            # Update action detector with proper format
            top_accel = {
                "x": sample_data["top_accel_x"],
                "y": sample_data["top_accel_y"],
                "z": sample_data["top_accel_z"],
            }
            rear_accel = {
                "x": sample_data["rear_accel_x"],
                "y": sample_data["rear_accel_y"],
                "z": sample_data["rear_accel_z"],
            }
            top_gyro = {"x": sample_data["top_gyro_x"], "y": sample_data["top_gyro_y"], "z": sample_data["top_gyro_z"]}
            rear_gyro = {
                "x": sample_data["rear_gyro_x"],
                "y": sample_data["rear_gyro_y"],
                "z": sample_data["rear_gyro_z"],
            }
            mag = {"x": sample_data["mag_x"], "y": sample_data["mag_y"], "z": sample_data["mag_z"]}
            force = {"x": sample_data["force_x"]}

            detector.update(top_accel, rear_accel, top_gyro, rear_gyro, mag, force)

        state = detector.get_action()
        if state:
            print(f"✓ Action detection works (detected state: {state})")
            tests_passed += 1
        else:
            print(f"✗ Action detection failed (no state detected)")
            tests_failed += 1

    except Exception as e:
        print(f"✗ Action detection test failed: {e}")
        tests_failed += 1

    print()
    return tests_passed, tests_failed


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "END-TO-END INTEGRATION TEST" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    total_passed = 0
    total_failed = 0

    # Test imports
    p, f = test_imports()
    total_passed += p
    total_failed += f

    # Test class instantiation
    p, f = test_classes()
    total_passed += p
    total_failed += f

    # Test data flow
    p, f = test_data_flow()
    total_passed += p
    total_failed += f

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Passed: {total_passed}")
    print(f"✗ Failed: {total_failed}")
    print(f"Total:   {total_passed + total_failed}")
    print()

    if total_failed == 0:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Run GUI: python run.py")
        print("  2. Train model: python train_character_model.py")
        print("  3. Verify setup: python verify_setup.py")
        return 0
    else:
        print(f"⚠️  {total_failed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
