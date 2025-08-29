#!/usr/bin/env python3
"""
Quick test script for LabelMe integration
Tests the label generator GUI with LabelMe features
"""

import os
import sys

def test_labelme_availability():
    """Check if LabelMe is available"""
    try:
        import labelme
        print("✅ LabelMe is installed")
        print(f"   Version: {labelme.__version__}")
        return True
    except ImportError:
        print("❌ LabelMe not installed")
        print("   Install with: pip install labelme")
        return False

def test_gui_launch():
    """Test if the GUI can be imported"""
    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Import the GUI functions
        from label_generator_gui import launch_labelme, convert_labelme_annotations
        print("✅ Label generator GUI imported successfully")
        
        # Test launch_labelme function
        print("\n🧪 Testing launch_labelme function...")
        # Don't actually launch it, just test the function exists
        print("   Function 'launch_labelme' is available")
        print("   Function 'convert_labelme_annotations' is available")
        
        return True
    except Exception as e:
        print(f"❌ Error importing GUI: {e}")
        return False

def main():
    print("=" * 50)
    print("CERAMATIC LABELME INTEGRATION TEST")
    print("=" * 50)
    print()
    
    # Test 1: Check LabelMe
    labelme_ok = test_labelme_availability()
    
    # Test 2: Check GUI
    gui_ok = test_gui_launch()
    
    print("\n" + "=" * 50)
    if labelme_ok and gui_ok:
        print("✅ ALL TESTS PASSED")
        print("\nYou can now:")
        print("1. Run: python label_generator_gui.py")
        print("2. Click 'Open LabelMe' buttons in the interface")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding")
    print("=" * 50)

if __name__ == "__main__":
    main()