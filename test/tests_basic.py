def test_imports():

    """Test that core modules can be imported"""

    try:

        import numpy

        import cv2

        import PIL

        print("All core imports successful")

        assert True

    except ImportError as e:

        print(f"Import failed: {e}")

        assert False



def test_basic_functionality():

    """Basic functionality test"""

    import numpy as np

    arr = np.array([1, 2, 3])

    assert len(arr) == 3
