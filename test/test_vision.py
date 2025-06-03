def test_opencv_basic():

    """Test OpenCV basic functionality"""

    import cv2

    import numpy as np

    

    # Create a simple test image

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    assert img.shape == (100, 100, 3)



def test_pillow_basic():

    """Test PIL/Pillow basic functionality"""

    from PIL import Image

    import numpy as np

    

    # Create a test image

    img_array = np.zeros((100, 100, 3), dtype=np.uint8)

    img = Image.fromarray(img_array)

    assert img.size == (100, 100)
