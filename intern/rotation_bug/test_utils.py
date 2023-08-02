import skimage
import utils
import matplotlib.pyplot as plt


def test_rotated_image():
    """Test for image shape after rotation
    """
    image = skimage.data.rocket()
    result = utils.rotated_image(image)
    skimage.io.imshow(result)
    plt.show()
    assert image.shape == result.shape


