import numpy as np

from traenslenzor.image_provider.image_provider import ImageProvider
from traenslenzor.image_renderer.image_renderer import ImageRenderer


def test_inpainting():
    print("starting test_inpainting")

    # Initialize with data directory
    img_provider = ImageProvider("./data")
    img_renderer = ImageRenderer(img_provider=img_provider)

    # Test basic inpainting by loading images directly
    door_defect_img = img_provider.get_image("sbahn-door-defect.jpg")
    door_defect_mask_img = img_provider.get_image("sbahn-door-defect-mask.png").convert("L")

    # Convert mask to numpy array format expected by inpainter
    mask_array = np.array(door_defect_mask_img)
    mask_array = (mask_array > 127).astype(np.uint8)  # Binary mask
    mask_array = mask_array.reshape((1, mask_array.shape[0], mask_array.shape[1]))

    print("loaded img and mask")
    inpainted_result = img_renderer.inpainter.inpaint(door_defect_img, mask_array)

    # Convert result to PIL and save
    inpainted_img = img_provider.img_to_pil(inpainted_result)
    img_provider.save_image(inpainted_img, "../tests/door_defect_inpainted.jpg")
    # assert inpainted_img is not None
