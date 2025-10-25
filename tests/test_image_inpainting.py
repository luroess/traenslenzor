from pathlib import Path

from PIL import Image as Pillow

from traenslenzor.image_renderer.image_renderer import ImageRenderer


def test_inpainting():
    print("starting test_inpainting")
    img_renderer = ImageRenderer()

    door_defect_path = Path("./data/sbahn-door-defect.jpg")
    door_defect_mask_path = Path("./data/sbahn-door-defect-mask.png")
    door_defect_result_path = Path("./data/sbahn-door-defect-result.png")

    # interuption_path = Path("./data/sbahn-betriebstoerung.jpg")
    # interuption_mask_path = Path("./data/sbahn-betriebstoerung-mask.png")
    # interuption_result_path = Path("./data/sbahn-betriebstoerung-result.png")

    with (
        Pillow.open(door_defect_path) as img,
        Pillow.open(door_defect_mask_path).convert("L") as mask,
        Pillow.open(door_defect_result_path) as _result,
    ):
        print("loaded img and mask")
        inpainted_img = img_renderer.inpaint_mask(img, mask)
        inpainted_img.save("./tests/door_defect_inpainted.jpg")
        # assert inpainted_img is not None
