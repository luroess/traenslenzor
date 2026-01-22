#import "@preview/supercharged-hm:0.1.2": *

== Document Image Renderer <comp_document_image_renderer>

The Document Image Renderer is responsible for the final stage of the translation pipeline: replacing extracted text in document images with their translations while preserving the visual appearance of the original document.
It consumes the output of three preceding components:
- Text bounding boxes and coordinates from the Text Extractor (@comp_text_extractor)
- Font information (family, size, color) from the Font Detector (@comp_font_detector)
- Translated text from the Document Translator (@comp_document_translator)

These inputs are combined into `RenderReadyItem` objects, which the renderer uses to produce the final translated document image.
The rendering process requires two key capabilities: removing the original text from the image through AI-powered inpainting, and rendering the translated text in the correct position, font, and orientation.

=== Inpainting with LaMa

The core challenge of text replacement lies in removing the original text without leaving visible artifacts.
Simply painting over text with a solid color would create obvious rectangular patches that break the document's visual coherence.
Instead, the renderer uses LaMa (Large Mask Inpainting), a deep learning model specifically designed for filling large masked regions in images.

LaMa was introduced by Suvorov et al. in "Resolution-robust Large Mask Inpainting with Fourier Convolutions" @suvorovResolutionrobustLargeMask2021.
The model's key innovation is its use of Fast Fourier Convolutions (FFCs), which provide an image-wide receptive field in a single layer.
Traditional convolutional networks have a limited receptive field determined by kernel size, requiring many stacked layers to capture global context.
FFCs solve this by operating in the frequency domain via the Fast Fourier Transform, allowing the network to "see" the entire image at once.
This design is particularly effective for document images, where background textures and patterns must be seamlessly continued across the inpainted region.

A notable property of LaMa is its resolution robustness: despite being trained on 256×256 images, it produces high-quality results on images up to approximately 2000×2000 pixels.
This allows the renderer to process high-resolution document scans without downscaling.

=== Model Deployment

During development, two deployment strategies for the LaMa model were evaluated: PyTorch JIT (TorchScript) and ONNX Runtime.

#figure(caption: [LaMa inpainter using PyTorch JIT with lazy model loading: `traenslenzor/image_renderer/inpainting.py`])[
#code()[```py
class Inpainter:
    def __init__(
        self,
        path_to_model: str = "./traenslenzor/image_renderer/lama/big-lama.pt",
        device: str = "mps",
    ) -> None:
        self.device = torch.device(device)
        model_path = Path(path_to_model)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            _ = urlretrieve(MODEL_URL, model_path, reporthook=_download_progress)

        self.LaMa = torch.jit.load(model_path, map_location=self.device)
        self.LaMa.to(self.device)
        self.LaMa.eval()
```]]

The production implementation uses PyTorch's TorchScript format, loaded via `torch.jit.load`.
This approach provides native hardware acceleration on macOS via the Metal Performance Shaders (MPS) backend, which proved essential as development primarily occurred on Apple Silicon machines.

An ONNX-based backend was also implemented for portability.
However, the CoreML Execution Provider required for hardware acceleration on Apple devices was not reliably available, causing the ONNX backend to fall back to CPU execution.
The resulting performance difference was substantial enough that the PyTorch JIT backend was chosen for production use.
The ONNX implementation remains in the codebase for testing and comparison purposes.

Both backends implement an identical interface, accepting a PIL Image and a binary mask array of shape `(1, H, W)`, returning a normalized float32 numpy array of shape `(H, W, 3)` with values in the `[0, 1]` range.
This consistent interface allows swapping backends without changes to the calling code.

=== Text Rendering with Rotation Support

After inpainting removes the original text, the translated text must be rendered in its place.
This is complicated by the fact that text in documents is often rotated—either because the document was photographed at an angle, or because text elements themselves have non-horizontal orientations.

The renderer calculates the rotation angle from the four-point polygon bounding box provided by the text extractor.
By analyzing the vector from the upper-left to upper-right corner, the rotation can be computed via arctangent:

#figure(caption: [Rotation angle calculation from bounding box: `traenslenzor/image_renderer/text_operations.py`])[
#code()[```py
def get_angle_from_bbox(bbox: List[BBoxPoint]) -> tuple[float, NDArray[np.float64]]:
    ul = bbox[0]  # upper-left
    ur = bbox[1]  # upper-right

    delta_x = ur.x - ul.x
    delta_y = ur.y - ul.y

    radians = np.arctan2(delta_y, delta_x)
    transformation_matrix = np.array(
        [[np.cos(radians), -np.sin(radians)],
         [np.sin(radians), np.cos(radians)]]
    )
    return (np.degrees(radians) % 360, transformation_matrix)
```]]

Text is drawn on a temporary RGBA canvas with sufficient padding to prevent clipping during rotation.
The rotated text is then composited onto the inpainted image at the correct position.
This approach handles arbitrary rotation angles while preserving text quality through bicubic resampling.

=== Mask Creation and Dilation

The text regions to be inpainted are defined by binary masks.
For each text item, the bounding box polygon is filled in a mask image.
A `MaxFilter` is then applied to slightly dilate the mask, filling gaps between text lines and ensuring complete coverage of text regions.
This dilation step proved important for visual quality, as thin uncovered strips between lines would otherwise create visible artifacts in the inpainted result.

#figure(caption: [Mask creation with dilation: `traenslenzor/image_renderer/text_operations.py`])[
#code()[```py
def create_mask(texts: list[RenderReadyItem], mask_shape: tuple[int, int]) -> NDArray[np.uint8]:
    mask = Image.new("L", (mask_shape[1], mask_shape[0]), 0)
    draw_mask = ImageDraw.Draw(mask)
    for text in texts:
        draw_mask.polygon([(point.x, point.y) for point in text.bbox], fill=255)

    # Slightly dilate the mask to fill gaps between text lines
    mask = mask.filter(ImageFilter.MaxFilter(3))

    mask_array = np.array(mask).reshape((1, mask_shape[0], mask_shape[1]))
    mask_array = (mask_array > 127).astype(np.uint8)
    return mask_array
```]]

=== Transformation Pipeline

Document images undergo perspective correction during preprocessing—specifically, the document deskewing performed by the Text Extractor (@comp_text_extractor).
When a photograph is taken at an angle, the Text Extractor detects the document boundary and applies a perspective transformation to produce a rectangular image suitable for #gls("ocr").

The renderer must account for this transformation to place translated text correctly on the original document.
The Text Extractor stores the transformation matrix in the session's `DocumentInfo`, which the renderer retrieves during the final compositing stage.
After rendering text onto the deskewed image, the renderer applies the inverse transformation to map the result back to the original document's coordinate space.
The transformed result is then composited onto the original image using alpha blending, preserving any parts of the original that were not affected by text replacement.

=== MCP Server Integration

The renderer is exposed as an #gls("mcp") tool via FastMCP, making it available to the Supervisor (@comp_supervisor) for dynamic invocation.
The `replace_text` tool validates that all prerequisites are met before processing: the session must contain extracted text, all text items must have font information and translations (i.e., all items must be `RenderReadyItem`), and the original document must be available.
This validation ensures that the Font Detector (@comp_font_detector) and Document Translator (@comp_document_translator) have completed their work before rendering begins.

#figure(caption: [RenderResult dataclass returned by the MCP tool: `traenslenzor/image_renderer/mcp_server.py`])[
#code()[```py
@dataclass
class RenderResult:
    """Result of a text replacement operation."""
    success: bool
    rendered_document_id: str
```]]

The tool loads images from the file server, processes them through the rendering pipeline, stores the result back to the file server, and updates the session state with the rendered document ID.
Error handling uses `ToolError` exceptions with descriptive messages to guide the #gls("llm") in case of missing prerequisites.

=== Testing

The component includes a test suite covering both inpainting backends and the text operations module.
Unit tests verify that inpainters produce correct output shapes and value ranges, with snapshot reference images detecting regressions in inpainting quality.
The text operations module is tested with various rotation angles, including edge cases at 0°, 90°, and arbitrary angles.
Integration tests exercise the complete pipeline from session loading to final image output.
Tests requiring model downloads are skipped in CI environments to avoid slow automated builds.
