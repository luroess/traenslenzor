#import "@preview/supercharged-hm:0.1.2": *
#import "@preview/wrap-it:0.1.1": wrap-content
#import "@preview/algorithmic:1.0.7"
#import algorithmic: algorithm-figure, style-algorithm
#show: style-algorithm

== Doc Scanner (UVDoc Deskew + Super-Resolution) <comp_doc_scanner>

The Doc Scanner component converts sloppily photographed documents into a scan-like, top-down view and stores the required geometric metadata in the session state.
It is exposed as an #gls("mcp") server and writes an `ExtractedDocument` to the file server. It provides the agent with two tools:
- `deskew_document(session_id, crop_document?)`
- `super_resolve_document(session_id, source="deskewed")`

=== Background: UVDoc

UVDoc frames document dewarping as dense resampling.
// TODO: don't say a "neural network", use term to describe the specific architecture.
A neural network predicts a coarse 2D sampling grid alongside a coarse 3D surface grid; the 2D grid is upsampled and applied via `grid_sample` to obtain the unwarped image @VerhoevenUVDoc2023.

=== Background: Text Super-Resolution

Document photos often contain small fonts or fine stroke structure that become hard to read after deskewing, especially when the raw image is low-resolution or heavily compressed.
To improve downstream #gls("ocr") and rendering quality, we optionally apply a lightweight super-resolution model to the deskewed document.

We use OpenVINO's Open Model Zoo network `text-image-super-resolution-0001`, which performs a fixed 3#sym.times upscaling and supports reshaping to different input sizes @OpenVINOTextImageSR0001.
Architecturally, it is a compact CNN that upsamples with a transposed convolution head (`ConvTranspose2d`) (instead of a PixelShuffle block) @OpenVINOTextImageSR0001, aligning with fast SR designs such as FSRCNN @Dong_2016_ECCV.
Like many super-resolution pipelines, we operate in a luminance/chroma color space: we super-resolve the `Y` channel and upscale chroma channels with bicubic interpolation, which preserves color consistency while sharpening glyph edges.

==== Super-resolution stress test (blur + downscale)

To validate the effect of super-resolution under unfavorable capture conditions, we use a cluttered invoice photo and synthetically degrade it by downscaling to 1/4 of the original resolution and applying a mild Gaussian blur (kernel width #sym.sigma=1).
We then run UVDoc deskewing and apply text super-resolution to the deskewed result.
@fig-doc-scanner-superres-grid shows the end-to-end effect, and @fig-doc-scanner-superres-zoom highlights the improvement in local text sharpness.

#figure(
  caption: [Deskew + text super-resolution on a degraded invoice input (downscale 1/4 + Gaussian blur, #sym.sigma=1).],
)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    [
      #image("/imgs/doc-scanner/invoice_clutter_downscaled_blur.jpg", width: 60%)
      #text(size: 8pt)[(a) Degraded input image.]
    ],
    [
      #image("/imgs/doc-scanner/invoice_clutter_downscaled_blur_deskewed_superres.png", width: 60%)
      #text(size: 8pt)[(b) UVDoc deskew output + OpenVINO text SR (3#sym.times).]
    ],
  )
] <fig-doc-scanner-superres-grid>

#figure(
  caption: [Zoomed crop: low-resolution vs. super-resolved text region.],
)[
  #image("/imgs/doc-scanner/invoice_compare_super_res.png", width: 100%)
] <fig-doc-scanner-superres-zoom>

=== Implementation (Traenslenzor)

Input: `SessionState.rawDocumentId` (the raw photo). \
Output: `SessionState.extractedDocument`, containing:

- `id`: file id of the deskewed image
- `documentCoordinates`: four page corners in original image coordinates (UL, UR, LR, LL)
- `mapXYId` (+ shape): downsampled flow field mapping output pixels #sym.arrow.r original pixels
- `transformation_matrix`: homography approximation (fallback if no flow field is available)


==== Deskew pipeline (UVDoc + flow field)

UVDoc predicts a dense sampling grid in the `grid_sample` convention.
After upsampling the grid to full resolution, we resample the original image and obtain an unwarped output.
Crucially, we also convert the normalized grid into a pixel-space backward map `map_xy`, which enables accurate non-linear backtransforms (preferred) while still providing a planar homography approximation (`transformation_matrix`) as a lightweight fallback.

Algorithm @alg-uvdoc-deskew summarizes the end-to-end deskew pipeline and the produced artifacts.
#pagebreak
#algorithm-figure(
  [UVDoc deskew with flow-field export],
  vstroke: .5pt + luma(220),
  {
    import algorithmic: *

    Procedure(
      "DeskewUVDoc",
      ("I", "cfg"),
      {
        Comment[Prepare inference input and normalize source]
        Assign[`(H, W)`][`shape(I)`]
        Assign[`inp`][`Resize(I, DEFAULT_IMG_SIZE)`]

        Comment[Predict coarse sampling grid (normalized coordinates)]
        Assign[`points2d`][`UVDocNet(inp)`]

        Comment[Upsample grid and unwarp document]
        Assign[`grid`][`UpsampleAndClamp(points2d, size=(H, W))`]
        Assign[`I_unwarp`][`grid_sample(I, grid, align_corners=true)`]

        Comment[Convert normalized grid to pixel-space backward map `map_xy`]
        Line[`map_xy_full[y,x] = ((u+1)/2*(W-1), (v+1)/2*(H-1))`]
        Assign[`s`][`ComputeMapStride((H, W), cfg.max_map_pixels)`]

        IfElseChain(
          `cfg.crop_page`,
          {
            Assign[`C_u, r`][`FindPageCorners(I_unwarp, cfg.min_area_ratio)`]
            If(`C_u is None`, { Assign[`C_u`][`FullImageCorners(I_unwarp)`] })

            Comment[Crop output and propagate flow field]
            Assign[`I_out, H_crop`][`WarpFromCorners(I_unwarp, C_u)`]
            Assign[`map_xy`][`CropAndDownsampleMap(map_xy_full, H_crop, stride=s)`]
          },
          Else({
            Assign[`I_out`][`I_unwarp`]
            Assign[`map_xy`][`map_xy_full[::s, ::s]`]
          }),
        )

        Comment[Map corners to original space and compute homography fallback]
        Assign[`C_orig`][`SampleMapXY(map_xy_full, C_u)`]
        Assign[`H`][`getPerspectiveTransform(C_orig, output_rectangle)`]

        Return[`(I_out, C_orig, map_xy, H)`]
      },
    )
  },
) <alg-uvdoc-deskew>


Technically, the backend runs UVDocNet on a fixed-size input, upsamples the predicted 2D grid to the original resolution, and applies `grid_sample(..., align_corners: true)` to produce the deskewed output.
The normalized sampling grid is also converted into a pixel-space flow field `map_xy`, which enables accurate non-linear backtransforms.
To keep stored flow fields manageable, `map_xy` may be stride-downsampled to satisfy a pixel budget (controlled by `max_map_pixels`) or disabled entirely (`generate_map_xy=false`).

In more detail, UVDoc predicts normalized sampling coordinates `(u, v)` in the `grid_sample` convention (`u, v in [-1, 1]`).
With `align_corners: true`, we convert normalized coordinates into original-image pixel coordinates:

$ x_in = ((u + 1) / 2) * (W - 1) $
$ y_in = ((v + 1) / 2) * (H - 1) $

We store this backward mapping as a flow field `map_xy[y, x] = (x_in, y_in)`,
i.e., each deskewed output pixel `(x, y)` points to the originating location in the raw input image.
For a lightweight planar fallback, we also compute a homography `H` from 4 corner correspondences (stored as `transformation_matrix`).

The tool optionally crops the unwarped image to the detected page contour while keeping a consistent `map_xy` for backtransforming edits back to the raw image.

Page cropping uses a fast contour-based heuristic (`find_page_corners`) on the UVDoc output:
- grayscale + Gaussian blur
- Otsu thresholding (evaluated on mask and inverted mask) @opencv_tutorial_thresholding_otsu
- morphological closing (fills gaps)
- largest external contour #sym.arrow.r convex hull #sym.arrow.r polygon approximation (`approxPolyDP`, `#sym.epsilon = 0.02 * perimeter`) @opencv_tutorial_contour_features
- fallback to `minAreaRect` if no clean quadrilateral is found
- reject small candidates via an area-ratio threshold; order corners (UL, UR, LR, LL)

*Why Otsu?* We want a robust, fully automatic foreground/background split without hand-tuning a threshold for each lighting condition.
Otsu's method selects a global threshold from the grayscale histogram by minimizing within-class variance; for document photos (after UVDoc unwarping) the page/background distribution is often close to bimodal, which makes Otsu a good fit in practice @opencv_tutorial_thresholding_otsu.
We apply a small Gaussian blur first (as recommended in the tutorial) to suppress noise, and we try both mask polarities (mask and inverted mask) to handle both dark-on-light and light-on-dark backgrounds.

*Why a convex hull?* Raw contours on a thresholded image can contain small concavities caused by wrinkles, shadows, or broken edges.
`cv.convexHull` constructs a convex envelope of the contour by “correcting convexity defects” @opencv_tutorial_contour_features.
Using the hull before polygon approximation stabilizes the subsequent quadrilateral fit: it removes local inward dents and yields a smoother, globally consistent page boundary.

==== Page detection + planar rectification (adapted utilities)

The page-corner detection and planar warping primitives are implemented in `traenslenzor/doc_scanner/utils.py` and are *adapted* from the earlier OpenCV-based deskew in `traenslenzor/text_extractor/flatten_image.py`.
We keep the same geometric core (ordering corners, estimating output size, and computing a homography), but rework the corner ordering heuristic (sum/diff rule) and generalize the utilities for the UVDoc pipeline (returning the homography and adding fallbacks like `minAreaRect`).

In other words: the original `flatten_image.py` utilities (Felix Schladt) implement a fully classical corner-based deskew on the *raw* photo, while our `utils.py` functions serve as *post-processing* around UVDoc (page cropping + planar fallback).
This shift in context drives several deliberate changes:

#figure(
  caption: [Key differences between the legacy OpenCV deskew utilities (Felix) and our UVDoc-aligned geometry helpers.],
)[
  #set text(size: 8pt)
  #table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 6pt,
    table.header([*Function(s)*], [*Change + rationale*]),

    [`_order_points` #sym.arrow `order_points_clockwise`],
    [
      Sort by y/x (top pair, bottom pair) #sym.arrow sum/diff heuristic (`x+y`, `x-y`) to order corners to (UL, UR, LR, LL) for improved robustness against rotations.
    ],

    [`_warp_to_rectangle` #sym.arrow `warp_from_corners`],
    [
      Same homography-based rectification core, but returns additional metadata: output size and the homography matrix (float32) to persist a consistent planar fallback (`transformation_matrix`) and enable size-aware cropping/backtransforms in the UVDoc pipeline.
    ],

    [`find_document_corners` #sym.arrow `find_page_corners`],
    [
      Canny edge detection + contour search on the raw photo #sym.arrow Otsu thresholding on the UVDoc-unwarped output (mask and inverted mask), using external contours, convex hull, and `minAreaRect` fallback; improves robustness against wrinkles/text edges and enables stable cropping with an area-ratio safeguard.
    ],
  )
] <tbl-doc-scanner-utils-diff>

The algorithms below reflect *our* UVDoc-aligned implementations; the legacy `flatten_image.py` versions remain in the Text Extractor as a heuristic fallback when no `ExtractedDocument` from the scanner is present.

In @alg-page-corners we formalize our contour-based page-cropping routine (`find_page_corners`) that runs on the UVDoc-unwarped output.
#algorithm-figure(
  [Contour-based page corners and planar rectification],
  vstroke: .5pt + luma(220),
  {
    import algorithmic: *

    Procedure(
      "FindPageCorners",
      ("I", "alpha_min"),
      {
        Assign[`(H, W)`][`shape(I)`]
        Assign[`gray`][`to_grayscale(I)`]
        Assign[`blur`][`GaussianBlur(gray, (5,5))`]
        Assign[`mask`][`OtsuThreshold(blur)`]

        Assign[`best`][`None`]
        Assign[`bestArea`][`0`]
        For(`candidate in {mask, 255-mask}`, {
          Assign[`cand`][`MorphClose(candidate, kernel=(7,7))`]
          Assign[`contours`][`FindContours(cand, external)`]
          If(`contours not empty`, {
            Assign[`c`][`largestByArea(contours)`]
            If(`area(c) > bestArea`, {
              Assign[`best`][`c`]
              Assign[`bestArea`][`area(c)`]
            })
          })
        })

        If(`best is None`, { Return[`None`] })
        Assign[`hull`][`ConvexHull(best)`]
        Assign[`approx`][`ApproxPolyDP(hull, eps=0.02*perimeter(hull))`]
        If(`len(approx) != 4`, { Assign[`approx`][`BoxPoints(minAreaRect(hull))`] })

        If(`bestArea / (H*W) < alpha_min`, { Return[`None`] })
        Return[`OrderPointsClockwise(approx)`]
      },
    )
  },
) <alg-page-corners>

*Corner ordering (`order_points_clockwise`).* We order the detected quadrilateral into (UL, UR, LR, LL) via the standard sum/diff heuristic:
UL #sym.colon.eq argmin(x+y), LR #sym.colon.eq argmax(x+y), UR #sym.colon.eq argmin(x-y), LL #sym.colon.eq argmax(x-y).
This yields a stable orientation for convex page contours even under mild rotations.

*Planar rectification (`warp_from_corners`).* To compute the homography fallback (`transformation_matrix`), we estimate an output rectangle size from the maximum side lengths, solve `H` via `cv2.getPerspectiveTransform`, and warp via `cv2.warpPerspective`.
In the UVDoc pipeline we return and persist both `H` (float32) and the output size, so downstream tools can crop consistently and still backtransform edits.

==== Flow-field sampling and stride downsampling

Two small helper procedures are central for keeping the UVDoc flow field usable in a session-based system:

- `ComputeMapStride`: chooses a stride such that the stored number of `map_xy` samples stays below a pixel budget `max_map_pixels`.
- `SampleMapXY`: maps points from *unwarped space* to *original image space* by nearest-neighbor indexing into the backward flow field.

We use nearest-neighbor sampling because we mostly map sparse geometric objects (page corners, coarse grids). For high-frequency backtransforms we rely on the renderer to interpolate the flow field as needed.

*Stride selection.* We downsample `map_xy` with a stride `s` chosen to satisfy a storage budget:
`s = ceil(sqrt((H*W) / max_map_pixels))` (or `s = 1` if already under budget).

*Map sampling.* For sparse point mappings, we sample `map_xy` via nearest neighbor:
`(x_in, y_in) = map_xy[round(y), round(x)]`, with clipping to image bounds.

=== Downstream Use

For rendering translated text back onto the original photo, the Image Renderer prefers the UVDoc flow field (`mapXYId`) and falls back to the homography (`transformation_matrix`) when needed.
Additionally, the component exposes an optional OpenVINO-based super-resolution step (`super_resolve_document`) and stores the result as `SessionState.superResolvedDocument` for downstream OCR/rendering.

==== Backtransform (deskewed #sym.arrow.r original canvas)

Downstream tools may need to project a modified deskewed image back onto the original photo canvas (e.g., for compositing rendered text).
Our backtransform in `traenslenzor/doc_scanner/backtransform.py` supports two modes:

- *Flow-field backtransform (`map_xy`).* For each pixel `(x_out, y_out)` in the deskewed output, `map_xy[y_out, x_out] = (x_in, y_in)` stores the originating pixel in the raw input image. We round to integer coordinates, write `extracted_rgb[y_out, x_out]` into `back[y_in, x_in]`, and mark the boolean mask at the same location.
- *Corner-based fallback (homography).* If no `map_xy` is available, we compute a planar homography from the deskewed rectangle corners to the original document quadrilateral (`documentCoordinates`) and warp both the image and an all-ones mask via `cv2.warpPerspective`.

The mask enables clean compositing: composite = raw * (1 - mask) + back * mask.

#figure(
  caption: [Backtransform artifacts for compositing a deskewed image back onto the original canvas.],
)[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 10pt,
    [
      #image("/imgs/doc-scanner/invoice_clutter_downscaled_blur_deskewed_backtransformed.png", width: 100%)
      #text(size: 8pt)[(a) Backtransformed image.]
    ],
    [
      #image("/imgs/doc-scanner/invoice_clutter_downscaled_blur_deskewed_backtransform_mask.png", width: 100%)
      #text(size: 8pt)[(b) Backtransform mask.]
    ],
    [
      #image("/imgs/doc-scanner/invoice_clutter_downscaled_blur_deskewed_backtransform_composite.png", width: 100%)
      #text(size: 8pt)[(c) Composite (raw + backtransform).]
    ],
  )
] <fig-doc-scanner-backtransform-grid>

=== Discussion and Limitations

- Flow-field based backtransforms are only as accurate as the predicted sampling grid and may show artifacts near strong folds or occlusions.
- Downsampling `map_xy` trades accuracy for storage and introduces approximation error (usually acceptable for compositing).
- The homography `transformation_matrix` is a coarse approximation and should be treated as a fallback when the flow field is unavailable.
