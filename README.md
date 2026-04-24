# Feature Matching with SIFT and LoG

A computer vision pipeline demonstrating image feature matching by restricting SIFT keypoint detection strictly to structural contours extracted via a Laplacian of Gaussian (LoG) filter. 

This approach is compared against a standard unmasked SIFT baseline to evaluate the impact of edge-masking on match quality and geometric accuracy.

## Tech Stack
* **Language:** Python 3
* **Libraries:** `opencv-contrib-python`, `numpy`, `matplotlib`

## Pipeline Overview
1. **Edge Extraction:** Applied a LoG filter (Gaussian blur + 64-bit float Laplacian) to extract coarse edges, avoiding the overly thin 1-pixel edges produced by Canny.
2. **Morphological Dilation:** Dilated the binary edge mask to create a spatial band, providing a sufficient 16x16 patch neighborhood for SIFT descriptor calculation.
3. **Masked SIFT Detection:** Executed `cv2.SIFT_create().detectAndCompute()` passing the dilated mask (uint8) to eliminate keypoints in flat, textureless regions.
4. **FLANN Matching & Lowe's Ratio:** Used a KD-Tree FLANN matcher (k=2) and applied Lowe's Ratio Test (threshold = 0.75) to filter out ambiguous matches.
5. **Baseline Comparison:** Ran standard full-image SIFT to quantify the reduction in total keypoints versus the retention of high-quality structural matches.

## How to Run
Ensure you have the `contrib` version of OpenCV installed, as standard OpenCV does not include the SIFT implementation in its public API.
```bash
pip uninstall opencv-python
pip install opencv-contrib-python numpy matplotlib
