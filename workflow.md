Input Image: 640x640x3
    ↓
Conv (32 kernels, 3x3, stride=2, padding=1):
Output: 320x320x32  (stride=2 halves the size)
    ↓
C2f (64 kernels, 3x3, stride=1, padding=1):
Output: 320x320x64  (stride=1 keeps size same)
    ↓
Conv (128 kernels, 3x3, stride=2, padding=1):
Output: 160x160x128  (stride=2 halves the size)
    ↓
C2f (128 kernels, 3x3, stride=1, padding=1):
Output: 160x160x128  (stride=1 keeps size same)
    ↓
Conv (256 kernels, 3x3, stride=2, padding=1):
Output: 80x80x256  (stride=2 halves the size)
    ↓
C2f (256 kernels, 3x3, stride=1, padding=1):
Output: 80x80x256  (stride=1 keeps size same)
    ↓
Conv (512 kernels, 3x3, stride=2, padding=1):
Output: 40x40x512  (stride=2 halves the size)
    ↓
C2f (512 kernels, 3x3, stride=1, padding=1):
Output: 40x40x512  (stride=1 keeps size same)
    ↓
Conv (1024 kernels, 3x3, stride=2, padding=1):
Output: 20x20x1024  (stride=2 halves the size)
    ↓
SPPF (1024 kernels, 5x5, stride=1, padding=2):
Output: 20x20x1024  (stride=1 keeps size same)
    ↓
[Neck + Head...]
Final Output: List of [bbox, confidence, class_id]
