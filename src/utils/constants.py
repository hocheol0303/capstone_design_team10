# wrinkle 관련 상수
SECTORS = ["forehead", "right_eye", "left_eye", "nasolabial", "perioral", "right_vol", "left_vol"]
SECTOR2IDX = {sec:i for i, sec in enumerate(SECTORS)}
IDX2SECTOR = {i:sec for i, sec in enumerate(SECTORS)}

# pigment regression 관련 상수 (RangeMSELoss, RangeMetrics 공유)
PIGMENT_RANGES = [
    [91.0, 100.0, 95.5],    # label 0
    [76.0, 90.0, 83.0],     # label 1
    [56.0, 75.0, 65.5],     # label 2
    [31.0, 55.0, 43.0],     # label 3
    [1.0, 30.0, 15.5],      # label 4
]

try:
    WEBHOOK_URL = open('/home/hocheol/inskin_ai/no_track/webhook_url.txt', 'r').read().strip()
except FileNotFoundError:
    WEBHOOK_URL = None
