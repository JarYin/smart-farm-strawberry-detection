"""
ตัวช่วยเล็กๆ ให้สคริปต์ใน tools/ มองเห็นแพ็กเกจ src/ ได้

สคริปต์ทุกตัวใน tools/ ขึ้นต้นด้วย  import _bootstrap  เป็นบรรทัดแรก
จึงรันได้จากทุกโฟลเดอร์โดยไม่ต้องตั้ง PYTHONPATH เอง
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
