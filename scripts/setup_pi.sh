#!/usr/bin/env bash
# ตั้งค่า Raspberry Pi 4 สำหรับรันระบบจริง
#
# วิธีใช้ (บน Raspberry Pi):
#     chmod +x scripts/setup_pi.sh
#     ./scripts/setup_pi.sh
#
# สคริปต์นี้ไม่ติดตั้ง ultralytics หรือ PyTorch โดยตั้งใจ
# เพราะกินพื้นที่กว่า 2 GB และช้ามากบน Pi — เราใช้แค่ tflite-runtime

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${HOME}/smartfarm-env"

echo "========================================================"
echo " ตั้งค่า Raspberry Pi สำหรับ Smart Farm Autonomous Robot"
echo "========================================================"
echo " โฟลเดอร์โปรเจกต์: ${PROJECT_ROOT}"

# --- ตรวจว่าอยู่บน Raspberry Pi จริงหรือไม่ ---
if [ -f /proc/device-tree/model ]; then
    echo " บอร์ด: $(tr -d '\0' < /proc/device-tree/model)"
else
    echo " [เตือน] ดูเหมือนจะไม่ได้อยู่บน Raspberry Pi — สคริปต์นี้อาจไม่ทำงานตามที่ควร"
fi

echo ""
echo "[1/5] อัปเดตรายการแพ็กเกจของระบบ"
sudo apt update

echo ""
echo "[2/5] ติดตั้งแพ็กเกจของระบบ (กล้อง + GPIO + เครื่องมือพื้นฐาน)"
# picamera2 และ gpiozero ต้องติดตั้งผ่าน apt เท่านั้น ติดตั้งผ่าน pip จะใช้ไม่ได้
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-gpiozero \
    libatlas-base-dev \
    libopenjp2-7

echo ""
echo "[3/5] สร้าง virtual environment ที่ ${VENV_DIR}"
if [ -d "${VENV_DIR}" ]; then
    echo "      มีอยู่แล้ว ข้ามขั้นตอนนี้"
else
    # --system-site-packages จำเป็นมาก เพื่อให้ venv มองเห็น picamera2 ที่มาจาก apt
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo ""
echo "[4/5] ติดตั้งไลบรารี Python"
pip install --upgrade pip
if ! pip install -r requirements-pi.txt; then
    echo ""
    echo " [เตือน] ติดตั้งบางตัวไม่สำเร็จ — น่าจะเป็น tflite-runtime"
    echo "         ลองทางเลือกอื่น:"
    echo "           pip install ai-edge-litert"
    echo "         หรือ:"
    echo "           pip install tensorflow    # หนักกว่ามาก ใช้เป็นทางเลือกสุดท้าย"
fi

echo ""
echo "[5/5] สร้างโฟลเดอร์ที่จำเป็น"
mkdir -p models logs

echo ""
echo "========================================================"
echo " ติดตั้งเสร็จเรียบร้อย"
echo "========================================================"
cat <<'EOF'

สิ่งที่ต้องทำต่อ:

  1) เปิดใช้งานกล้อง (ถ้ายังไม่ได้เปิด):
       sudo raspi-config  ->  Interface Options  ->  Legacy Camera / Camera
       แล้ว reboot

  2) คัดลอกไฟล์โมเดลจากเครื่อง PC มาไว้ที่ models/
       (รันคำสั่งนี้ที่เครื่อง PC)
       scp models/best_float32.tflite pi@<ไอพีของ Pi>:~/CODE/models/

  3) แก้ config.yaml บน Pi:
       camera.source : picamera2
       model.weights : models/best_float32.tflite
       runtime.show_window : false

  4) เปิดใช้งาน environment ทุกครั้งก่อนรัน:
       source ~/smartfarm-env/bin/activate

  5) ทดสอบทีละส่วนตามลำดับนี้:
       python tools/camera_test.py --source picamera2
       python tools/relay_test.py            # ถอดสายปั๊มออกก่อน ฟังแค่เสียงคลิก
       python tools/benchmark.py
       python -m src.main --headless

EOF
