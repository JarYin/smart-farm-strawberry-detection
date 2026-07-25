# โฟลเดอร์ไฟล์โมเดล

ไฟล์โมเดลไม่ถูกเก็บใน git เพราะมีขนาดใหญ่ (ดู `.gitignore`)

## ไฟล์ที่จะปรากฏที่นี่

| ไฟล์ | ได้จาก | ใช้ที่ไหน |
|---|---|---|
| `best.pt` | `tools/train.py` | เครื่อง PC (ตอนพัฒนา ประเมินผล และแปลงโมเดล) |
| `best_float32.tflite` | `tools/export_tflite.py` | Raspberry Pi — แนะนำให้เริ่มจากตัวนี้ |
| `best_int8.tflite` | `tools/export_tflite.py --int8` | Raspberry Pi — เล็กและเร็วกว่า แม่นยำลดลงเล็กน้อย |

## เปรียบเทียบ

| รูปแบบ | ขนาดโดยประมาณ | ความเร็วบน Pi 4 | ความแม่นยำ |
|---|---|---|---|
| `.pt` (PyTorch) | ~6 MB + ไลบรารี ~2 GB | ช้ามาก | อ้างอิง 100% |
| `.tflite` float32 | ~12 MB | 7-10 FPS | ~100% |
| `.tflite` int8 | ~3 MB | 12-18 FPS | ~95-98% |

## การส่งไฟล์ไป Raspberry Pi

```powershell
scp models\best_float32.tflite pi@<ไอพีของ Pi>:~/CODE/models/
```

แล้วแก้ `config.yaml` บน Pi:

```yaml
model:
  weights: models/best_float32.tflite
```

## การเก็บสำรอง

ไฟล์โมเดลคือผลลัพธ์ที่ใช้เวลาเทรนนานที่สุด **ควรสำรองไว้บน Google Drive**
พร้อมบันทึกว่าเทรนจากชุดข้อมูลไหน กี่ epoch และได้ mAP เท่าไร
(ดูได้จาก `logs/eval_report.json`)
