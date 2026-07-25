# โน้ตบุ๊กสำหรับ Google Colab

## [train_smartfarm_colab.ipynb](train_smartfarm_colab.ipynb)

ทำงานแทนสคริปต์ `tools/train.py`, `tools/evaluate.py` และ `tools/export_tflite.py`
บนเครื่อง โดยใช้ GPU ฟรีของ Colab

### วิธีใช้

1. เปิด [colab.research.google.com](https://colab.research.google.com)
2. เมนู **File → Upload notebook** แล้วเลือกไฟล์ `train_smartfarm_colab.ipynb`
3. **เปิด GPU ก่อนเสมอ**: **Runtime → Change runtime type → T4 GPU → Save**
4. รันทีละเซลล์จากบนลงล่าง (`Shift + Enter`)

### สิ่งที่ต้องเตรียม

เลือกวิธีนำเข้าชุดข้อมูลวิธีใดวิธีหนึ่ง แล้วแก้ค่า `METHOD` ในเซลล์ "ขั้นที่ 2":

| `METHOD` | ต้องเตรียม |
|---|---|
| `"roboflow"` | API Key + ชื่อ workspace / project / version |
| `"upload"` | ไฟล์ `.zip` รูปแบบ YOLOv8 ในเครื่อง |
| `"drive"` | ไฟล์ `.zip` ที่อยู่ใน Google Drive แล้ว |

**หา Roboflow API Key ได้ที่:** roboflow.com → Settings → Roboflow API → Private API Key

**หาชื่อ workspace/project/version จาก URL ของชุดข้อมูล:**

```
https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
```

### ลำดับการทำงานในโน้ตบุ๊ก

| ขั้น | ทำอะไร | เวลาโดยประมาณ |
|---|---|---|
| 0 | ตรวจ GPU + ติดตั้ง ultralytics | 1-2 นาที |
| 1 | ตั้งค่าโปรเจกต์ (IMGSZ=320 ฯลฯ) | ทันที |
| 2 | นำเข้าชุดข้อมูล | 1-5 นาที |
| 3 | ตรวจสอบ + **ซ่อมพาธใน data.yaml** | ทันที |
| 4 | เทรน YOLOv8n | 15-40 นาที |
| 5 | ประเมินผล + ไล่ค่า confidence | 3-8 นาที |
| 6 | แปลงเป็น TFLite (float32 + int8) | 5-10 นาที |
| 7 | **ตรวจสอบไฟล์ .tflite ที่ได้** | ทันที |
| 8 | ดาวน์โหลด / บันทึกเข้า Drive | 1 นาที |

### จุดที่โน้ตบุ๊กนี้ช่วยแก้ปัญหาให้

**ซ่อมพาธใน `data.yaml` (ขั้นที่ 3)** — Roboflow สร้างไฟล์มาด้วยพาธแบบ `../train/images`
ซึ่งชี้ผิดที่เมื่ออยู่บน Colab ทำให้ ultralytics ฟ้องว่าไม่มีภาพสำหรับเทรน
โน้ตบุ๊กเขียนพาธใหม่เป็นแบบเต็มให้อัตโนมัติ

**ตรวจสอบไฟล์ `.tflite` (ขั้นที่ 7)** — โหลดไฟล์ขึ้นมารันจริงแล้วเช็คว่ารูปร่างอินพุต
เป็น `(1, 320, 320, 3)` และเอาต์พุตมีแกนขนาด `4 + จำนวนคลาส` ตรงกับที่
[../src/detector.py](../src/detector.py) คาดไว้ — เจอปัญหาตรงนี้ดีกว่าไปเจอตอนอยู่หน้า Pi

**กู้ตัวแปรคืนหลัง Restart session (ขั้นที่ 6)** — ถ้าการแปลง TFLite ล้มเหลวเพราะไลบรารี
ที่เพิ่งติดตั้ง ต้องกด Restart session ซึ่งจะล้างตัวแปรทั้งหมด เซลล์แปลงโมเดล
จะค้นหาไฟล์ที่เทรนไว้แล้วกลับมาเอง **ไม่ต้องเทรนซ้ำ**

**พิมพ์ค่าที่ต้องไปแก้ใน `config.yaml` ให้** — ทั้ง `classes.names` (ขั้นที่ 3)
และ `per_class_conf` ที่เหมาะที่สุด (ขั้นที่ 5) คัดลอกไปวางได้เลย

### หลังรันเสร็จ

จะได้ไฟล์ `smartfarm_model.zip` ที่มี:

```
best.pt                  -> วางที่ CODE/models/ ใช้ทดสอบบน PC
best_float32.tflite      -> ส่งไป Raspberry Pi
best_int8.tflite         -> ส่งไป Raspberry Pi (ทางเลือกที่เร็วกว่า)
results.png              -> กราฟการเรียนรู้ ใส่รายงานบทที่ 4
confusion_matrix.png     -> Confusion Matrix ใส่รายงานบทที่ 4
eval_per_class.csv       -> ตารางประสิทธิภาพรายคลาส
eval_conf_sweep.csv      -> ตารางเปรียบเทียบค่า confidence
eval_report.json         -> ข้อมูลดิบทั้งหมด
```

### ข้อควรรู้เกี่ยวกับ Colab

- **เซสชันหลุดได้** ถ้าปล่อยทิ้งไว้นานเกินไป หรือใช้ GPU ครบโควตารายวัน
  ถ้าเทรนนานให้เปิดแท็บทิ้งไว้และอย่าปิดเครื่อง
- **ไฟล์ใน `/content` หายทั้งหมดเมื่อเซสชันจบ** — อย่าลืมรันขั้นที่ 8 เพื่อดาวน์โหลด
  หรือบันทึกเข้า Drive ก่อนปิด
- ถ้าโควตา GPU หมด จะยังรันได้แต่ใช้ CPU ซึ่งช้ามาก ให้รอวันถัดไปหรือใช้บัญชีอื่น
