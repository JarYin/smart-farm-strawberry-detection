# ตั้งค่าสภาพแวดล้อมบนเครื่อง Windows สำหรับพัฒนาและเทรนโมเดล
#
# วิธีใช้ (เปิด PowerShell ที่โฟลเดอร์ CODE):
#     .\scripts\setup_pc.ps1
#
# ถ้าติด error เรื่อง execution policy ให้รันคำสั่งนี้ก่อน:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " ตั้งค่าสภาพแวดล้อมสำหรับพัฒนาบน PC" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " โฟลเดอร์โปรเจกต์: $projectRoot"

# --- ตรวจเวอร์ชัน Python ---
$pythonVersion = (python --version 2>&1) -replace "Python ", ""
Write-Host "`n[1/4] Python ที่พบ: $pythonVersion"

$versionParts = $pythonVersion.Split(".")
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]

if ($major -ne 3 -or $minor -lt 9) {
    Write-Host " [ผิดพลาด] ต้องใช้ Python 3.9 ขึ้นไป" -ForegroundColor Red
    exit 1
}
if ($minor -ge 13) {
    Write-Host " [เตือน] Python 3.13 ขึ้นไปอาจมีปัญหาตอนแปลงโมเดลเป็น TFLite" -ForegroundColor Yellow
    Write-Host "         ส่วนอื่นใช้งานได้ปกติ ถ้าติดปัญหาให้แปลงโมเดลบน Google Colab แทน" -ForegroundColor Yellow
}

# --- สร้าง virtual environment ---
Write-Host "`n[2/4] สร้าง virtual environment ที่ .venv"
if (Test-Path ".venv") {
    Write-Host "      มีอยู่แล้ว ข้ามขั้นตอนนี้"
} else {
    python -m venv .venv
    Write-Host "      สร้างเสร็จ"
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Host " [ผิดพลาด] ไม่พบ $pythonExe" -ForegroundColor Red
    exit 1
}

# --- ติดตั้งไลบรารี ---
Write-Host "`n[3/4] ติดตั้งไลบรารี (ครั้งแรกใช้เวลาสักพัก เพราะ PyTorch ไฟล์ใหญ่)"
& $pythonExe -m pip install --upgrade pip --quiet
& $pythonExe -m pip install -r requirements-pc.txt

# --- สร้างโฟลเดอร์ที่ต้องใช้ ---
Write-Host "`n[4/4] สร้างโฟลเดอร์ที่จำเป็น"
foreach ($folder in @("models", "datasets", "logs", "runs")) {
    if (-not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder | Out-Null
        Write-Host "      สร้าง $folder/"
    }
}

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " ติดตั้งเสร็จเรียบร้อย" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host @"

เปิดใช้งาน environment ทุกครั้งก่อนทำงาน:
    .\.venv\Scripts\Activate.ps1

ขั้นตอนถัดไป:
    1) ทดสอบว่าโค้ดทำงานถูกต้อง :  pytest tests/ -v
    2) ทดสอบกล้อง               :  python tools\camera_test.py --list
    3) วางชุดข้อมูลจาก Roboflow  :  datasets\
    4) ตรวจชุดข้อมูล             :  python tools\dataset_check.py
    5) เทรนโมเดล                :  python tools\train.py
"@
