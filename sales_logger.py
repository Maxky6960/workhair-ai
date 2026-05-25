import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from sheets_client import get_sheet


load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def validate_sale(service: str, quantity: int, price: float) -> None:
    if not service or not service.strip():
        raise ValueError("ชื่อบริการห้ามว่าง")
    if quantity <= 0:
        raise ValueError("จำนวนต้องมากกว่า 0")
    if price <= 0:
        raise ValueError("ราคาต้องมากกว่า 0")


def log_sale(
    service: str,
    quantity: int,
    price: float,
    staff: str = "",
    payment_method: str = "",
    note: str = "",
    sale_date: Optional[str] = None,
) -> dict:
    validate_sale(service, quantity, price)

    total = quantity * price
    date_value = sale_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        date_value,
        service.strip(),
        quantity,
        total,
        staff.strip(),
        payment_method.strip(),
        note.strip(),
    ]
    get_sheet().append_row(row)

    return {
        "date": date_value,
        "service": service.strip(),
        "quantity": quantity,
        "price": price,
        "total": total,
        "staff": staff.strip(),
        "payment_method": payment_method.strip(),
        "note": note.strip(),
    }


def parse_sale_input(raw: str) -> dict:
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) < 3:
        raise ValueError("รูปแบบต้องเป็น บริการ:จำนวน:ราคา[:ช่าง[:วิธีชำระเงิน[:หมายเหตุ]]]")

    service = parts[0]
    quantity = int(parts[1])
    price = float(parts[2])
    staff = parts[3] if len(parts) > 3 else ""
    payment_method = parts[4] if len(parts) > 4 else ""
    note = parts[5] if len(parts) > 5 else ""
    return {
        "service": service,
        "quantity": quantity,
        "price": price,
        "staff": staff,
        "payment_method": payment_method,
        "note": note,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("วิธีใช้: python sales_logger.py \"ตัดผมชาย:1:150:ช่างมิน:โอน:walk-in\"")
        sys.exit(1)

    try:
        sale = log_sale(**parse_sale_input(sys.argv[1]))
        print(f"บันทึกสำเร็จ: {sale['service']} x{sale['quantity']} = {sale['total']} บาท")
    except Exception as exc:
        print(f"บันทึกไม่สำเร็จ: {exc}")
        sys.exit(1)
