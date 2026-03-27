import requests, os

def send_train_message(message: str):
    # optional SMS notification for training status
    assert len(message) < 64, f"Message too long: {(message)}"
    if os.getenv("ALIYUN_SMS_SERVICE"):
        try: requests.post(
                json={"phone_numbers": "your-phone-number", "server_code": "DLC", "error": message, "error_level": "无"},
                url=os.getenv("ALIYUN_SMS_SERVICE", "http://localhost:8000/send-sms"),
                headers={"Content-Type": "application/json"}
            )
        except Exception as e: print(f"Failed to send sms: {e}")