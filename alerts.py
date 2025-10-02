import os, cv2
from datetime import datetime
from twilio.rest import Client

class AlertManager:
    def __init__(self, snapshot_dir='snapshots', twilio_cfg=None, smtp_cfg=None, alert_cooldown=30):
        os.makedirs(snapshot_dir, exist_ok=True)
        self.snapshot_dir = snapshot_dir
        self.twilio_cfg = twilio_cfg
        self.smtp_cfg = smtp_cfg
        self.last_alert_time = 0
        self.alert_cooldown = alert_cooldown

    def save_snapshot(self, frame, decision=None):
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        label = "_".join(decision.reasons) if decision and decision.reasons else "alert"
        fname = os.path.join(self.snapshot_dir, f"{label}_{ts}.jpg")
        cv2.imwrite(fname, frame)
        return fname

    def send_sms(self, body):
        if not self.twilio_cfg:
            print("Twilio config missing; SMS skipped")
            return
        client = Client(self.twilio_cfg['account_sid'], self.twilio_cfg['auth_token'])
        client.messages.create(to=self.twilio_cfg['to'], from_=self.twilio_cfg['from'], body=body)

    def send_email(self, subject, body, attachments=None):
        import smtplib
        from email.message import EmailMessage
        if not self.smtp_cfg:
            print("SMTP config missing; email skipped")
            return
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.smtp_cfg["from"]
        msg["To"] = self.smtp_cfg["to"]
        msg.set_content(body)
        # Attach files
        if attachments:
            for path in attachments:
                with open(path, "rb") as f:
                    data = f.read()
                    msg.add_attachment(data, maintype="image", subtype="jpeg", filename=os.path.basename(path))
        try:
            with smtplib.SMTP_SSL(self.smtp_cfg["server"], self.smtp_cfg.get("port", 465)) as server:
                server.login(self.smtp_cfg["username"], self.smtp_cfg["password"])
                server.send_message(msg)
            print("Email sent!")
        except Exception as e:
            print("Email send error:", e)
    def make_twilio_call(self, message=None):
        if not self.twilio_cfg:
            print("Twilio config missing; call skipped")
            return
        client = Client(self.twilio_cfg['account_sid'], self.twilio_cfg['auth_token'])
        try:
            call = client.calls.create(
                to=self.twilio_cfg['to'],
                from_=self.twilio_cfg['from'],
                twiml=f'<Response><Say>{message or "Emergency at ATM!"}</Say></Response>'
            )
            print("Twilio call initiated! SID:", call.sid)
        except Exception as e:
            print("Twilio call error:", e)

    def send(self, decision, frame):
        import time
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            print("Alert suppressed due to cooldown.")
            return
        self.last_alert_time = now
        snap = self.save_snapshot(frame, decision)
        body = f"ALERT: {decision.level}\nReasons: {decision.reasons}"
        print("ALERT:", body, "snapshot saved to", snap)
        # send SMS/email/call if configured
        self.send_sms(body)
        self.send_email(f"ATM Alert - {decision.level}", body, attachments=[snap])
        self.make_twilio_call(message=body)
 