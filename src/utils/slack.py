from .constants import WEBHOOK_URL
import requests

def send_slack_message(message):
    '''
    api 문서:
    https://api.slack.com/messaging/webhooks
    '''
    if not WEBHOOK_URL:
        print("Slack webhook URL이 없어 메시지를 전송하지 않습니다.")
        return

    payload = {"text": message}
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code == 200:
            print("Message sent successfully")
        else:
            print(f"Failed to send message: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Slack 전송 실패 (네트워크 오류): {e}")
