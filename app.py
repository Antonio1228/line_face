import io
import os
from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, ImageSendMessage
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))


@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
        return '', 200
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        app.logger.error(f"Error handling request: {e}")
        return '', 500


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = io.BytesIO(message_content.content)
    image_path = 'input_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(image_data.read())

    # 此處原本執行 Notebook 的代碼需要調整為直接處理圖片的代碼

    # 假設處理完畢後保存圖片為 'output_image.jpg'
    output_image_path = 'output_image.jpg'
    line_bot_api.reply_message(
        event.reply_token,
        ImageSendMessage(
            original_content_url=f"https://{os.getenv('kaomoji')}.herokuapp.com/images/{output_image_path}",
            preview_image_url=f"https://{os.getenv('kaomoji')}.herokuapp.com/images/{output_image_path}"
        )
    )


@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory('images', filename)


if __name__ == "__main__":
    app.run()
