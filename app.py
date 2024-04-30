import io
import os
from flask import Flask, request, abort, send_from_directory
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, ImageSendMessage
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from facefuncs import get_path_return_output

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
    except InvalidSignatureError:
        abort(400)
    except Exception as e:
        app.logger.error(f"Error handling request: {e}")
        return '', 500
    return '', 200


@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    image_data = io.BytesIO(message_content.content)
    image_path = 'input_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(image_data.read())

    highest_score_image, highest_score_image_path = get_path_return_output(
        image_path)
    img = Image.open(highest_score_image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Most similar image: {highest_score_image}")
    plt.savefig('output_image.jpg')
    plt.close()

    line_bot_api.reply_message(
        event.reply_token,
        ImageSendMessage(
            original_content_url=f"https://{os.getenv('kaomoji')}.herokuapp.com/images/output_image.jpg",
            preview_image_url=f"https://{os.getenv('kaomoji')}.herokuapp.com/images/output_image.jpg"
        )
    )


@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory('images', filename)


if __name__ == "__main__":
    app.run()
