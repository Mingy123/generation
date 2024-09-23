import json
import base64
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = 'fonts/Segoe UI.ttf'
MARGIN = 5 // 100

def find_max_font_size(text, width, height, draw):
    width *= 1-MARGIN
    height *= 1-MARGIN
    font_size = 1
    font = ImageFont.truetype(FONT_PATH, font_size)
    while True:
        lines = wrap_text(text, width, font, draw)
        text_height = sum([font_size for line in lines])
        if text_height > height or any(draw.textlength(line, font=font) > width for line in lines):
            font_size -= 1
            break
        font_size += 1
        font = ImageFont.truetype(FONT_PATH, font_size)
    return ImageFont.truetype(FONT_PATH, font_size)

def wrap_text(text, width, font, draw):
    words = text.split()
    lines = []
    current_line = words[0]
    for word in words[1:]:
        test_line = f"{current_line} {word}"
        if draw.textlength(test_line, font=font) <= width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    return lines

# template: dict
# returns a PIL image
def main(template):
    width, height = template['width'], template['height']
    canvas = Image.new('RGB', (width, height), 'white')

    draw = ImageDraw.Draw(canvas)

    for item in template['elements']:
        x0, y0 = item['x'], item['y']
        x1, y1 = x0 + item['width'], y0 + item['height']
        draw.rectangle([(x0, y0), (x1, y1)], outline="black", width=3)

        if item['contentType'] == 'text':
            text = item['content'] or item['name']
            rect_width = x1 - x0
            rect_height = y1 - y0
            font = find_max_font_size(text, rect_width, rect_height, draw)
            lines = wrap_text(text, rect_width, font, draw)
            total_text_height = sum([font.size for line in lines])
            current_y = y0 + (rect_height - total_text_height) / 2

            for line in lines:
                text_width = draw.textlength(line, font=font)
                text_x = x0 + (rect_width - text_width) / 2
                draw.text((text_x, current_y), line, fill="black", font=font)
                current_y += font.size
        else:
            assert item['contentType'] == 'image'
            data = item['content'].removeprefix("data:image/jpeg;base64,")
            image = Image.open(BytesIO(base64.b64decode( data )))
            image = image.resize((item['width'], item['height']))
            canvas.paste(image, (item['x'], item['y']))

    return canvas
    #canvas.show()

if __name__ == '__main__':
    if not os.path.exists('final.json'):
        print("final.json does not exist. exiting.")
        exit()
    infile = open("final.json", 'r')
    template = json.load(infile)
    infile.close()
    image = main(template)
    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
    print("generated image of b64 length", len(img_str))
