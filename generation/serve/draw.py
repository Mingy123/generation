import json
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = 'arial.ttf'

# returns a PIL image
def main(poster_path):
    infile = open(poster_path, 'r')
    template = json.load(infile)
    infile.close()

    width, height = template['width'], template['height']
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)

    def find_max_font(text, width, height, draw, font_path=FONT_PATH):
        font_size = 1
        font = ImageFont.truetype(font_path, font_size)
        while True:
            text_width = draw.textlength(text, font=font)
            text_height = font_size
            if text_width > width or text_height > height:
                font_size -= 1
                break
            font_size += 1
            font = ImageFont.truetype(font_path, font_size)
        return font

    for item in template['elements']:
        x0, y0 = item['x'], item['y']
        x1, y1 = x0 + item['width'], y0 + item['height']
        draw.rectangle([(x0, y0), (x1, y1)], outline="black", width=3)

        text = item['content'] or item['name']
        font = find_max_font(text, x1-x0, y1-y0, draw)
        text_width = draw.textlength(text, font=font)
        text_height = font.size
        text_x = x0 + (x1 - x0 - text_width) / 2
        text_y = y0 + (y1 - y0 - text_height) / 2
        draw.text((text_x, text_y), text, fill="black", font=font)

    #canvas.save('out.png')
