import os
os.environ["HF_HOME"] = '/mnt/cache'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
import fitz
from PIL import Image

disable_torch_init()
model_path = "liuhaotian/llava-v1.5-7b"
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name
)

# boilerplate function copied from run_llava.py. Uses the globally defined `model` above.
def eval_model(images, query, batch=False, temperature=0.5, top_p=None, num_beams=1, max_new_tokens=1024):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    out = []
    for image_file in images:
        image = Image.open(image_file).convert("RGB")
        out.append(image)
    images = out
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    if batch:
        input_ids = torch.cat([input_ids for _ in images]).cuda()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [i.strip() for i in outputs]





def get_image_info(image_files):
    prompt = "Describe the following image in full detail, including any text in the image."
    return eval_model(image_files, prompt, batch=True)

def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# returns a list of paths e.g. "pdf_images/page_1.png"
def pdf_to_img(pdf_path):
    doc = fitz.open(pdf_path)
    output_folder = "pdf_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        filename = f"page_{page_num}.png"
        files.append(output_folder+'/'+filename)
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)
    doc.close()
    return files



files = pdf_to_img("info.pdf")
runs = split_list(files, 10)
print("Split into", runs)
import time
for i in range(len(runs)):
    r = runs[i]
    start = time.time()
    print(get_image_info(r))
    end = time.time()
    print("Processed", len(r), "in", end-start)
