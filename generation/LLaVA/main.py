import os
os.environ["HF_HOME"] = '/mnt/cache'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Describe the following image in full detail, including any text in the image."
#image_files = ['outimg-' + str(i).zfill(2) + '.png' for i in range(1,5)]
image_files = ['https://www.htx.gov.sg/images/default-source/htx-image-library/htx-valuesedm_fa-(no-footer).png']
sep = ','

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": sep.join(image_files),
    "sep": sep,
    "temperature": 0.5,
    "top_p": None,
    "num_beams": 5,
    "max_new_tokens": 1024,
    "batch": True
})()

import time

start = time.time()
eval_model(args)
end = time.time()

input(f"Completed in {end-start}s. Press Enter to exit.")
