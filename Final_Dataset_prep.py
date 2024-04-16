from vstar_bench_eval import *
import torch
from PIL import Image
import csv
from tqdm import tqdm
import pandas as pd
from PIL import Image
import argparse

def parse_args_vqallm(args):
	parser = argparse.ArgumentParser()
	parser.add_argument("--vqa-model-path", type=str, default="craigwu/seal_vqa_7b")
	parser.add_argument("--vqa-model-base", type=str, default=None)
	parser.add_argument("--conv_type", default="v1", type=str,)
	parser.add_argument("--vsm-model-path", type=str, default="craigwu/seal_vsm_7b")
	parser.add_argument("--minimum_size_scale", default=4.0, type=float)
	parser.add_argument("--minimum_size", default=224, type=int)
	return parser.parse_args(args)



args = parse_args_vqallm({})
# init VQA LLM
vqa_llm = VQA_LLM(args)
# init VSM
vsm_args = parse_args({})
vsm_args.version = args.vsm_model_path
vsm = VSM(vsm_args)

torch.inference_mode()


csv_file_path = '/home/qblocks/interns/zaccaria/VStar_Model/vstar/25_sampled_list.csv'
csv_out = "/home/qblocks/interns/zaccaria/VStar_Model/vstar/caption_25.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Create a list to store updated rows
updated_rows = []
p = 0
p1 = 0
p2 = 0
p3 = 0

# Process each row in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="row",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]"):
    serial_no, image_name, location, category , subcategory ,alt_text = row[:6]
    # Unnamed: 0,image_name,image_location,category,sub_category,alt_text

    try:
        # info = location.split("/")[-2:-1]
        # print(info)
        # category = info[0]
        # subcategory = info[1]

        category = category.replace("_", " ")
        subcategory = subcategory.replace("_", " ")

        image = Image.open(location)
        prompt = f"Describe the contents of the image. This is a {category} known as {subcategory}."
        caption = vqa_llm.free_form_inference(
            image=image,
            question=prompt,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=200,
            object_crops=None,
            images_long=None,
            objects_long=None
        )

        search_string = "Sorry, I can not answer the question. Some visual information about the following objects is missing or unclear"
        if search_string in caption:
            prompt = f"Describe the contents of the image. This is {subcategory} in {category}."
            caption = vqa_llm.free_form_inference(
                image=image,
                question=prompt,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=200,
                object_crops=None,
                images_long=None,
                objects_long=None
            )
            if search_string in caption:
                prompt = "Describe the contents of the image."
                caption = vqa_llm.free_form_inference(
                    image=image,
                    question=prompt,
                    temperature=0,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=200,
                    object_crops=None,
                    images_long=None,
                    objects_long=None
                )
                p2 += 1
                tag = "Prompt-3"
            else:
                p1 += 1
                tag = "Prompt-2"
        else:
            p += 1
            tag = "Prompt-1"

        # Add the additional information to the row
        updated_row = [serial_no, category, subcategory, location, image_name, prompt, caption, tag]
        # Add the updated row to the list
        updated_rows.append(updated_row)
        tqdm.write(f"Processed: {index + 1}/{len(df)}")
    except Exception as e:
        print(f"Error processing row {index + 1}: {e}")

# Create a new DataFrame with updated rows
updated_df = pd.DataFrame(updated_rows, columns=["Serial no.","Category", "Subcategory", "Location", "Image Name", "Prompt", "Caption", "Tag"])

# Save the updated DataFrame to a new CSV file
updated_df.to_csv(csv_out, index=False)