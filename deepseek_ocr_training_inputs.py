#训练输入构造 helper：复刻 infer() 里的多模态输入构造，专门把一条样本转换成 forward() 需要的 input_ids / labels / images / images_seq_mask / images_spatial_crop
import inspect


DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."


def _get_infer_module(model):
    module = inspect.getmodule(model.infer)
    if module is None:
        raise RuntimeError("Cannot locate DeepSeek-OCR infer module.")
    return module


def _make_conversation(prompt, image_path, answer=""):
    return [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [str(image_path)],
        },
        {
            "role": "<|Assistant|>",
            "content": answer,
        },
    ]


def _build_multimodal_inputs(
    model,
    tokenizer,
    image_path,
    prompt,
    answer="",
    base_size=1024,
    image_size=640,
    crop_mode=False,
):
    """Rebuild the multimodal inputs used by DeepSeek-OCR infer()."""
    import math
    import torch
    from PIL import ImageOps

    infer_module = _get_infer_module(model)
    format_messages = getattr(infer_module, "format_messages")
    load_pil_images = getattr(infer_module, "load_pil_images")
    BasicImageTransform = getattr(infer_module, "BasicImageTransform")
    text_encode = getattr(infer_module, "text_encode")
    dynamic_preprocess = getattr(infer_module, "dynamic_preprocess")

    conversation = _make_conversation(prompt, image_path, answer=answer)
    formatted_prompt = format_messages(
        conversations=conversation,
        sft_format="plain",
        system_prompt=""
    )

    patch_size = 16
    downsample_ratio = 4
    images = load_pil_images(conversation)

    if not images:
        raise RuntimeError(f"No image loaded for {image_path}")

    image_draw = images[0].copy()
    w, h = image_draw.size
    ratio = 1 - ((max(w, h) - min(w, h)) / max(w, h))

    image_transform = BasicImageTransform(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        normalize=True
    )

    image_token = "<image>"
    image_token_id = 128815
    text_splits = formatted_prompt.split(image_token)

    tokenized_str = []
    images_seq_mask = []
    images_list = []
    images_crop_list = []
    images_spatial_crop = []

    for text_sep, image in zip(text_splits, images):
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(image)

            global_view = ImageOps.pad(
                image,
                (base_size, base_size),
                color=tuple(int(x * 255) for x in image_transform.mean)
            )
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for crop_image in images_crop_raw:
                    images_crop_list.append(image_transform(crop_image).to(torch.bfloat16))

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            num_queries_base = math.ceil((base_size // patch_size) / downsample_ratio)

            tokenized_image = ([image_token_id] * num_queries_base + [image_token_id]) * num_queries_base
            tokenized_image += [image_token_id]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([image_token_id] * (num_queries * width_crop_num) + [image_token_id]) * (
                    num_queries * height_crop_num
                )
        else:
            if image_size <= 640:
                image = image.resize((image_size, image_size))

            global_view = ImageOps.pad(
                image,
                (image_size, image_size),
                color=tuple(int(x * 255) for x in image_transform.mean)
            )
            images_list.append(image_transform(global_view).to(torch.bfloat16))

            width_crop_num, height_crop_num = 1, 1
            images_spatial_crop.append([width_crop_num, height_crop_num])

            num_queries = math.ceil((image_size // patch_size) / downsample_ratio)
            tokenized_image = ([image_token_id] * num_queries + [image_token_id]) * num_queries
            tokenized_image += [image_token_id]

        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    bos_id = 0
    tokenized_str = [bos_id] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))
    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))

    return {
        "input_ids": input_ids,
        "images": [(images_crop, images_ori)],
        "images_seq_mask": images_seq_mask,
        "images_spatial_crop": images_spatial_crop,
        "formatted_prompt": formatted_prompt,
        "ratio": ratio,
    }


def build_training_inputs(
    model,
    tokenizer,
    image_path,
    answer,
    prompt=DEFAULT_PROMPT,
    base_size=1024,
    image_size=640,
    crop_mode=False,
):
    """Build one supervised DeepSeek-OCR training sample.

    Labels are masked for the prompt and image-token prefix, and only the
    assistant answer tokens are supervised.
    """
    prompt_inputs = _build_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        image_path=image_path,
        prompt=prompt,
        answer="",
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )
    full_inputs = _build_multimodal_inputs(
        model=model,
        tokenizer=tokenizer,
        image_path=image_path,
        prompt=prompt,
        answer=answer,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
    )

    import torch

    labels = full_inputs["input_ids"].clone()
    prompt_len = len(prompt_inputs["input_ids"])
    labels[:prompt_len] = -100
    labels[full_inputs["images_seq_mask"]] = -100

    return {
        "input_ids": full_inputs["input_ids"],
        "attention_mask": torch.ones_like(full_inputs["input_ids"], dtype=torch.bool),
        "labels": labels,
        "images": full_inputs["images"],
        "images_seq_mask": full_inputs["images_seq_mask"],
        "images_spatial_crop": full_inputs["images_spatial_crop"],
        "prompt_len": prompt_len,
    }
