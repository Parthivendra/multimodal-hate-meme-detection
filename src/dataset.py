import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MemeDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer=None, transform=None, max_length=128):
        self.df = df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.lable_map = {
    		"not_offensive": 0,
	    	"slight": 1,
	    	"very_offensive": 2,
	    	"hateful_offensive": 3
	    	}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- TEXT ----
        text = str(row["text_corrected"])

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
        else:
            input_ids, attention_mask = None, None

        # ---- IMAGE ----
        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ---- LABEL ---- 
        label = torch.tensor(self.lable_map[row["offensive"]], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label
        }
