import wandb
import os
from arguments import parse_args
import json
import csv
import pandas as pd

folder_to_sync = "runs/"
exps = next(os.walk(f'{folder_to_sync}'))[1]

for exp in exps:
	wandb_group_name = '_'.join(exp.split('_')[:3])
	seeds = next(os.walk(f'{folder_to_sync}/{exp}'))[1]
	for seed in seeds:
		print(exp, seed)
		files_loc = f'{folder_to_sync}/{exp}/{seed}'
		try:
			try:
				with open(os.path.join(files_loc, 'config.log'), "r") as file:
					data = file.read()
					data = json.loads(data)
					cfg = data["args"]
					cfg["wandb_group_name"] = wandb_group_name
					run = wandb.init(project="mvd", group=wandb_group_name, job_type="train", config=cfg, reinit=True)
			except:
				with open(os.path.join(files_loc, 'config_continued.log'), "r") as file:
					data = file.read()
					data = json.loads(data)
					cfg = data["args"]
					cfg["wandb_group_name"] = wandb_group_name
					run = wandb.init(project="multi-view", group=wandb_group_name, job_type="train", config=cfg, reinit=True)

			eval_data = pd.read_csv(os.path.join(files_loc, "eval.csv"))
			eval_headers = list(eval_data.columns)
			eval_headers.remove("step")
			for _, row in eval_data.iterrows():
				step = row["step"]
				for header in eval_headers:
					wandb.log({f"eval/{header}": row[header], "training_timestep": step}, commit=False)
				wandb.log({}, commit=True)

			eval_scenarios_data = pd.read_csv(os.path.join(files_loc, "eval_scenarios.csv"))
			eval_scenarios_headers = list(eval_scenarios_data.columns)
			eval_scenarios_headers.remove("step")
			for _, row in eval_scenarios_data.iterrows():
				step = row["step"]
				for header in eval_scenarios_headers:
					wandb.log({f"eval_scenarios/{header}": row[header], "training_timestep": step}, commit=False)
				wandb.log({}, commit=True)

			train_data = pd.read_csv(os.path.join(files_loc, "train.csv"))
			train_headers = list(train_data.columns)
			train_headers.remove("step")
			for _, row in train_data.iterrows():
				step = row["step"]
				for header in train_headers:
					if "actor" in header or "critic" in header or "alpha" in header or "encoder" in header:
						wandb_header = "train_" + header
						wandb_header = wandb_header.replace("actor_", "actor/").replace("critic_", "critic/").replace("alpha_", "alpha/").replace("encoder_", "encoder/")
					else:
						wandb_header = "train/" + header
					wandb.log({wandb_header: row[header], "training_timestep": step}, commit=False)
				wandb.log({}, commit=True)

			run.finish()

		except Exception as e:
			print(e)
