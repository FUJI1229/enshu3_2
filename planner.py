#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   planner.py
@Time    :   2023/05/16 09:12:11
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   Planner module for different tasks
'''

import os
import re
import requests
from typing import Any
from mediator import *
from utils import global_param
from openai import AzureOpenAI
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()

class Base_Planner(ABC):
    """The base class for Planner."""

    def __init__(self, offline=True, soft=False, prefix=''):
        super().__init__()
        self.offline = offline
        self.soft = soft
        self.prompt_prefix = prefix
        self.plans_dict = {}
        self.mediator = None

        self.dialogue_system = self.prompt_prefix
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = False
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_MODEL")

    def reset(self, show=False):
        self.dialogue_user = ''
        self.dialogue_logger = ''
        self.show_dialogue = show
        if self.show_dialogue:
            print(self.dialogue_system)
        self.mediator.reset()

    def init_llm(self):
        self.dialogue_system += self.prompt_prefix
        server_error_cnt = 0
        while server_error_cnt < 10:
            try:
                headers = {'Content-Type': 'application/json'}
                data = {'model': self.model, "messages": [{"role": "system", "content": self.prompt_prefix}]}
                response = requests.post(self.llm_url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    break
                else:
                    raise RuntimeError(f"Failed to initialize: status code {response.status_code}")
            except Exception as e:
                server_error_cnt += 1
                print(f"Initialization failed: {e}")

    def query_codex(self, prompt_text, retries=3):
        for attempt in range(retries):
            try:
                messages = [
                    {"role": "system", "content": self.prompt_prefix},
                    {"role": "user", "content": prompt_text}
                ]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                result = response.choices[0].message.content
                plan = re.search(r"Action[s]*\:\s*\{([\w\s\<\>\,]*)\}", result, re.I | re.M).group(1)
                return plan
            except Exception as e:
                print(f"[Azure LLM Error] {e}")
        print(f"[LLM Response Invalid] Could not parse after {retries} tries.")
        return ""

    def plan(self, text, n_ask=10):
        if text in self.plans_dict:
            return self.plans_dict[text]

        print(f"new obs: {text}")
        plans = {}
        for _ in range(n_ask):
            plan = self.query_codex(text)
            if plan:
                plans[plan] = plans.get(plan, 0) + 1 / n_ask

        plans_list, probs_list = list(plans.keys()), list(plans.values())
        self.plans_dict[text] = (plans_list, probs_list)

        for k, v in self.plans_dict.items():
            print(f"{k}: {v}")

        return plans_list, probs_list

    def __call__(self, obs):
        text = self.mediator.RL2LLM(obs)
        plans, probs = self.plan(text)
        self.dialogue_user = text + "\n" + str(plans) + "\n" + str(probs)
        if self.show_dialogue:
            print(self.dialogue_user)
        skill_list, probs = self.mediator.LLM2RL(plans, probs)
        return skill_list, probs


class SimpleDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = SimpleDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>.": [["explore"], [1.0]],
                "Agent sees <door>, holds <nothing>.": [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>.": [["go to <key>, pick up <key>", "pick up <key>"], [0.98, 0.02]],
                "Agent sees <nothing>, holds <key>.": [["explore", "go to <door>, open <door>"], [0.68, 0.32]],
                "Agent sees <door>, holds <key>.": [["go to <door>, open <door> with <key>"], [1.0]],
                "Agent sees <key>, <door>, holds <nothing>.": [["go to <key>, pick up <key>, go to <door>, open <door>"], [1.0]]
            }


class ColoredDoorKey_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = ColoredDoorKey_Mediator(soft)
        if offline:
            self.plans_dict = {
                # (略) --- 同じように色付きのケースを定義 ---
            }

    def plan(self, text):
        pattern = r'\b(blue|green|grey|purple|red|yellow)\b'
        color_words = re.findall(pattern, text)
        words = list(dict.fromkeys(color_words))  # Preserve order and remove duplicates
        color_map = dict(zip(words, [f'color{i+1}' for i in range(len(words))]))

        for color, label in color_map.items():
            text = text.replace(color, label)

        plans, probs = super().plan(text)

        plans_str = str(plans)
        for color, label in color_map.items():
            plans_str = plans_str.replace(label, color)
        plans = eval(plans_str)

        return plans, probs


class TwoDoor_Planner(Base_Planner):
    def __init__(self, offline, soft, prefix):
        super().__init__(offline, soft, prefix)
        self.mediator = TwoDoor_Mediator(soft)
        if offline:
            self.plans_dict = {
                "Agent sees <nothing>, holds <nothing>.": [["explore"], [1.0]],
                "Agent sees <door1>, holds <nothing>.": [["explore"], [1.0]],
                "Agent sees <key>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <nothing>, holds <key>.": [["explore"], [1.0]],
                "Agent sees <door1>, holds <key>.": [["go to <door1>, open <door1>"], [1.0]],
                "Agent sees <key>, <door1>, holds <nothing>.": [["go to <key>, pick up <key>"], [1.0]],
                "Agent sees <door1>, <door2>, holds <key>.": [["go to <door1>, open <door1>", "go to <door2>, open <door2>"], [0.5, 0.5]]
            }


def Planner(task, offline=True, soft=False, prefix=''):
    task = task.lower()
    if task == "simpledoorkey" or task == "lavadoorkey":
        return SimpleDoorKey_Planner(offline, soft, prefix)
    elif task == "coloreddoorkey":
        return ColoredDoorKey_Planner(offline, soft, prefix)
    elif task == "twodoor":
        return TwoDoor_Planner(offline, soft, prefix)
    else:
        raise ValueError(f"Unsupported task type: {task}")
