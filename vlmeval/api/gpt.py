from ..smp import *
import os
from .base import BaseAPI
from openai import OpenAI
import numpy as np
from PIL import Image
from loguru import logger
from vlmeval.smp.vlm import encode_image_to_base64


APIBASES = {
    "OFFICIAL": "https://api.openai.com/v1/chat/completions",
    "OPENAI_API_BASE": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


def GPT_context_window(model):
    length_map = {
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-turbo-preview": 128000,
        "gpt-4-1106-preview": 128000,
        "gpt-4-0125-preview": 128000,
        "gpt-4-vision-preview": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-2024-04-09": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo-1106": 16385,
        "gpt-3.5-turbo-instruct": 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000


class OpenAIWrapper(BaseAPI):

    is_api: bool = True

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-0613",
        retry: int = 5,
        wait: int = 5,
        key: str | None = None,
        verbose: bool = False,
        system_prompt: str | None = None,
        temperature: float = 0,
        timeout: int = 60,
        api_base: str | None = None,
        max_tokens: int = 2048,
        img_size: int = 512,
        img_detail: str = "low",
        use_azure: bool = False,
        **kwargs,
    ):
        print(f"use {model} as judge")

        self.model = model
        self.cur_idx = 0
        self.fail_msg = "Failed to obtain answer via API. "
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_azure = use_azure

        if "step" in model:
            env_key = os.environ.get("STEPAI_API_KEY", "")
        elif "yi-vision" in model:
            env_key = os.environ.get("YI_API_KEY", "")
        elif "internvl2-pro" in model:
            env_key = os.environ.get("InternVL2_PRO_KEY", "")
        elif "abab" in model:
            env_key = os.environ.get("MiniMax_API_KEY", "")
        elif "moonshot" in model:
            env_key = os.environ.get("MOONSHOT_API_KEY", "")
        elif "grok" in model:
            env_key = os.environ.get("XAI_API_KEY", "")
        elif "qwen" in model:
            env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        elif "qwen" in model:
            env_key = os.environ.get("DASHSCOPE_API_KEY", "")
        else:
            if use_azure:
                env_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
                assert (
                    env_key is not None
                ), "Please set the environment variable AZURE_OPENAI_API_KEY."

                assert isinstance(
                    key, str
                ), "Please set the environment variable AZURE_OPENAI_API_KEY to your openai key."
            else:
                env_key = os.environ.get("OPENAI_API_KEY", "")

                assert isinstance(env_key, str) and env_key.startswith("sk-"), (
                    f"Illegal openai_key {env_key}. "
                    "Please set the environment variable OPENAI_API_KEY to your openai key."
                )

        if key is None:
            key = env_key

        self.key = key
        assert img_size > 0 or img_size == -1
        self.img_size = img_size
        assert img_detail in ["high", "low"]
        self.img_detail = img_detail
        self.timeout = timeout
        self.o1_model = "o1" in model or "o3" in model

        super().__init__(
            wait=wait,
            retry=retry,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

        # Configure API base URL
        if use_azure:
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            assert (
                self.azure_endpoint is not None
            ), "Please set the environment variable AZURE_OPENAI_ENDPOINT."

            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            assert (
                self.azure_deployment is not None
            ), "Please set the environment variable AZURE_OPENAI_DEPLOYMENT_NAME."

            self.azure_api_version = os.getenv("OPENAI_API_VERSION")
            assert (
                self.azure_api_version is not None
            ), "Please set the environment variable OPENAI_API_VERSION."

            api_base = None  # Not used with client

            self.client = OpenAI(
                api_key=self.key,
                base_url=f"{self.azure_endpoint}openai/deployments/{self.azure_deployment}",
                default_headers={"api-key": self.key},
            )
        else:
            if api_base is None:
                api_base = os.environ["API_BASE_URL"]

            self.client = OpenAI(api_key=self.key, base_url=api_base)

        logger.info(
            f"Using API Base: {api_base if not self.use_azure else self.azure_endpoint}"
        )


    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text":
                    content_list.append({"type": "text", "text": msg["value"]})
                elif msg["type"] == "image":
                    img = Image.open(msg["value"])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": self.img_detail,
                    }
                    content_list.append({"type": "image_url", "image_url": img_struct})
        else:
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join([x["value"] for x in inputs])
            content_list = [{"type": "text", "text": text}]
        return content_list

    def prepare_inputs(self, inputs):
        import numpy as np

        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append({"role": "system", "content": self.system_prompt})
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(["type" in x for x in inputs]) or np.all(
            ["role" in x for x in inputs]
        ), inputs

        if "role" in inputs[0]:
            assert inputs[-1]["role"] == "user", inputs[-1]
            for item in inputs:
                input_msgs.append(
                    {
                        "role": item["role"],
                        "content": self.prepare_itlist(item["content"]),
                    }
                )
        else:
            input_msgs.append({"role": "user", "content": self.prepare_itlist(inputs)})
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> tuple:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        # Constrain temperature to be within [0, 2)
        if temperature < 0:
            temperature = 0
        elif temperature >= 2:
            temperature = 1.99

        try:
            # Use the OpenAI client for API calls
            if self.o1_model:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=input_msgs,
                    max_completion_tokens=max_tokens,
                    n=1,
                    **kwargs,
                )
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=input_msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n=1,
                    **kwargs,
                )

            answer = completion.choices[0].message.content.strip()
            return 0, answer, completion
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return 1, e, None

    def get_image_token_len(self, img_path, detail="low"):
        import math
        from PIL import Image

        if detail == "low":
            return 85

        im = Image.open(img_path)
        height, width = im.size
        if width > 1024 or height > 1024:
            if width > height:
                height = int(height * 1024 / width)
                width = 1024
            else:
                width = int(width * 1024 / height)
                height = 1024

        h = math.ceil(height / 512)
        w = math.ceil(width / 512)
        total = 85 + 170 * h * w
        return total

    def get_token_len(self, inputs) -> int:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception as err:
            if "gpt" in self.model.lower():
                if self.verbose:
                    logger.warning(f"{type(err)}: {err}")
                enc = tiktoken.encoding_for_model("gpt-4")
            else:
                return 0
        assert isinstance(inputs, list)
        tot = 0
        for item in inputs:
            if "role" in item:
                tot += self.get_token_len(item["content"])
            elif item["type"] == "text":
                tot += len(enc.encode(item["value"]))
            elif item["type"] == "image":
                tot += self.get_image_token_len(item["value"], detail=self.img_detail)
        return tot


class GPT4V(OpenAIWrapper):

    def generate(self, message, dataset=None):
        return super(GPT4V, self).generate(message)
