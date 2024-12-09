# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import os
from io import BytesIO
from typing import List, Union

import requests
from fastapi import File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from ..proto.api_protocol import (
    AudioChatCompletionRequest,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    DocSumChatCompletionRequest,
    EmbeddingRequest,
    UsageInfo,
)
from ..proto.docarray import DocSumDoc, LLMParams, LLMParamsDoc, RerankedDoc, RerankerParms, RetrieverParms, TextDoc
from .constants import MegaServiceEndpoint, ServiceRoleType, ServiceType
from .micro_service import MicroService


def read_pdf(file):
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file)
    docs = loader.load_and_split()
    return docs


def read_text_from_file(file, save_file_name):
    import docx2txt
    from langchain.text_splitter import CharacterTextSplitter

    # read text file
    if file.headers["content-type"] == "text/plain":
        file.file.seek(0)
        content = file.file.read().decode("utf-8")
        # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(content)
        # Create multiple documents
        file_content = texts
    # read pdf file
    elif file.headers["content-type"] == "application/pdf":
        documents = read_pdf(save_file_name)
        file_content = [doc.page_content for doc in documents]
    # read docx file
    elif (
        file.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file.headers["content-type"] == "application/octet-stream"
    ):
        file_content = docx2txt.process(save_file_name)

    return file_content


class Gateway:
    def __init__(
        self,
        megaservice,
        host="0.0.0.0",
        port=8888,
        endpoint=str(MegaServiceEndpoint.CHAT_QNA),
        input_datatype=ChatCompletionRequest,
        output_datatype=ChatCompletionResponse,
    ):
        self.megaservice = megaservice
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.input_datatype = input_datatype
        self.output_datatype = output_datatype
        self.service = MicroService(
            self.__class__.__name__,
            service_role=ServiceRoleType.MEGASERVICE,
            service_type=ServiceType.GATEWAY,
            host=self.host,
            port=self.port,
            endpoint=self.endpoint,
            input_datatype=self.input_datatype,
            output_datatype=self.output_datatype,
        )
        self.define_routes()
        self.service.start()

    def define_routes(self):
        self.service.app.router.add_api_route(self.endpoint, self.handle_request, methods=["POST"])
        self.service.app.router.add_api_route(str(MegaServiceEndpoint.LIST_SERVICE), self.list_service, methods=["GET"])
        self.service.app.router.add_api_route(
            str(MegaServiceEndpoint.LIST_PARAMETERS), self.list_parameter, methods=["GET"]
        )

    def add_route(self, endpoint, handler, methods=["POST"]):
        self.service.app.router.add_api_route(endpoint, handler, methods=methods)

    def stop(self):
        self.service.stop()

    async def handle_request(self, request: Request):
        raise NotImplementedError("Subclasses must implement this method")

    def list_service(self):
        response = {}
        for node, service in self.megaservice.services.items():
            # Check if the service has a 'description' attribute and it is not None
            if hasattr(service, "description") and service.description:
                response[node] = {"description": service.description}
            # Check if the service has an 'endpoint' attribute and it is not None
            if hasattr(service, "endpoint") and service.endpoint:
                if node in response:
                    response[node]["endpoint"] = service.endpoint
                else:
                    response[node] = {"endpoint": service.endpoint}
            # If neither 'description' nor 'endpoint' is available, add an error message for the node
            if node not in response:
                response[node] = {"error": f"Service node {node} does not have 'description' or 'endpoint' attribute."}
        return response

    def list_parameter(self):
        pass

    def _handle_message(self, messages):
        images = []
        if isinstance(messages, str):
            prompt = messages
        else:
            messages_dict = {}
            system_prompt = ""
            prompt = ""
            for message in messages:
                msg_role = message["role"]
                if msg_role == "system":
                    system_prompt = message["content"]
                elif msg_role == "user":
                    if type(message["content"]) == list:
                        text = ""
                        text_list = [item["text"] for item in message["content"] if item["type"] == "text"]
                        text += "\n".join(text_list)
                        image_list = [
                            item["image_url"]["url"] for item in message["content"] if item["type"] == "image_url"
                        ]
                        if image_list:
                            messages_dict[msg_role] = (text, image_list)
                        else:
                            messages_dict[msg_role] = text
                    else:
                        messages_dict[msg_role] = message["content"]
                elif msg_role == "assistant":
                    messages_dict[msg_role] = message["content"]
                else:
                    raise ValueError(f"Unknown role: {msg_role}")

            if system_prompt:
                prompt = system_prompt + "\n"
            for role, message in messages_dict.items():
                if isinstance(message, tuple):
                    text, image_list = message
                    if text:
                        prompt += role + ": " + text + "\n"
                    else:
                        prompt += role + ":"
                    for img in image_list:
                        # URL
                        if img.startswith("http://") or img.startswith("https://"):
                            response = requests.get(img)
                            image = Image.open(BytesIO(response.content)).convert("RGBA")
                            image_bytes = BytesIO()
                            image.save(image_bytes, format="PNG")
                            img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                        # Local Path
                        elif os.path.exists(img):
                            image = Image.open(img).convert("RGBA")
                            image_bytes = BytesIO()
                            image.save(image_bytes, format="PNG")
                            img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                        # Bytes
                        else:
                            img_b64_str = img

                        images.append(img_b64_str)
                else:
                    if message:
                        prompt += role + ": " + message + "\n"
                    else:
                        prompt += role + ":"
        if images:
            return prompt, images
        else:
            return prompt


class ChatQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.CHAT_QNA), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        print("data in handle request", data)
        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)
        print("chat request in handle request", chat_request)
        prompt = self._handle_message(chat_request.messages)
        print("prompt in gateway", prompt)
        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
            chat_template=chat_request.chat_template if chat_request.chat_template else None,
            model=(
                chat_request.model
                if chat_request.model
                else os.getenv("MODEL_ID") if os.getenv("MODEL_ID") else "Intel/neural-chat-7b-v3-3"
            ),
        )
        retriever_parameters = RetrieverParms(
            search_type=chat_request.search_type if chat_request.search_type else "similarity",
            k=chat_request.k if chat_request.k else 4,
            distance_threshold=chat_request.distance_threshold if chat_request.distance_threshold else None,
            fetch_k=chat_request.fetch_k if chat_request.fetch_k else 20,
            lambda_mult=chat_request.lambda_mult if chat_request.lambda_mult else 0.5,
            score_threshold=chat_request.score_threshold if chat_request.score_threshold else 0.2,
        )
        reranker_parameters = RerankerParms(
            top_n=chat_request.top_n if chat_request.top_n else 1,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"text": prompt},
            llm_parameters=parameters,
            retriever_parameters=retriever_parameters,
            reranker_parameters=reranker_parameters,
        )
        for node, response in result_dict.items():
            if isinstance(response, StreamingResponse):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="chatqna", choices=choices, usage=usage)


class CodeGenGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.CODE_GEN), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)
        prompt = self._handle_message(chat_request.messages)
        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.2,
            streaming=stream_opt,
            model=chat_request.model if chat_request.model else None,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"query": prompt}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="codegen", choices=choices, usage=usage)


class CodeTransGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.CODE_TRANS), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        language_from = data["language_from"]
        language_to = data["language_to"]
        source_code = data["source_code"]
        prompt_template = """
            ### System: Please translate the following {language_from} codes into {language_to} codes. Don't output any other content except translated codes.

            ### Original {language_from} codes:
            '''

            {source_code}

            '''

            ### Translated {language_to} codes:

        """
        prompt = prompt_template.format(language_from=language_from, language_to=language_to, source_code=source_code)

        parameters = LLMParams(
            max_tokens=data.get("max_tokens", 1024),
            top_k=data.get("top_k", 10),
            top_p=data.get("top_p", 0.95),
            temperature=data.get("temperature", 0.01),
            repetition_penalty=data.get("repetition_penalty", 1.03),
            streaming=data.get("stream", True),
        )

        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"query": prompt}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="codetrans", choices=choices, usage=usage)


class TranslationGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.TRANSLATION), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        language_from = data["language_from"]
        language_to = data["language_to"]
        source_language = data["source_language"]
        prompt_template = """
            Translate this from {language_from} to {language_to}:

            {language_from}:
            {source_language}

            {language_to}:
        """
        prompt = prompt_template.format(
            language_from=language_from, language_to=language_to, source_language=source_language
        )
        result_dict, runtime_graph = await self.megaservice.schedule(initial_inputs={"query": prompt})
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="translation", choices=choices, usage=usage)


class DocSumGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice,
            host,
            port,
            str(MegaServiceEndpoint.DOC_SUMMARY),
            input_datatype=DocSumChatCompletionRequest,
            output_datatype=ChatCompletionResponse,
        )

    async def handle_request(self, request: Request, files: List[UploadFile] = File(default=None)):

        if "application/json" in request.headers.get("content-type"):
            data = await request.json()
            stream_opt = data.get("stream", True)
            chat_request = ChatCompletionRequest.model_validate(data)
            prompt = self._handle_message(chat_request.messages)

            initial_inputs_data = {data["type"]: prompt}

        elif "multipart/form-data" in request.headers.get("content-type"):
            data = await request.form()
            stream_opt = data.get("stream", True)
            chat_request = ChatCompletionRequest.model_validate(data)

            data_type = data.get("type")

            file_summaries = []
            if files:
                for file in files:
                    file_path = f"/tmp/{file.filename}"

                    if data_type is not None and data_type in ["audio", "video"]:
                        raise ValueError(
                            "Audio and Video file uploads are not supported in docsum with curl request, please use the UI."
                        )

                    else:
                        import aiofiles

                        async with aiofiles.open(file_path, "wb") as f:
                            await f.write(await file.read())

                        docs = read_text_from_file(file, file_path)
                        os.remove(file_path)

                        if isinstance(docs, list):
                            file_summaries.extend(docs)
                        else:
                            file_summaries.append(docs)

            if file_summaries:
                prompt = self._handle_message(chat_request.messages) + "\n".join(file_summaries)
            else:
                prompt = self._handle_message(chat_request.messages)

            data_type = data.get("type")
            if data_type is not None:
                initial_inputs_data = {}
                initial_inputs_data[data_type] = prompt
            else:
                initial_inputs_data = {"query": prompt}

        else:
            raise ValueError(f"Unknown request type: {request.headers.get('content-type')}")

        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
            model=chat_request.model if chat_request.model else None,
            language=chat_request.language if chat_request.language else "auto",
        )

        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs=initial_inputs_data, llm_parameters=parameters
        )

        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="docsum", choices=choices, usage=usage)


class AudioQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice,
            host,
            port,
            str(MegaServiceEndpoint.AUDIO_QNA),
            AudioChatCompletionRequest,
            ChatCompletionResponse,
        )

    async def handle_request(self, request: Request):
        data = await request.json()

        chat_request = AudioChatCompletionRequest.parse_obj(data)
        parameters = LLMParams(
            # relatively lower max_tokens for audio conversation
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 128,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=False,  # TODO add streaming LLM output as input to TTS
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"byte_str": chat_request.audio}, llm_parameters=parameters
        )

        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["byte_str"]

        return response


class SearchQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.SEARCH_QNA), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)
        prompt = self._handle_message(chat_request.messages)
        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"text": prompt}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="searchqna", choices=choices, usage=usage)


class FaqGenGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.FAQ_GEN), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request, files: List[UploadFile] = File(default=None)):
        data = await request.form()
        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)
        file_summaries = []
        if files:
            for file in files:
                file_path = f"/tmp/{file.filename}"

                import aiofiles

                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(await file.read())
                docs = read_text_from_file(file, file_path)
                os.remove(file_path)
                if isinstance(docs, list):
                    file_summaries.extend(docs)
                else:
                    file_summaries.append(docs)

        if file_summaries:
            prompt = self._handle_message(chat_request.messages) + "\n".join(file_summaries)
        else:
            prompt = self._handle_message(chat_request.messages)

        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
            model=chat_request.model if chat_request.model else None,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"query": prompt}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LLM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LLM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="faqgen", choices=choices, usage=usage)


class VisualQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.VISUAL_QNA), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        stream_opt = data.get("stream", False)
        chat_request = ChatCompletionRequest.parse_obj(data)
        prompt, images = self._handle_message(chat_request.messages)
        parameters = LLMParams(
            max_new_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"prompt": prompt, "image": images[0]}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LVM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LVM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="visualqna", choices=choices, usage=usage)


class VideoQnAGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice,
            host,
            port,
            str(MegaServiceEndpoint.VIDEO_RAG_QNA),
            ChatCompletionRequest,
            ChatCompletionResponse,
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        stream_opt = data.get("stream", False)
        chat_request = ChatCompletionRequest.parse_obj(data)
        prompt = self._handle_message(chat_request.messages)
        parameters = LLMParams(
            max_new_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
        )
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"text": prompt}, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # Here it suppose the last microservice in the megaservice is LVM.
            if (
                isinstance(response, StreamingResponse)
                and node == list(self.megaservice.services.keys())[-1]
                and self.megaservice.services[node].service_type == ServiceType.LVM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["text"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="videoqna", choices=choices, usage=usage)


class RetrievalToolGateway(Gateway):
    """embed+retrieve+rerank."""

    def __init__(self, megaservice, host="0.0.0.0", port=8889):
        super().__init__(
            megaservice,
            host,
            port,
            str(MegaServiceEndpoint.RETRIEVALTOOL),
            Union[TextDoc, ChatCompletionRequest],
            Union[RerankedDoc, LLMParamsDoc],
        )

    async def handle_request(self, request: Request):
        def parser_input(data, TypeClass, key):
            chat_request = None
            try:
                chat_request = TypeClass.parse_obj(data)
                query = getattr(chat_request, key)
            except:
                query = None
            return query, chat_request

        data = await request.json()
        query = None
        for key, TypeClass in zip(["text", "messages"], [TextDoc, ChatCompletionRequest]):
            query, chat_request = parser_input(data, TypeClass, key)
            if query is not None:
                break
        if query is None:
            raise ValueError(f"Unknown request type: {data}")
        if chat_request is None:
            raise ValueError(f"Unknown request type: {data}")

        if isinstance(chat_request, ChatCompletionRequest):
            retriever_parameters = RetrieverParms(
                search_type=chat_request.search_type if chat_request.search_type else "similarity",
                k=chat_request.k if chat_request.k else 4,
                distance_threshold=chat_request.distance_threshold if chat_request.distance_threshold else None,
                fetch_k=chat_request.fetch_k if chat_request.fetch_k else 20,
                lambda_mult=chat_request.lambda_mult if chat_request.lambda_mult else 0.5,
                score_threshold=chat_request.score_threshold if chat_request.score_threshold else 0.2,
            )
            reranker_parameters = RerankerParms(
                top_n=chat_request.top_n if chat_request.top_n else 1,
            )

            initial_inputs = {
                "messages": query,
                "input": query,  # has to be input due to embedding expects either input or text
                "search_type": chat_request.search_type if chat_request.search_type else "similarity",
                "k": chat_request.k if chat_request.k else 4,
                "distance_threshold": chat_request.distance_threshold if chat_request.distance_threshold else None,
                "fetch_k": chat_request.fetch_k if chat_request.fetch_k else 20,
                "lambda_mult": chat_request.lambda_mult if chat_request.lambda_mult else 0.5,
                "score_threshold": chat_request.score_threshold if chat_request.score_threshold else 0.2,
                "top_n": chat_request.top_n if chat_request.top_n else 1,
            }

            result_dict, runtime_graph = await self.megaservice.schedule(
                initial_inputs=initial_inputs,
                retriever_parameters=retriever_parameters,
                reranker_parameters=reranker_parameters,
            )
        else:
            result_dict, runtime_graph = await self.megaservice.schedule(initial_inputs={"text": query})

        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]
        return response


class MultimodalQnAGateway(Gateway):
    asr_port = int(os.getenv("ASR_SERVICE_PORT", 3001))
    asr_endpoint = os.getenv("ASR_SERVICE_ENDPOINT", "http://0.0.0.0:{}/v1/audio/transcriptions".format(asr_port))

    def __init__(self, multimodal_rag_megaservice, lvm_megaservice, host="0.0.0.0", port=9999):
        self.lvm_megaservice = lvm_megaservice
        self._role_labels = self._get_role_labels()
        super().__init__(
            multimodal_rag_megaservice,
            host,
            port,
            str(MegaServiceEndpoint.MULTIMODAL_QNA),
            ChatCompletionRequest,
            ChatCompletionResponse,
        )

    def _get_role_labels(self):
        """
        Returns a dictionary of role labels that are used in the chat prompt based on the LVM_MODEL_ID
        environment variable. The function defines the role labels used by the llava-1.5, llava-v1.6-vicuna,
        llava-v1.6-mistral, and llava-interleave models, and then defaults to use "USER:" and "ASSISTANT:" if the
        LVM_MODEL_ID is not one of those.
        """
        lvm_model = os.getenv("LVM_MODEL_ID", "")

        # Default to labels used by llava-1.5 and llava-v1.6-vicuna models
        role_labels = {
            "user": "USER:",
            "assistant": "ASSISTANT:"
        }

        if "llava-interleave" in lvm_model:
            role_labels["user"] = "<|im_start|>user"
            role_labels["assistant"] = "<|im_end|><|im_start|>assistant"
        elif "llava-v1.6-mistral" in lvm_model:
            role_labels["user"] = "[INST]"
            role_labels["assistant"] = " [/INST]"
        elif "llava-1.5" not in lvm_model and "llava-v1.6-vicuna" not in lvm_model:
            print(f"[ MultimodalQnAGateway ] Using default role labels for prompt formatting: {role_labels}")

        return role_labels

    # this overrides _handle_message method of Gateway
    def _handle_message(self, messages):
        images = []
        audios = []
        b64_types = {}
        messages_dicts = []
        decoded_audio_input = ""
        if isinstance(messages, str):
            prompt = messages
        else:
            messages_dict = {}
            system_prompt = ""
            prompt = ""
            role_label_dict = self._role_labels
            for message in messages:
                msg_role = message["role"]
                messages_dict = {}
                if msg_role == "system":
                    system_prompt = message["content"]
                elif msg_role == "user":
                    if type(message["content"]) == list:
                        # separate each media type and store accordingly
                        text = ""
                        text_list = [item["text"] for item in message["content"] if item["type"] == "text"]
                        text += "\n".join(text_list)
                        image_list = [
                            item["image_url"]["url"] for item in message["content"] if item["type"] == "image_url"
                        ]
                        audios = [item["audio"] for item in message["content"] if item["type"] == "audio"]
                        if audios:
                            # translate audio to text. From this point forward, audio is treated like text
                            decoded_audio_input = self.convert_audio_to_text(audios)
                            b64_types["audio"] = decoded_audio_input

                        if text and not audios and not image_list:
                            messages_dict[msg_role] = text
                        elif audios and not text and not image_list:
                            messages_dict[msg_role] = decoded_audio_input
                        else:
                            messages_dict[msg_role] = (text, decoded_audio_input, image_list)

                    else:
                        messages_dict[msg_role] = message["content"]
                    messages_dicts.append(messages_dict)
                elif msg_role == "assistant":
                    messages_dict[msg_role] = message["content"]
                    messages_dicts.append(messages_dict)
                else:
                    raise ValueError(f"Unknown role: {msg_role}")
            if system_prompt:
                prompt = system_prompt + "\n"
            for i, messages_dict in enumerate(messages_dicts):
                for role, message in messages_dict.items():
                    if isinstance(message, tuple):
                        text, decoded_audio_input, image_list = message
                        # Remove empty items from the image list
                        image_list = [x for x in image_list if x]
                        # Add image indicators within the conversation
                        image_tags = "<image>\n" * len(image_list)
                        if i == 0:
                            # do not add role for the very first message.
                            # this will be added by llava_server
                            if text:
                                prompt += image_tags + text + "\n"
                            elif decoded_audio_input:
                                prompt += image_tags + decoded_audio_input + "\n"
                        else:
                            if text:
                                prompt += role_label_dict[role] + " " + image_tags + text + "\n"
                            elif decoded_audio_input:
                                prompt += role_label_dict[role] + " " + image_tags + decoded_audio_input + "\n"
                            else:
                                prompt += role_label_dict[role] + " " + image_tags
                        for img in image_list:
                            # URL
                            if img.startswith("http://") or img.startswith("https://"):
                                response = requests.get(img)
                                image = Image.open(BytesIO(response.content)).convert("RGBA")
                                image_bytes = BytesIO()
                                image.save(image_bytes, format="PNG")
                                img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                            # Local Path
                            elif os.path.exists(img):
                                image = Image.open(img).convert("RGBA")
                                image_bytes = BytesIO()
                                image.save(image_bytes, format="PNG")
                                img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                            # Bytes
                            else:
                                img_b64_str = img

                        if image_list:
                            for img in image_list:
                                # URL
                                if img.startswith("http://") or img.startswith("https://"):
                                    response = requests.get(img)
                                    image = Image.open(BytesIO(response.content)).convert("RGBA")
                                    image_bytes = BytesIO()
                                    image.save(image_bytes, format="PNG")
                                    img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                                # Local Path
                                elif os.path.exists(img):
                                    image = Image.open(img).convert("RGBA")
                                    image_bytes = BytesIO()
                                    image.save(image_bytes, format="PNG")
                                    img_b64_str = base64.b64encode(image_bytes.getvalue()).decode()
                                # Bytes
                                else:
                                    img_b64_str = img

                                images.append(img_b64_str)

                    elif isinstance(message, str):
                        if i == 0:
                            # do not add role for the very first message.
                            # this will be added by llava_server
                            if message:
                                prompt += message + "\n"
                        else:
                            if message:
                                prompt += role_label_dict[role] + " " + message + "\n"
                            else:
                                prompt += role_label_dict[role]

        if images:
            b64_types["image"] = images

        # If the query has multiple media types, return all types
        if prompt and b64_types:
            return prompt, b64_types
        else:
            return prompt

    def convert_audio_to_text(self, audio):
        # translate audio to text by passing in dictionary to ASR
        if isinstance(audio, dict):
            input_dict = {"byte_str": audio["audio"][0]}
        else:
            input_dict = {"byte_str": audio[0]}

        response = requests.post(self.asr_endpoint, data=json.dumps(input_dict), proxies={"http": None})

        if response.status_code != 200:
            return JSONResponse(
                status_code=503, content={"message": "Unable to convert audio to text. {}".format(response.text)}
            )

        response = response.json()
        return response["query"]

    async def handle_request(self, request: Request):
        """
        MultimodalQnA accepts input queries as text, images, and/or audio. The messages in the request can be a single
        message (which would be assumed to be a first query from the user) or back and forth conversation between the
        user and the assistant.
        Audio queries are converted to text before being sent to the megaservice and the translated text is returned
        as part of the metadata in the response.
        First queries are sent to the full Multimodal megaserivce, which includes using the embedding microservice and
        retriever, in order to get relevant information from the vector store to send to the LVM along with the user's
        query. Follow up queries are sent directly to the LVM without searching for more similar information from the
        vector store.
        """
        data = await request.json()
        stream_opt = bool(data.get("stream", False))
        if stream_opt:
            print("[ MultimodalQnAGateway ] stream=True not used, this has not support streaming yet!")
            stream_opt = False
        chat_request = ChatCompletionRequest.model_validate(data)
        num_messages = len(data["messages"]) if isinstance(data["messages"], list) else 1
        messages = self._handle_message(chat_request.messages)
        decoded_audio_input = ""

        if num_messages > 1:
            # This is a follow up query, go to LVM
            cur_megaservice = self.lvm_megaservice
            if isinstance(messages, tuple):
                prompt, b64_types = messages
                if "audio" in b64_types:
                    # for metadata storage purposes
                    decoded_audio_input = b64_types["audio"]
                if "image" in b64_types:
                    initial_inputs = {"prompt": prompt, "image": b64_types["image"]}
                else:
                    initial_inputs = {"prompt": prompt, "image": ""}
            else:
                prompt = messages
                initial_inputs = {"prompt": prompt, "image": ""}
        else:
            # This is the first query. Ignore image input
            cur_megaservice = self.megaservice
            if isinstance(messages, tuple):
                prompt, b64_types = messages
                initial_inputs = {"text": prompt}
                if "audio" in b64_types:
                    # for metadata storage purposes
                    decoded_audio_input = b64_types["audio"]
                if "image" in b64_types and len(b64_types["image"]) > 0:
                    initial_inputs["image"] = {"base64_image": b64_types["image"][0]}
            else:
                initial_inputs = {"text": messages}

        parameters = LLMParams(
            max_new_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
            chat_template=chat_request.chat_template if chat_request.chat_template else None,
        )
        result_dict, runtime_graph = await cur_megaservice.schedule(
            initial_inputs=initial_inputs, llm_parameters=parameters
        )
        for node, response in result_dict.items():
            # the last microservice in this megaservice is LVM.
            # checking if LVM returns StreamingResponse
            # Currently, LVM with LLAVA has not yet supported streaming.
            # @TODO: Will need to test this once LVM with LLAVA supports streaming
            if (
                isinstance(response, StreamingResponse)
                and node == runtime_graph.all_leaves()[-1]
                and self.megaservice.services[node].service_type == ServiceType.LVM
            ):
                return response
        last_node = runtime_graph.all_leaves()[-1]

        if "text" in result_dict[last_node].keys():
            response = result_dict[last_node]["text"]
        else:
            # text is not in response message
            # something wrong, for example due to empty retrieval results
            if "detail" in result_dict[last_node].keys():
                response = result_dict[last_node]["detail"]
            else:
                response = "The server failed to generate an answer to your query!"
        if "metadata" in result_dict[last_node].keys():
            # from retrieval results
            metadata = result_dict[last_node]["metadata"]
            if decoded_audio_input:
                metadata["audio"] = decoded_audio_input
        else:
            # follow-up question, no retrieval
            if decoded_audio_input:
                metadata = {"audio": decoded_audio_input}
            else:
                metadata = None

        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response),
                finish_reason="stop",
                metadata=metadata,
            )
        )
        return ChatCompletionResponse(model="multimodalqna", choices=choices, usage=usage)


class AvatarChatbotGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice,
            host,
            port,
            str(MegaServiceEndpoint.AVATAR_CHATBOT),
            AudioChatCompletionRequest,
            ChatCompletionResponse,
        )

    async def handle_request(self, request: Request):
        data = await request.json()

        chat_request = AudioChatCompletionRequest.model_validate(data)
        parameters = LLMParams(
            # relatively lower max_tokens for audio conversation
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 128,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            repetition_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 1.03,
            streaming=False,  # TODO add streaming LLM output as input to TTS
        )
        # print(parameters)

        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs={"byte_str": chat_request.audio}, llm_parameters=parameters
        )

        last_node = runtime_graph.all_leaves()[-1]
        response = result_dict[last_node]["video_path"]
        return response


class GraphragGateway(Gateway):
    def __init__(self, megaservice, host="0.0.0.0", port=8888):
        super().__init__(
            megaservice, host, port, str(MegaServiceEndpoint.GRAPH_RAG), ChatCompletionRequest, ChatCompletionResponse
        )

    async def handle_request(self, request: Request):
        data = await request.json()
        stream_opt = data.get("stream", True)
        chat_request = ChatCompletionRequest.parse_obj(data)

        def parser_input(data, TypeClass, key):
            chat_request = None
            try:
                chat_request = TypeClass.parse_obj(data)
                query = getattr(chat_request, key)
            except:
                query = None
            return query, chat_request

        query = None
        for key, TypeClass in zip(["text", "input", "messages"], [TextDoc, EmbeddingRequest, ChatCompletionRequest]):
            query, chat_request = parser_input(data, TypeClass, key)
            if query is not None:
                break
        if query is None:
            raise ValueError(f"Unknown request type: {data}")
        if chat_request is None:
            raise ValueError(f"Unknown request type: {data}")
        prompt = self._handle_message(chat_request.messages)
        parameters = LLMParams(
            max_tokens=chat_request.max_tokens if chat_request.max_tokens else 1024,
            top_k=chat_request.top_k if chat_request.top_k else 10,
            top_p=chat_request.top_p if chat_request.top_p else 0.95,
            temperature=chat_request.temperature if chat_request.temperature else 0.01,
            frequency_penalty=chat_request.frequency_penalty if chat_request.frequency_penalty else 0.0,
            presence_penalty=chat_request.presence_penalty if chat_request.presence_penalty else 0.0,
            repetition_penalty=chat_request.repetition_penalty if chat_request.repetition_penalty else 1.03,
            streaming=stream_opt,
            chat_template=chat_request.chat_template if chat_request.chat_template else None,
        )
        retriever_parameters = RetrieverParms(
            search_type=chat_request.search_type if chat_request.search_type else "similarity",
            k=chat_request.k if chat_request.k else 4,
            distance_threshold=chat_request.distance_threshold if chat_request.distance_threshold else None,
            fetch_k=chat_request.fetch_k if chat_request.fetch_k else 20,
            lambda_mult=chat_request.lambda_mult if chat_request.lambda_mult else 0.5,
            score_threshold=chat_request.score_threshold if chat_request.score_threshold else 0.2,
        )
        initial_inputs = chat_request
        result_dict, runtime_graph = await self.megaservice.schedule(
            initial_inputs=initial_inputs,
            llm_parameters=parameters,
            retriever_parameters=retriever_parameters,
        )
        for node, response in result_dict.items():
            if isinstance(response, StreamingResponse):
                return response
        last_node = runtime_graph.all_leaves()[-1]
        response_content = result_dict[last_node]["choices"][0]["message"]["content"]
        choices = []
        usage = UsageInfo()
        choices.append(
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        )
        return ChatCompletionResponse(model="chatqna", choices=choices, usage=usage)
