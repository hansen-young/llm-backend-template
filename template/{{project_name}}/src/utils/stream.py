from .types import ChatResponseMessage, Choice, ChoiceChunk
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function,
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


class ChoiceAggregator:
    def __init__(self):
        self._choice: Choice
        self.reset()

    def reset(self):
        self._choice = Choice.model_construct(
            finish_reason="",
            index=0,
            message=ChatResponseMessage(role="assistant"),
        )

    def _update_finish_reason(self, finish_reason: str | None):
        if finish_reason:
            self._choice.finish_reason = finish_reason  # type: ignore

    def _update_role(self, role: str | None):
        if role:
            self._choice.message.role = role  # type: ignore

    def _update_content(self, content: str | None):
        if not content:
            return
        if self._choice.message.content is None:
            self._choice.message.content = ""
        self._choice.message.content += content

    def _update_tool_calls(self, tool_calls: list[ChoiceDeltaToolCall] | None):
        if not tool_calls:
            return
        if self._choice.message.tool_calls is None:
            self._choice.message.tool_calls = []

        for tc in tool_calls:
            i = tc.index

            if i >= len(self._choice.message.tool_calls):
                self._choice.message.tool_calls.append(
                    ChatCompletionMessageFunctionToolCall(
                        id="",
                        type="function",
                        function=Function(name="", arguments=""),
                    )
                )

            if tc.id is not None:
                self._choice.message.tool_calls[i].id += tc.id

            if tc.function is not None:
                if tc.function.name is not None:
                    self._choice.message.tool_calls[i].function.name += tc.function.name
                if tc.function.arguments is not None:
                    self._choice.message.tool_calls[
                        i
                    ].function.arguments += tc.function.arguments

    def update(self, chunk: ChoiceChunk):
        self._update_finish_reason(chunk.finish_reason)
        self._update_role(chunk.delta.role)
        self._update_content(chunk.delta.content)
        self._update_tool_calls(chunk.delta.tool_calls)

    @property
    def choice(self):
        return self._choice

    def __bool__(self):
        return bool(self._choice.message.content) or bool(
            self._choice.message.tool_calls
        )
