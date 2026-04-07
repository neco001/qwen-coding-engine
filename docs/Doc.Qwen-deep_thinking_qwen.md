Deep thinking models reason before they generate a response, improving accuracy on complex tasks such as logical reasoning and numerical calculation.

> This topic covers the OpenAI-compatible Chat Completion API and DashScope API. For the Responses API, see [Deep thinking](/help/en/model-studio/compatibility-with-openai-responses-api#example-deep-thinking-title).

![QwQ Logo](https://assets.alicdn.com/g/qwenweb/qwen-webui-fe/0.0.54/static/favicon.png)

Qwen

Show Thinking Process ▼

Send Virtual Request

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } } @keyframes blink { 0%, 100% { opacity: 0; } 50% { opacity: 1; } } .arrow-up { transform: rotate(180deg); } .arrow-down { transform: rotate(0deg); } .toggle-thinking:hover { background: #e6e8eb; } .send-button:hover { transform: scale(1.05); box-shadow: 0 2px 8px rgba(79, 118, 227, 0.3); }

## **Usage**

Alibaba Cloud Model Studio provides APIs for deep thinking models in two modes: hybrid thinking mode and thinking-only mode.

- **Hybrid thinking mode**: Use the `enable_thinking` parameter to enable or disable thinking mode:
  - When set to `true`, the model thinks before it responds.
  - When set to `false`, the model responds directly.

  ## OpenAI compatible

  ```
  # Import dependencies and create a client...
  completion = client.chat.completions.create(
      model="qwen-plus", # Select a model
      messages=[{"role": "user", "content": "Who are you"}],
      # Since enable_thinking is not a standard OpenAI parameter, pass it in extra_body.
      extra_body={"enable_thinking":True},
      # Enable streaming output.
      stream=True,
      # Configure the stream to include token consumption information in the last data packet.
      stream_options={
          "include_usage": True
      }
  )
  ```

  ## DashScope

  > The DashScope API for the Qwen3.5 series uses a multimodal interface. The following example returns a `url error`. For the correct usage, see [Enable or disable thinking mode](/help/en/model-studio/vision#bc67a9a2bd2of).

  ```
  # Import dependencies...

  response = Generation.call(
      # If you have not set the environment variable, replace the next line with your Model Studio API key, for example: api_key = "sk-xxx",
      api_key=os.getenv("DASHSCOPE_API_KEY"),
      # You can use other deep thinking models as needed.
      model="qwen-plus",
      messages=messages,
      result_format="message",
      enable_thinking=True,
      stream=True,
      incremental_output=True
  )
  ```

- **Thinking-only mode**: The model always thinks before it responds, and this behavior cannot be disabled. The request format is the same as hybrid thinking mode, but you do not need to set the enable_thinking parameter.

The API returns reasoning content in the `reasoning_content` field and the response in the `content` field. Deep thinking models reason before they respond, which increases latency. Because most of these models support only streaming output, all examples use streaming calls.

## **Supported models**

## Qwen3.6

**Qwen3.6 Plus series** (hybrid thinking mode, **enabled by default**): qwen3.6-plus, qwen3.6-plus-2026-04-02

## Qwen3.5

- **Commercial edition**
  - **Qwen3.5 Plus series** (hybrid thinking mode, **enabled by default**): qwen3.5-plus, qwen3.5-plus-2026-02-15
  - **Qwen3.5 Flash series** (hybrid thinking mode, **enabled by default**): qwen3.5-flash, qwen3.5-flash-2026-02-23

- **Open source edition**
  - Hybrid thinking mode (**enabled by default**): qwen3.5-397b-a17b, qwen3.5-122b-a10b, qwen3.5-27b, qwen3.5-35b-a3b

## Qwen3

- **Commercial edition**
  - **Qwen Max series** (hybrid thinking mode, disabled by default): qwen3-max-2026-01-23, qwen3-max-preview
  - **Qwen Plus series** (hybrid thinking mode, disabled by default): qwen-plus, qwen-plus-latest, qwen-plus-2025-04-28 and later snapshots
  - **Qwen Flash series** (hybrid thinking mode, disabled by default): qwen-flash, qwen-flash-2025-07-28 and later snapshots
  - **Qwen Turbo series** (hybrid thinking mode, disabled by default): qwen-turbo, qwen-turbo-latest, qwen-turbo-2025-04-28 and later snapshots

- **Open source edition**
  - Hybrid thinking mode (enabled by default): qwen3-235b-a22b, qwen3-32b, qwen3-30b-a3b, qwen3-14b, qwen3-8b, qwen3-4b, qwen3-1.7b, qwen3-0.6b
  - Thinking-only mode: qwen3-next-80b-a3b-thinking, qwen3-235b-a22b-thinking-2507, qwen3-30b-a3b-thinking-2507

## **QwQ (Qwen2.5)**

Thinking-only mode: qwq-plus, qwq-plus-latest, qwq-plus-2025-03-05, qwq-32b

## DeepSeek (Beijing)

- Hybrid thinking mode (disabled by default): deepseek-v3.2, deepseek-v3.2-exp, deepseek-v3.1
- Thinking-only mode: deepseek-r1, deepseek-r1-0528, the deepseek-r1 distilled model

## GLM (Beijing)

Hybrid thinking mode (enabled by default): glm-5, glm-4.7, glm-4.6

## Kimi (Beijing)

- Hybrid thinking mode (disabled by default): kimi-k2.5
- Thinking-only mode: kimi-k2-thinking

For model names, context windows, pricing, and snapshot versions, see [Model list](/help/en/model-studio/models). For rate limiting, see [Rate limiting](/help/en/model-studio/rate-limit).

## **Quick start**

Make sure you have [obtained an API key](/help/en/model-studio/get-api-key) and [configured the API key as an environment variable (to be deprecated and merged into API key configuration)](/help/en/model-studio/configure-api-key-through-environment-variables). If you use an SDK, [install the OpenAI or DashScope SDK](/help/en/model-studio/install-sdk#8833b9274f4v8) (the DashScope Java SDK version must be 2.19.4 or later).

Call the qwen-plus model in thinking mode with streaming output.

## OpenAI compatible

## Python

### **Sample code**

```
from openai import OpenAI
import os

# Initialize the OpenAI client.
client = OpenAI(
    # API keys vary by region. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
    # If an environment variable is not configured, provide your Model Studio API key directly: api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # Configurations vary by region. Modify the base_url based on your region.
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "Who are you"}]

completion = client.chat.completions.create(
    model="qwen-plus",  # You can replace this with other deep-thinking models as needed.
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={
        "include_usage": True
    },
)

reasoning_content = ""  # Full thinking process
answer_content = ""  # Full response
is_answering = False  # Tracks if the response phase has started
print("\n" + "=" * 20 + "Thinking process" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
        continue

    delta = chunk.choices[0].delta

    # Collect only the reasoning content.
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content

    # When content is received, start responding.
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Full response" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
```

### **Response**

```
====================Thinking process====================

The user's query "Who are you?" requires an accurate and friendly response. The answer should first establish my identity as Qwen, developed by Tongyi Lab at Alibaba Cloud. It will then outline key capabilities such as question answering, text generation, and logical reasoning. The language must be simple and the tone approachable. To encourage interaction, I will invite the user to ask more questions. Finally, I'll check that all key details are present, including my name (Qwen) and developer, to provide a comprehensive answer.
====================Full response====================

Hello! I am Qwen, a large language model developed by Tongyi Lab at Alibaba Group. I can answer questions, generate text, perform logical reasoning, write code, and more, to provide you with high-quality information and services. You can call me Qwen. How can I help you?
```

## Node.js

### **Sample code**

```
import OpenAI from "openai";
import process from 'process';

// Initialize the OpenAI client.
const openai = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY, // Read from an environment variable.
    // The following is the base URL for the Singapore region. If you use models in the US (Virginia) region, change the base URL to https://dashscope-us.aliyuncs.com/compatible-mode/v1. The base URL varies by region. Update it for your deployment's region.
    baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
});

let reasoningContent = '';
let answerContent = '';
let isAnswering = false;

async function main() {
    try {
        const messages = [{ role: 'user', content: 'Who are you' }];
        const stream = await openai.chat.completions.create({
            model: 'qwen-plus',
            messages,
            stream: true,
            enable_thinking: true
        });
        console.log('\n' + '='.repeat(20) + 'Thinking process' + '='.repeat(20) + '\n');

        for await (const chunk of stream) {
            if (!chunk.choices?.length) {
                console.log('\nUsage:');
                console.log(chunk.usage);
                continue;
            }

            const delta = chunk.choices[0].delta;

            // Collect only the reasoning content.
            if (delta.reasoning_content !== undefined && delta.reasoning_content !== null) {
                if (!isAnswering) {
                    process.stdout.write(delta.reasoning_content);
                }
                reasoningContent += delta.reasoning_content;
            }

            // When content is received, start responding.
            if (delta.content !== undefined && delta.content) {
                if (!isAnswering) {
                    console.log('\n' + '='.repeat(20) + 'Full response' + '='.repeat(20) + '\n');
                    isAnswering = true;
                }
                process.stdout.write(delta.content);
                answerContent += delta.content;
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

### **Response**

```
====================Thinking process====================

The user's direct query "Who are you?" requires a concise and clear response. The answer will state my identity as Qwen, a large language model from Alibaba Cloud. It will mention key functions like question answering, text generation, and logical reasoning, and highlight multilingual support (Chinese, English). To remain concise, use cases will be mentioned briefly, if at all. The tone will be friendly, and the response will end with an invitation for further questions. Finally, I'll check to ensure accuracy without including unnecessary details like version numbers.
====================Full response====================

I am Qwen, a large language model developed by Tongyi Lab at Alibaba Group. I can perform a variety of tasks, including answering questions, generating text, logical reasoning, and coding, and I support multiple languages, including Chinese and English. If you have any questions or need help, feel free to ask me at any time!
```

## HTTP

### **Sample code**

## curl

```
# ======= Important =======
# The following URL is for the Singapore region. For the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
# For the US (Virginia) region, replace it with: https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions
# === Remove this comment before execution ===
curl -X POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-plus",
    "messages": [
        {
            "role": "user",
            "content": "Who are you"
        }
    ],
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "enable_thinking": true
}'
```

### **Response**

```
data: {"choices":[{"delta":{"content":null,"role":"assistant","reasoning_content":""},"index":0,"logprobs":null,"finish_reason":null}],"object":"chat.completion.chunk","usage":null,"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

.....

data: {"choices":[{"finish_reason":"stop","delta":{"content":"","reasoning_content":null},"index":0,"logprobs":null}],"object":"chat.completion.chunk","usage":null,"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

data: {"choices":[],"object":"chat.completion.chunk","usage":{"prompt_tokens":10,"completion_tokens":360,"total_tokens":370},"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

data: [DONE]
```

## DashScope

> Because the DashScope API for the Qwen3.5 series uses a multimodal interface, the following example returns a `url error`. For the correct usage, see [Enable or disable thinking mode](/help/en/model-studio/vision#bc67a9a2bd2of).

## Python

### **Sample code**

```
import os
from dashscope import Generation
import dashscope

# Configurations vary by region. Modify this value based on your region.
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

# Initialize request parameters
messages = [{"role": "user", "content": "Who are you?"}]

completion = Generation.call(
    # If an environment variable is not configured, replace the following line with your Model Studio API key: api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",
    messages=messages,
    result_format="message",
    enable_thinking=True,
    stream=True,
    incremental_output=True,
)

# Full thinking process
reasoning_content = ""
# Full response
answer_content = ""
# Tracks if the response phase has started.
is_answering = False

print("=" * 20 + "Thinking process" + "=" * 20)

for chunk in completion:
    # If both the thinking and response content are empty, ignore the chunk.
    if (
        chunk.output.choices[0].message.content == ""
        and chunk.output.choices[0].message.reasoning_content == ""
    ):
        pass
    else:
        # If the current chunk is part of the thinking process.
        if (
            chunk.output.choices[0].message.reasoning_content != ""
            and chunk.output.choices[0].message.content == ""
        ):
            print(chunk.output.choices[0].message.reasoning_content, end="", flush=True)
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        # If the current chunk is part of the response.
        elif chunk.output.choices[0].message.content != "":
            if not is_answering:
                print("\n" + "=" * 20 + "Full response" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content, end="", flush=True)
            answer_content += chunk.output.choices[0].message.content

# To print the full thinking process and full response, uncomment the following code.
# print("=" * 20 + "Full thinking process" + "=" * 20 + "\n")
# print(f"{reasoning_content}")
# print("=" * 20 + "Full response" + "=" * 20 + "\n")
# print(f"{answer_content}")

```

### **Response**

```
====================Thinking process====================
To answer the query "Who are you?", the response must state my identity as Qwen, a large language model from Alibaba Cloud. It will then explain my purpose as a helpful assistant by outlining key functions like question answering, text generation, and logical reasoning. The response will maintain a conversational tone, avoiding jargon. To encourage further engagement, it will end with an open-ended question. Finally, I'll check for clarity, conciseness, and a balance between a friendly and professional tone.
====================Full response====================
Hello! I am Qwen, a large-scale language model developed by Alibaba Cloud. I can answer questions, generate text, perform logical reasoning, write code, and more, to provide help and support. Whether you have a question about daily life or a professional topic, I will do my best to help. Is there anything I can help you with?
```

## Java

### **Sample code**

```
// The version of the DashScope SDK must be 2.19.4 or later.
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.alibaba.dashscope.aigc.generation.Generation;
import com.alibaba.dashscope.aigc.generation.GenerationParam;
import com.alibaba.dashscope.aigc.generation.GenerationResult;
import com.alibaba.dashscope.common.Message;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.InputRequiredException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import io.reactivex.Flowable;
import java.lang.System;
import com.alibaba.dashscope.utils.Constants;

public class Main {
    static {
        // The following base_url is for the Singapore region. If you use a model in the Virginia region, you must change the base_url to https://dashscope-us.aliyuncs.com/api/v1.
        // Configurations vary by region. Modify the configuration based on your actual region.
        Constants.baseHttpApiUrl="https://dashscope-intl.aliyuncs.com/api/v1";
    }
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    private static StringBuilder reasoningContent = new StringBuilder();
    private static StringBuilder finalContent = new StringBuilder();
    private static boolean isFirstPrint = true;

    private static void handleGenerationResult(GenerationResult message) {
        String reasoning = message.getOutput().getChoices().get(0).getMessage().getReasoningContent();
        String content = message.getOutput().getChoices().get(0).getMessage().getContent();

        if (!reasoning.isEmpty()) {
            reasoningContent.append(reasoning);
            if (isFirstPrint) {
                System.out.println("====================Thinking Process====================");
                isFirstPrint = false;
            }
            System.out.print(reasoning);
        }

        if (!content.isEmpty()) {
            finalContent.append(content);
            if (!isFirstPrint) {
                System.out.println("\n====================Complete Response====================");
                isFirstPrint = true;
            }
            System.out.print(content);
        }
    }
    private static GenerationParam buildGenerationParam(Message userMsg) {
        return GenerationParam.builder()
                 // API keys vary by region. To obtain an API key, visit https://www.alibabacloud.com/help/en/model-studio/get-api-key
                // If you have not configured the environment variable, replace the following line with your Alibaba Cloud Model Studio API key: .apiKey("sk-xxx")
                .apiKey(System.getenv("DASHSCOPE_API_KEY"))
                .model("qwen-plus")
                .enableThinking(true)
                .incrementalOutput(true)
                .resultFormat("message")
                .messages(Arrays.asList(userMsg))
                .build();
    }
    public static void streamCallWithMessage(Generation gen, Message userMsg)
            throws NoApiKeyException, ApiException, InputRequiredException {
        GenerationParam param = buildGenerationParam(userMsg);
        Flowable<GenerationResult> result = gen.streamCall(param);
        result.blockingForEach(message -> handleGenerationResult(message));
    }

    public static void main(String[] args) {
        try {
            Generation gen = new Generation();
            Message userMsg = Message.builder().role(Role.USER.getValue()).content("Who are you?").build();
            streamCallWithMessage(gen, userMsg);
//             Print the final result.
//            if (reasoningContent.length() > 0) {
//                System.out.println("\n====================Complete Response====================");
//                System.out.println(finalContent.toString());
//            }
        } catch (ApiException | NoApiKeyException | InputRequiredException e) {
            logger.error("An exception occurred: {}", e.getMessage());
        }
        System.exit(0);
    }
}
```

### **Response**

```
====================Thinking process====================
The response to "Who are you?" must be based on my predefined identity as Qwen, a large language model from Alibaba Cloud. The answer will be conversational, concise, and easy to understand. It will first state my identity, then explain my functions, including text creation, logical reasoning, coding, and multilingual support. The tone will be friendly, and the response will end with an invitation for the user to ask for help, to encourage further interaction.
====================Full response====================
Hello! I am Qwen, a large language model from Alibaba Group. I can answer questions; create text such as stories, official documents, emails, and scripts; perform logical reasoning; write code; express opinions; and even play games. I am proficient in multiple languages, including but not limited to Chinese, English, German, French, and Spanish. Is there anything I can help you with?
```

## HTTP

### **Sample code**

## curl

```
# ======= Important =======
# API keys vary by region. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
# The following is the URL for the Singapore region. For the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation
# For the US (Virginia) region, replace the URL with: https://dashscope-us.aliyuncs.com/api/v1/services/aigc/text-generation/generation
# === Remove this comment before execution ===
curl -X POST "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-H "X-DashScope-SSE: enable" \
-d '{
    "model": "qwen-plus",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": "Who are you?"
            }
        ]
    },
    "parameters":{
        "enable_thinking": true,
        "incremental_output": true,
        "result_format": "message"
    }
}'
```

### **Response**

```
id:1
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"Hmm","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":14,"input_tokens":11,"output_tokens":3},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:2
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"，","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":15,"input_tokens":11,"output_tokens":4},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:3
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"user","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":16,"input_tokens":11,"output_tokens":5},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:4
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"asks","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":17,"input_tokens":11,"output_tokens":6},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:5
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"“","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":18,"input_tokens":11,"output_tokens":7},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}
......

id:358
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"help","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":373,"input_tokens":11,"output_tokens":362},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:359
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"，","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":374,"input_tokens":11,"output_tokens":363},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:360
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"Please feel free","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":375,"input_tokens":11,"output_tokens":364},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:361
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"to","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":376,"input_tokens":11,"output_tokens":365},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:362
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"let me know","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":377,"input_tokens":11,"output_tokens":366},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:363
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"！","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":378,"input_tokens":11,"output_tokens":367},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:364
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"","role":"assistant"},"finish_reason":"stop"}]},"usage":{"total_tokens":378,"input_tokens":11,"output_tokens":367},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}
```

## **Core capabilities**

### **Toggle thinking and non-thinking modes**

Thinking mode improves response quality but increases latency and cost. With hybrid thinking models, switch between thinking and non-thinking modes based on query complexity:

- For queries that do not require complex reasoning (such as casual conversation or simple Q&A), set `enable_thinking` to `false` to disable thinking mode.
- For queries that require complex reasoning (such as logical reasoning, code generation, or mathematical solutions), set `enable_thinking` to `true` to enable thinking mode.

## OpenAI compatible

**Important**

`enable_thinking` is not a standard OpenAI parameter. In the OpenAI Python SDK, pass it via `extra_body`. In the Node.js SDK, pass it as a top-level parameter.

## Python

### **Sample code**

```
from openai import OpenAI
import os

# Initialize the OpenAI client.
client = OpenAI(
    # If the environment variable is not configured, replace the value with your Model Studio API key: api_key="sk-xxx"
    # API keys differ by region. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # The base_url varies by region. Modify it based on your actual region.
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "Who are you?"}]
completion = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    # Set enable_thinking in extra_body to enable the reasoning process.
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={
        "include_usage": True
    },
)

reasoning_content = ""  # Full reasoning process
answer_content = ""  # Full response
is_answering = False  # Indicates if the response phase has started
print("\n" + "=" * 20 + "Reasoning process" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\n" + "=" * 20 + "Token usage" + "=" * 20 + "\n")
        print(chunk.usage)
        continue

    delta = chunk.choices[0].delta

    # Collect only the reasoning content.
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content

    # When content is received, start the response.
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Full response" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
```

### **Response**

```
====================Reasoning process====================

The user is asking "Who are you?". I need to determine what they want to know. They might be interacting with me for the first time or want to confirm my identity. I should introduce myself as Qwen, developed by Tongyi Lab. Then, I should explain my capabilities, such as answering questions, creating text, and coding, so the user understands how I can assist them. I should also mention my multilingual support so international users know they can communicate in different languages. Finally, I should maintain a friendly tone and invite them to ask questions to encourage further interaction. The explanation must be clear and simple, avoiding technical jargon. The user likely wants a quick overview of my abilities, so I will focus on my functions and applications. I should also consider if any information is missing, such as mentioning Alibaba Group or more technical details. However, the user probably only needs basic information. I will ensure the response is friendly and professional, and encourages them to continue the conversation.
====================Full response====================

I am Qwen, a large-scale language model developed by Tongyi Lab. I can help you answer questions, create text, code, and express opinions. I support communication in multiple languages. How can I help you?
====================Token usage====================

CompletionUsage(completion_tokens=221, prompt_tokens=10, total_tokens=231, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=172, rejected_prediction_tokens=None), prompt_tokens_details=PromptTokensDetails(audio_tokens=None, cached_tokens=0))
```

## Node.js

### **Sample code**

```
import OpenAI from "openai";
import process from 'process';

// Initialize the OpenAI client.
const openai = new OpenAI({
    // If the environment variable is not configured, replace the value with your Model Studio API key: apiKey: "sk-xxx"
    apiKey: process.env.DASHSCOPE_API_KEY,
    // Configurations vary by region. Modify this based on your actual region.
    baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
});

let reasoningContent = ''; // Full reasoning process
let answerContent = ''; // Full response
let isAnswering = false; // Indicates if the response phase has started

async function main() {
    try {
        const messages = [{ role: 'user', content: 'Who are you?' }];

        const stream = await openai.chat.completions.create({
            model: 'qwen-plus',
            messages,
            // In the Node.js SDK, non-standard parameters such as enable_thinking are passed as top-level properties and are not required in extra_body.
            enable_thinking: true,
            stream: true,
            stream_options: {
                include_usage: true
            },
        });

        console.log('\n' + '='.repeat(20) + 'Reasoning process' + '='.repeat(20) + '\n');

        for await (const chunk of stream) {
            if (!chunk.choices?.length) {
                console.log('\n' + '='.repeat(20) + 'Token usage' + '='.repeat(20) + '\n');
                console.log(chunk.usage);
                continue;
            }

            const delta = chunk.choices[0].delta;

            // Collect only the reasoning content.
            if (delta.reasoning_content !== undefined && delta.reasoning_content !== null) {
                if (!isAnswering) {
                    process.stdout.write(delta.reasoning_content);
                }
                reasoningContent += delta.reasoning_content;
            }

            // When content is received, start the response.
            if (delta.content !== undefined && delta.content) {
                if (!isAnswering) {
                    console.log('\n' + '='.repeat(20) + 'Full response' + '='.repeat(20) + '\n');
                    isAnswering = true;
                }
                process.stdout.write(delta.content);
                answerContent += delta.content;
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

### **Response**

```
====================Reasoning process====================

The user is asking "Who are you?". I need to determine what they want to know. They might be interacting with me for the first time or want to confirm my identity. I should introduce myself as Qwen, and mention my English name is also Qwen. Then, I will state that I am a large-scale language model independently developed by Tongyi Lab at Alibaba Group. Next, I should list my capabilities, such as answering questions, creating text, coding, and expressing opinions, so the user understands my purpose. I should also mention my multilingual support, which international users will find useful. Finally, I will invite them to ask questions with a friendly and open attitude. I must use simple, easy-to-understand language and avoid excessive technical jargon. The user may need help or just be curious, so the response should be welcoming and encourage further interaction. I should also consider if the user has deeper needs, such as testing my abilities or seeking specific help, but the initial response will focus on basic information and guidance. I will keep the tone conversational and use simple sentences for effective communication.
====================Full response====================

Hello! I am Qwen. I am a large-scale language model independently developed by Tongyi Lab at Alibaba Group. I can help you answer questions, create text such as stories, official documents, emails, and scripts, perform logical reasoning, code, and even express opinions and play games. I support multiple languages, including but not limited to Chinese, English, German, French, and Spanish.

If you have any questions or need help, feel free to ask!
====================Token usage====================

{
  prompt_tokens: 10,
  completion_tokens: 288,
  total_tokens: 298,
  completion_tokens_details: { reasoning_tokens: 188 },
  prompt_tokens_details: { cached_tokens: 0 }
}
```

## HTTP

### **Sample code**

## curl

```
# ======= IMPORTANT =======
# API keys differ by region. To get an API key, see https://www.alibabacloud.com/en/model-studio/get-api-key
# The base_url varies by region. For more information, see https://www.alibabacloud.com/help/en/model-studio/regions/
# === Delete this comment before execution ===
curl -X POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-plus",
    "messages": [
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "enable_thinking": true
}'
```

## DashScope

> The DashScope API for the Qwen3.5 series uses a multimodal interface. The following examples return a `url error`. For the correct usage, see [Enable or disable thinking mode](/help/en/model-studio/vision#bc67a9a2bd2of).

## Python

### **Sample code**

```
import os
from dashscope import Generation
import dashscope

# Configurations vary by region. Modify this as needed.
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"

# Initialize request parameters.
messages = [{"role": "user", "content": "Who are you?"}]

completion = Generation.call(
    # If you have not set an environment variable, replace this with your Model Studio API key: api_key="sk-xxx"
    # API keys are region-specific. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/regions
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",
    messages=messages,
    result_format="message",  # Set the result format to message.
    enable_thinking=True,     # Enable the thinking process.
    stream=True,              # Enable streaming output.
    incremental_output=True,  # Enable incremental output.
)

reasoning_content = ""  # Full thinking process
answer_content = ""     # Full response
is_answering = False    # Indicates if the model is in the answering phase.

print("\n" + "=" * 20 + "Thinking process" + "=" * 20 + "\n")

for chunk in completion:
    message = chunk.output.choices[0].message

    # Collect only the thinking content.
    if message.reasoning_content:
        if not is_answering:
            print(message.reasoning_content, end="", flush=True)
        reasoning_content += message.reasoning_content

    # When content is received, start building the response.
    if message.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Full response" + "=" * 20 + "\n")
            is_answering = True
        print(message.content, end="", flush=True)
        answer_content += message.content

print("\n" + "=" * 20 + "Token usage" + "=" * 20 + "\n")
print(chunk.usage)
# After the loop, the reasoning_content and answer_content variables contain the complete content.
# You can perform subsequent processing here as needed.
# print(f"\n\nFull thinking process:\n{reasoning_content}")
# print(f"\nFull response:\n{answer_content}")
```

### **Response**

```
====================Thinking process====================

Okay, the user is asking "Who are you?". I need to figure out what they want to know. They might be new to me or just want to confirm my identity. First, I should introduce myself as Qwen and state that I am a large-scale language model from Tongyi Lab. Then, I should explain my capabilities, such as answering questions, writing text, and coding, so the user knows what I can do. I should also mention my multilingual support for international users. Finally, I will be friendly and invite them to ask more questions to encourage interaction. It is important to use simple language and avoid technical jargon. The user might have other needs, like testing my abilities or getting help, so providing specific examples like writing stories, official documents, or emails would be helpful. I also need to make sure the response is well-structured. I can list my functions, but a natural flow might be better than bullets. I must also clarify that I am an AI assistant without personal consciousness and that my answers are based on training data to prevent misunderstandings. I should check if I missed any important details, like my multimodal capabilities or recent updates, but it is probably not necessary to go into too much detail for a first response. In short, the answer should be comprehensive but concise, friendly, and helpful, making the user feel understood and supported.
====================Full response====================

I am Qwen, a large-scale language model developed by Tongyi Lab of Alibaba Group. I can help you with the following:

1. **Answering questions**: I can try to answer your academic, general knowledge, or domain-specific questions.
2. **Creating text**: I can help you write stories, official documents, emails, scripts, and more.
3. **Logical reasoning**: I can help you with logical reasoning and problem-solving.
4. **Programming**: I can understand and generate code in various programming languages.
5. **Multilingual support**: I support multiple languages, including but not limited to Chinese, English, German, French, and Spanish.

If you have any questions or need help, feel free to let me know!
====================Token usage====================

{"input_tokens": 11, "output_tokens": 405, "total_tokens": 416, "output_tokens_details": {"reasoning_tokens": 256}, "prompt_tokens_details": {"cached_tokens": 0}}
```

## Java

### **Sample code**

```
// Requires DashScope SDK v2.19.4 or later.
import com.alibaba.dashscope.aigc.generation.Generation;
import com.alibaba.dashscope.aigc.generation.GenerationParam;
import com.alibaba.dashscope.aigc.generation.GenerationResult;
import com.alibaba.dashscope.common.Message;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.InputRequiredException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import com.alibaba.dashscope.utils.Constants;
import io.reactivex.Flowable;
import java.lang.System;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    private static StringBuilder reasoningContent = new StringBuilder();
    private static StringBuilder finalContent = new StringBuilder();
    private static boolean isFirstPrint = true;

    private static void handleGenerationResult(GenerationResult message) {
        String reasoning = message.getOutput().getChoices().get(0).getMessage().getReasoningContent();
        String content = message.getOutput().getChoices().get(0).getMessage().getContent();

        if (!reasoning.isEmpty()) {
            reasoningContent.append(reasoning);
            if (isFirstPrint) {
                System.out.println("====================Thinking process====================");
                isFirstPrint = false;
            }
            System.out.print(reasoning);
        }

        if (!content.isEmpty()) {
            finalContent.append(content);
            if (!isFirstPrint) {
                System.out.println("\n====================Full response====================");
                isFirstPrint = true;
            }
            System.out.print(content);
        }
    }
    private static GenerationParam buildGenerationParam(Message userMsg) {
        return GenerationParam.builder()
                // If you have not set an environment variable, replace the next line with your Model Studio API key: .apiKey("sk-xxx")
                .apiKey(System.getenv("DASHSCOPE_API_KEY"))
                .model("qwen-plus")
                .enableThinking(true)
                .incrementalOutput(true)
                .resultFormat("message")
                .messages(Arrays.asList(userMsg))
                .build();
    }
    public static void streamCallWithMessage(Generation gen, Message userMsg)
            throws NoApiKeyException, ApiException, InputRequiredException {
        GenerationParam param = buildGenerationParam(userMsg);
        Flowable<GenerationResult> result = gen.streamCall(param);
        result.blockingForEach(message -> handleGenerationResult(message));
    }

    public static void main(String[] args) {
        try {
            // The base_url varies by region.
            Generation gen = new Generation("http", "https://dashscope-intl.aliyuncs.com/api/v1");
            Message userMsg = Message.builder().role(Role.USER.getValue()).content("Who are you?").build();
            streamCallWithMessage(gen, userMsg);
//             Print the final result.
//            if (reasoningContent.length() > 0) {
//                System.out.println("\n====================Full response====================");
//                System.out.println(finalContent.toString());
//            }
        } catch (ApiException | NoApiKeyException | InputRequiredException e) {
            logger.error("An exception occurred: {}", e.getMessage());
        }
        System.exit(0);
    }
}
```

### **Response**

```
====================Thinking process====================
Okay, the user is asking "Who are you?". I need to figure out what they want to know. They might want to know my identity or are just testing my response. First, I should clearly state that I am Qwen, a large-scale language model from Alibaba Group. Then, I should briefly introduce my capabilities, such as answering questions, writing text, and coding, so the user understands what I can do. I should also mention that I support multiple languages so international users know they can communicate with me in different languages. Finally, I will be friendly and invite them to ask questions to make them feel comfortable and willing to interact further. The answer should not be too long but should be comprehensive. The user might have follow-up questions about my technical details or use cases, but the initial answer should be simple and clear. I will make sure not to use technical jargon so all users can understand. I will check for any missing important information, such as multilingual support and specific examples of my functions. Okay, this should cover the user's needs.
====================Full response====================
I am Qwen, a large-scale language model from Alibaba Group. I can answer questions, create text (such as stories, official documents, emails, and scripts), perform logical reasoning, code, express opinions, and play games. I also support multilingual communication, including but not limited to Chinese, English, German, French, and Spanish. If you have any questions or need help, feel free to let me know!
```

## HTTP

### **Sample code**

## curl

```
# ======= IMPORTANT =======
# API keys are region-specific. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
# The base_url varies by region. Modify it as needed.
# === Delete this comment before running ===
curl -X POST "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-H "X-DashScope-SSE: enable" \
-d '{
    "model": "qwen-plus",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": "Who are you?"
            }
        ]
    },
    "parameters":{
        "enable_thinking": true,
        "incremental_output": true,
        "result_format": "message"
    }
}'
```

### **Response**

```
id:1
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"Hmm","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":14,"input_tokens":11,"output_tokens":3},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:2
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":",","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":15,"input_tokens":11,"output_tokens":4},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:3
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"user","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":16,"input_tokens":11,"output_tokens":5},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:4
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":" asks","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":17,"input_tokens":11,"output_tokens":6},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:5
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":" \"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":18,"input_tokens":11,"output_tokens":7},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}
......

id:358
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"help","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":373,"input_tokens":11,"output_tokens":362},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:359
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":",","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":374,"input_tokens":11,"output_tokens":363},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:360
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":" feel free","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":375,"input_tokens":11,"output_tokens":364},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:361
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":" to","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":376,"input_tokens":11,"output_tokens":365},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:362
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":" let me know","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":377,"input_tokens":11,"output_tokens":366},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:363
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"!","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":378,"input_tokens":11,"output_tokens":367},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}

id:364
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"","role":"assistant"},"finish_reason":"stop"}]},"usage":{"total_tokens":378,"input_tokens":11,"output_tokens":367},"request_id":"25d58c29-c47b-9e8d-a0f1-d6c309ec58b1"}
```

Additionally, for **the Qwen3 open-source hybrid thinking models, along with the qwen-plus-2025-04-28 and qwen-turbo-2025-04-28 models**, you can control thinking mode with prompts. When `enable_thinking` is set to `true`, you can add `/no_think` to a prompt to disable thinking mode for that request. To re-enable it in a multi-turn conversation, add `/think` to the latest prompt. The model follows the most recent `/think` or `/no_think` command.

### **Limit thinking length**

Deep thinking models can produce lengthy thinking processes, increasing latency and token consumption. Use the `thinking_budget` parameter to limit the number of tokens used for thinking. If this limit is exceeded, the model immediately generates a response.

> `thinking_budget` defaults to the model's maximum chain-of-thought length. For details, see the [Model list](/help/en/model-studio/models).

**Important**

The `thinking_budget` parameter is supported by Qwen3 (in thinking mode) and Kimi models.

## OpenAI compatible

## Python

### **Sample code**

```
from openai import OpenAI
import os

# Initialize the OpenAI client.
client = OpenAI(
    # If the environment variable is not configured, replace "sk-xxx" with your Model Studio API key.
    # API keys are region-specific. To get an API key, visit https://www.alibabacloud.com/help/en/model-studio/get-api-key.
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # Configurations vary by region. Modify the base_url according to your region.
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

messages = [{"role": "user", "content": "Who are you"}]

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    # The enable_thinking parameter enables the thinking process, and thinking_budget sets its token limit.
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 50
        },
    stream=True,
    stream_options={
        "include_usage": True
    },
)

reasoning_content = ""  # Complete thinking process
answer_content = ""  # Complete response
is_answering = False  # Tracks if the response phase has started
print("\n" + "=" * 20 + "Thinking process" + "=" * 20 + "\n")

for chunk in completion:
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
        continue

    delta = chunk.choices[0].delta

    # Collect only the thinking content.
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content

    # When content is received, the response phase begins.
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Complete response" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
```

### **Response**

```
====================Thinking process====================

Okay, the user is asking, "Who are you?" I need to provide a clear and friendly response. First, I should state my identity as Qwen, developed by Tongyi Lab at Alibaba Group. Next, I need to explain my main functions, such as answering
====================Complete response====================

I am Qwen, a large-scale language model developed by Tongyi Lab at Alibaba Group. I can answer questions, create text, perform logical reasoning, and write code.
```

## Node.js

### **Sample code**

```
import OpenAI from "openai";
import process from 'process';

// Initialize the OpenAI client.
const openai = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY, // Read from an environment variable.
    // Configurations vary by region. Modify the baseURL according to your region.
    baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
});

let reasoningContent = '';
let answerContent = '';
let isAnswering = false;


async function main() {
    try {
        const messages = [{ role: 'user', content: 'Who are you' }];
        const stream = await openai.chat.completions.create({
            model: 'qwen-plus',
            messages,
            stream: true,
            // The enable_thinking parameter enables the thinking process, and thinking_budget sets its token limit.
            enable_thinking: true,
            thinking_budget: 50
        });
        console.log('\n' + '='.repeat(20) + 'Thinking process' + '='.repeat(20) + '\n');

        for await (const chunk of stream) {
            if (!chunk.choices?.length) {
                console.log('\nUsage:');
                console.log(chunk.usage);
                continue;
            }

            const delta = chunk.choices[0].delta;

            // Collect only the thinking content.
            if (delta.reasoning_content !== undefined && delta.reasoning_content !== null) {
                if (!isAnswering) {
                    process.stdout.write(delta.reasoning_content);
                }
                reasoningContent += delta.reasoning_content;
            }

            // When content is received, the response phase begins.
            if (delta.content !== undefined && delta.content) {
                if (!isAnswering) {
                    console.log('\n' + '='.repeat(20) + 'Complete response' + '='.repeat(20) + '\n');
                    isAnswering = true;
                }
                process.stdout.write(delta.content);
                answerContent += delta.content;
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

main();
```

### **Response**

```
====================Thinking process====================

Okay, the user is asking, "Who are you?" I need to provide a clear and accurate response. First, I should state my identity as Qwen, developed by Tongyi Lab at Alibaba Group. Next, I should explain my main functions, such as answering questions
====================Complete response====================

I am Qwen, a large-scale language model developed by Tongyi Lab at Alibaba Group. I can answer questions, create text, perform logical reasoning, and write code.
```

## HTTP

### **Sample code**

## curl

```
# ======= Important =======
# The following is the base URL for the Singapore region. For models in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
# For models in the US (Virginia) region, replace the URL with: https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions
# === Remove this comment before execution ===
curl -X POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen-plus",
    "messages": [
        {
            "role": "user",
            "content": "Who are you"
        }
    ],
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "enable_thinking": true,
    "thinking_budget": 50
}'
```

### **Response**

```
data: {"choices":[{"delta":{"content":null,"role":"assistant","reasoning_content":""},"index":0,"logprobs":null,"finish_reason":null}],"object":"chat.completion.chunk","usage":null,"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

.....

data: {"choices":[{"finish_reason":"stop","delta":{"content":"","reasoning_content":null},"index":0,"logprobs":null}],"object":"chat.completion.chunk","usage":null,"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

data: {"choices":[],"object":"chat.completion.chunk","usage":{"prompt_tokens":10,"completion_tokens":360,"total_tokens":370},"created":1745485391,"system_fingerprint":null,"model":"qwen-plus","id":"chatcmpl-e2edaf2c-8aaf-9e54-90e2-b21dd5045503"}

data: [DONE]
```

## DashScope

> The DashScope API for the Qwen3.5 series uses a multimodal interface. The following example returns a `url error`. For the correct usage, see [Enable or disable thinking mode](/help/en/model-studio/vision#bc67a9a2bd2of).

## Python

### **Sample code**

```
import os
from dashscope import Generation
import dashscope

# The base_url varies by region. Modify it according to your region.
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"

messages = [{"role": "user", "content": "Who are you?"}]

completion = Generation.call(
    # If the environment variable is not configured, replace the following line with your Model Studio API key: api_key = "sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus",
    messages=messages,
    result_format="message",
    enable_thinking=True,
    # Sets the token limit for the thinking process.
    thinking_budget=50,
    stream=True,
    incremental_output=True,
)

# Stores the complete thinking process.
reasoning_content = ""
# Stores the complete response.
answer_content = ""
# Tracks if the response phase has started.
is_answering = False

print("=" * 20 + "Thinking process" + "=" * 20)

for chunk in completion:
    # Ignore chunks where both thinking content and response content are empty.
    if (
        chunk.output.choices[0].message.content == ""
        and chunk.output.choices[0].message.reasoning_content == ""
    ):
        pass
    else:
        # If the current chunk contains thinking content.
        if (
            chunk.output.choices[0].message.reasoning_content != ""
            and chunk.output.choices[0].message.content == ""
        ):
            print(chunk.output.choices[0].message.reasoning_content, end="", flush=True)
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        # If the current chunk contains response content.
        elif chunk.output.choices[0].message.content != "":
            if not is_answering:
                print("\n" + "=" * 20 + "Complete response" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content, end="", flush=True)
            answer_content += chunk.output.choices[0].message.content

# To print the complete thinking process and response, uncomment and run the following code.
# print("=" * 20 + "Complete thinking process" + "=" * 20 + "\n")
# print(f"{reasoning_content}")
# print("=" * 20 + "Complete response" + "=" * 20 + "\n")
# print(f"{answer_content}")
```

### **Response**

```
====================Thinking process====================
Okay, the user is asking, "Who are you?" I need to provide a clear and friendly response. First, I must introduce myself as Qwen, developed by Tongyi Lab at Alibaba Group. Next, I should explain my main functions, such as
====================Complete response====================
I am Qwen, a large-scale language model developed by Tongyi Lab at Alibaba Group. I can answer questions, create text, perform logical reasoning, and write code.
```

## Java

### **Sample code**

```
// The DashScope SDK version must be 2.19.4 or later.
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.alibaba.dashscope.aigc.generation.Generation;
import com.alibaba.dashscope.aigc.generation.GenerationParam;
import com.alibaba.dashscope.aigc.generation.GenerationResult;
import com.alibaba.dashscope.common.Message;
import com.alibaba.dashscope.common.Role;
import com.alibaba.dashscope.exception.ApiException;
import com.alibaba.dashscope.exception.InputRequiredException;
import com.alibaba.dashscope.exception.NoApiKeyException;
import io.reactivex.Flowable;
import java.lang.System;
import com.alibaba.dashscope.utils.Constants;

public class Main {
    static {
        // The base HTTP API URL varies by region. Modify it according to your region.
        Constants.baseHttpApiUrl="https://dashscope-intl.aliyuncs.com/api/v1";
    }
    private static final Logger logger = LoggerFactory.getLogger(Main.class);
    private static StringBuilder reasoningContent = new StringBuilder();
    private static StringBuilder finalContent = new StringBuilder();
    private static boolean isFirstPrint = true;

    private static void handleGenerationResult(GenerationResult message) {
        String reasoning = message.getOutput().getChoices().get(0).getMessage().getReasoningContent();
        String content = message.getOutput().getChoices().get(0).getMessage().getContent();

        if (!reasoning.isEmpty()) {
            reasoningContent.append(reasoning);
            if (isFirstPrint) {
                System.out.println("====================Thinking process====================");
                isFirstPrint = false;
            }
            System.out.print(reasoning);
        }

        if (!content.isEmpty()) {
            finalContent.append(content);
            if (!isFirstPrint) {
                System.out.println("\n====================Complete response====================");
                isFirstPrint = true;
            }
            System.out.print(content);
        }
    }
    private static GenerationParam buildGenerationParam(Message userMsg) {
        return GenerationParam.builder()
                // If the environment variable is not configured, replace the following line with your Model Studio API key: .apiKey("sk-xxx")
                .apiKey(System.getenv("DASHSCOPE_API_KEY"))
                .model("qwen-plus")
                .enableThinking(true)
                .thinkingBudget(50)
                .incrementalOutput(true)
                .resultFormat("message")
                .messages(Arrays.asList(userMsg))
                .build();
    }
    public static void streamCallWithMessage(Generation gen, Message userMsg)
            throws NoApiKeyException, ApiException, InputRequiredException {
        GenerationParam param = buildGenerationParam(userMsg);
        Flowable<GenerationResult> result = gen.streamCall(param);
        result.blockingForEach(message -> handleGenerationResult(message));
    }

    public static void main(String[] args) {
        try {
            Generation gen = new Generation();
            Message userMsg = Message.builder().role(Role.USER.getValue()).content("Who are you?").build();
            streamCallWithMessage(gen, userMsg);
//             Print the final result.
//            if (reasoningContent.length() > 0) {
//                System.out.println("\n====================Complete response====================");
//                System.out.println(finalContent.toString());
//            }
        } catch (ApiException | NoApiKeyException | InputRequiredException e) {
            logger.error("An exception occurred: {}", e.getMessage());
        }
        System.exit(0);
    }
}
```

### **Response**

```
====================Thinking process====================
Okay, the user is asking, "Who are you?" I need to provide a clear and friendly response. First, I must introduce myself as Qwen, developed by Tongyi Lab at Alibaba Group. Next, I should explain my main functions, such as
====================Complete response====================
I am Qwen, a large-scale language model developed by Tongyi Lab at Alibaba Group. I can answer questions, create text, perform logical reasoning, and write code.
```

## HTTP

### **Sample code**

## curl

```
# ======= Important =======
# API keys are region-specific. To get an API key, visit https://www.alibabacloud.com/help/zh/model-studio/get-api-key.
# The endpoint URL varies by region. Modify it according to your region.
# === Remove this comment before execution ===
curl -X POST "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-H "X-DashScope-SSE: enable" \
-d '{
    "model": "qwen-plus",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": "Who are you?"
            }
        ]
    },
    "parameters":{
        "enable_thinking": true,
        "thinking_budget": 50,
        "incremental_output": true,
        "result_format": "message"
    }
}'
```

### **Response**

```
id:1
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"Okay","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":14,"output_tokens":3,"input_tokens":11,"output_tokens_details":{"reasoning_tokens":1}},"request_id":"2ce91085-3602-9c32-9c8b-fe3d583a2c38"}

id:2
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":",","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":15,"output_tokens":4,"input_tokens":11,"output_tokens_details":{"reasoning_tokens":2}},"request_id":"2ce91085-3602-9c32-9c8b-fe3d583a2c38"}

......

id:133
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"!","reasoning_content":"","role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":149,"output_tokens":138,"input_tokens":11,"output_tokens_details":{"reasoning_tokens":50}},"request_id":"2ce91085-3602-9c32-9c8b-fe3d583a2c38"}

id:134
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"","role":"assistant"},"finish_reason":"stop"}]},"usage":{"total_tokens":149,"output_tokens":138,"input_tokens":11,"output_tokens_details":{"reasoning_tokens":50}},"request_id":"2ce91085-3602-9c32-9c8b-fe3d583a2c38"}
```

### **Pass thinking process**

In a multi-turn conversation, the model does not reference the `reasoning_content` passed in the `messages` array. To make the model reference the previous thought process in subsequent replies, set `preserve_thinking` to `true`. When this parameter is enabled, the `reasoning_content` from the assistant's messages in the conversation history is appended to the model's input.

**Important**

The `preserve_thinking` parameter is only supported for qwen3.6-plus and qwen3.6-plus-2026-04-02.

> If the history messages do not contain `reasoning_content`, enabling this parameter does not cause an error.

> When this parameter is enabled, the `reasoning_content` from the conversation history is included in the input token count and billing.

## OpenAI compatible

**Note**

`preserve_thinking` is not a standard OpenAI parameter. When you use the Python software development kit (SDK), pass this parameter in `extra_body`.

## Python

### **Sample code**

```
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # Configurations vary by region. Modify this based on your actual region.
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# First turn of the conversation
messages = [
    {"role": "user", "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."}
]

first_reasoning = ""
first_content = ""
is_answering = False

completion = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=messages,
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={"include_usage": True},
)

print("=" * 20 + "First-turn thought process" + "=" * 20)

for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        first_reasoning += delta.reasoning_content
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "First-turn response" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
        first_content += delta.content

# Second turn: Pass the thought process and ask why the model excluded Kafka
messages = [
    {"role": "user", "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."},
    {
        "role": "assistant",
        "content": first_content,
        "reasoning_content": first_reasoning,
    },
    {"role": "user", "content": "Why did you exclude Kafka in your comparison?"},
]

reasoning_content = ""
answer_content = ""
is_answering = False

# Pass preserve_thinking through extra_body
completion = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=messages,
    extra_body={
        "enable_thinking": True,
        "preserve_thinking": True,
    },
    stream=True,
    stream_options={"include_usage": True},
)

print("\n" + "=" * 20 + "Second-turn thought process" + "=" * 20)

for chunk in completion:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "Second-turn response" + "=" * 20)
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
```

### **Response**

```
====================First-turn thought process====================
The user needs a message queue for an E-commerce system with tens of millions of daily messages. I will compare mainstream solutions based on dimensions such as throughput, reliability, delayed messages, and transaction support...

RocketMQ: Validated in Alibaba's E-commerce scenarios, natively supports transactional and delayed messages, and provides strict partition-level ordering...
Kafka: Extremely high throughput, but lacks native support for transactional and delayed messages, requiring custom compensation mechanisms...
RabbitMQ: Low latency, but limited cluster scalability, with a peak TPS in the tens of thousands...
====================First-turn response====================
Considering the core needs of an E-commerce scenario (transactional messages, delayed messages, ordering, and peak handling), I recommend Apache RocketMQ. If your team already has a Kafka ecosystem or requires strong real-time analytics, Kafka is also a viable option.
====================Second-turn thought process====================
The user is asking why Kafka was excluded. Reviewing my historical thought process, I did not exclude Kafka; I gave it a 4-star rating. I will refer to my previous detailed comparison to explain...

In the last turn, I compared the differences between RocketMQ and Kafka regarding transactional messages, delayed messages, and ordering. Kafka's main disadvantage is that it requires additional architectural design to compensate for E-commerce-specific semantics...
====================Second-turn response====================
I did not exclude Kafka. Kafka excels in throughput and ecosystem. The reason RocketMQ received a slightly higher rating is its better out-of-the-box match for core E-commerce workflows. RocketMQ natively supports transactional and delayed messages, while Kafka requires self-implementation through architectural patterns like the Outbox Pattern. If your team already has a Kafka ecosystem, it is fully capable of handling a scenario with tens of millions of messages.
```

## Node.js

### **Sample code**

```
import OpenAI from "openai";
import process from 'process';

const openai = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY,
    // Configurations vary by region. Modify this based on your actual region.
    baseURL: 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
});

async function main() {
    // First turn of the conversation
    let firstReasoning = '';
    let firstContent = '';
    let isAnswering = false;

    const stream1 = await openai.chat.completions.create({
        model: 'qwen3.6-plus',
        messages: [{ role: 'user', content: 'I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation.' }],
        stream: true,
        enable_thinking: true
    });

    console.log('='.repeat(20) + 'First-turn thought process' + '='.repeat(20));

    for await (const chunk of stream1) {
        if (!chunk.choices?.length) continue;
        const delta = chunk.choices[0].delta;
        if (delta.reasoning_content !== undefined && delta.reasoning_content !== null) {
            firstReasoning += delta.reasoning_content;
            if (!isAnswering) process.stdout.write(delta.reasoning_content);
        }
        if (delta.content !== undefined && delta.content) {
            if (!isAnswering) {
                console.log('\n' + '='.repeat(20) + 'First-turn response' + '='.repeat(20));
                isAnswering = true;
            }
            process.stdout.write(delta.content);
            firstContent += delta.content;
        }
    }

    // Second turn: Pass the thought process
    let reasoningContent = '';
    let answerContent = '';
    isAnswering = false;

    const stream2 = await openai.chat.completions.create({
        model: 'qwen3.6-plus',
        messages: [
            { role: 'user', content: 'I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation.' },
            {
                role: 'assistant',
                content: firstContent,
                reasoning_content: firstReasoning
            },
            { role: 'user', content: 'Why did you exclude Kafka in your comparison?' }
        ],
        stream: true,
        enable_thinking: true,
        // Pass preserve_thinking as a top-level parameter
        preserve_thinking: true
    });

    console.log('\n' + '='.repeat(20) + 'Second-turn thought process' + '='.repeat(20));

    for await (const chunk of stream2) {
        if (!chunk.choices?.length) continue;
        const delta = chunk.choices[0].delta;
        if (delta.reasoning_content !== undefined && delta.reasoning_content !== null) {
            if (!isAnswering) process.stdout.write(delta.reasoning_content);
            reasoningContent += delta.reasoning_content;
        }
        if (delta.content !== undefined && delta.content) {
            if (!isAnswering) {
                console.log('\n' + '='.repeat(20) + 'Second-turn response' + '='.repeat(20));
                isAnswering = true;
            }
            process.stdout.write(delta.content);
            answerContent += delta.content;
        }
    }
}

main();
```

### **Response**

```
====================First-turn thought process====================
The user needs a message queue for an E-commerce system with tens of millions of daily messages. I will compare mainstream solutions based on dimensions such as throughput, reliability, delayed messages, and transaction support...
====================First-turn response====================
Considering the core needs of an E-commerce scenario, I recommend Apache RocketMQ. If your team already has a Kafka ecosystem, Kafka is also a viable option.
====================Second-turn thought process====================
The user is asking why Kafka was excluded. Referring to my previous thought process, I did not exclude Kafka...
====================Second-turn response====================
I did not exclude Kafka. Kafka excels in throughput and ecosystem. The reason RocketMQ received a slightly higher rating is its better out-of-the-box match for core E-commerce workflows.
```

## HTTP

### **Sample code**

## curl

```
# The base_url varies by region. Modify it based on your actual region.
curl -X POST https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "qwen3.6-plus",
    "messages": [
        {
            "role": "user",
            "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."
        },
        {
            "role": "assistant",
            "content": "Considering the core needs of an E-commerce scenario, I recommend Apache RocketMQ.",
            "reasoning_content": "The user needs a message queue for an E-commerce system with tens of millions of daily messages. RocketMQ natively supports transactional and delayed messages, making it more suitable for E-commerce scenarios. Kafka has extremely high throughput but requires custom compensation mechanisms."
        },
        {
            "role": "user",
            "content": "Why did you exclude Kafka in your comparison?"
        }
    ],
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "enable_thinking": true,
    "preserve_thinking": true
}'
```

### **Response**

```
data: {"choices":[{"delta":{"content":null,"role":"assistant","reasoning_content":""},"index":0,"logprobs":null,"finish_reason":null}],"object":"chat.completion.chunk","usage":null,"created":1743523200,"system_fingerprint":null,"model":"qwen3.6-plus","id":"chatcmpl-example-001"}

.....

data: {"choices":[{"finish_reason":"stop","delta":{"content":"","reasoning_content":null},"index":0,"logprobs":null}],"object":"chat.completion.chunk","usage":null,"created":1743523200,"system_fingerprint":null,"model":"qwen3.6-plus","id":"chatcmpl-example-001"}

data: {"choices":[],"object":"chat.completion.chunk","usage":{"prompt_tokens":3463,"completion_tokens":2387,"total_tokens":5850},"created":1743523200,"system_fingerprint":null,"model":"qwen3.6-plus","id":"chatcmpl-example-001"}

data: [DONE]
```

## DashScope

**Note**

The Java SDK does not currently support the `preserve_thinking` parameter. When you make HTTP calls, place the `preserve_thinking` parameter in the `parameters` object.

## Python

### **Sample code**

```
import os
from dashscope import Generation
import dashscope

# The base_url varies by region. Modify it based on your actual region.
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1/"

# First turn of the conversation
messages = [
    {"role": "user", "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."}
]

first_reasoning = ""
first_content = ""
is_answering = False

completion = Generation.call(
    # If the environment variable is not set, replace the next line with your Model Studio API key: api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.6-plus",
    messages=messages,
    result_format="message",
    enable_thinking=True,
    stream=True,
    incremental_output=True,
)

print("=" * 20 + "First-turn thought process" + "=" * 20)

for chunk in completion:
    if (
        chunk.output.choices[0].message.content == ""
        and chunk.output.choices[0].message.reasoning_content == ""
    ):
        pass
    else:
        if (
            chunk.output.choices[0].message.reasoning_content != ""
            and chunk.output.choices[0].message.content == ""
        ):
            print(chunk.output.choices[0].message.reasoning_content, end="", flush=True)
            first_reasoning += chunk.output.choices[0].message.reasoning_content
        elif chunk.output.choices[0].message.content != "":
            if not is_answering:
                print("\n" + "=" * 20 + "First-turn response" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content, end="", flush=True)
            first_content += chunk.output.choices[0].message.content

# Second turn: Pass the thought process
messages = [
    {"role": "user", "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."},
    {
        "role": "assistant",
        "content": first_content,
        "reasoning_content": first_reasoning,
    },
    {"role": "user", "content": "Why did you exclude Kafka in your comparison?"},
]

reasoning_content = ""
answer_content = ""
is_answering = False

completion = Generation.call(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen3.6-plus",
    messages=messages,
    result_format="message",
    enable_thinking=True,
    # Pass the thought process
    preserve_thinking=True,
    stream=True,
    incremental_output=True,
)

print("\n" + "=" * 20 + "Second-turn thought process" + "=" * 20)

for chunk in completion:
    if (
        chunk.output.choices[0].message.content == ""
        and chunk.output.choices[0].message.reasoning_content == ""
    ):
        pass
    else:
        if (
            chunk.output.choices[0].message.reasoning_content != ""
            and chunk.output.choices[0].message.content == ""
        ):
            print(chunk.output.choices[0].message.reasoning_content, end="", flush=True)
            reasoning_content += chunk.output.choices[0].message.reasoning_content
        elif chunk.output.choices[0].message.content != "":
            if not is_answering:
                print("\n" + "=" * 20 + "Second-turn response" + "=" * 20)
                is_answering = True
            print(chunk.output.choices[0].message.content, end="", flush=True)
            answer_content += chunk.output.choices[0].message.content
```

### **Response**

```
====================First-turn thought process====================
The user needs a message queue for an E-commerce system with tens of millions of daily messages. I will compare mainstream solutions based on dimensions such as throughput, reliability, delayed messages, and transaction support...
====================First-turn response====================
Considering the core needs of an E-commerce scenario, I recommend Apache RocketMQ. If your team already has a Kafka ecosystem, Kafka is also a viable option.
====================Second-turn thought process====================
The user is asking why Kafka was excluded. Referring to my previous thought process, I did not exclude Kafka...
====================Second-turn response====================
I did not exclude Kafka. Kafka excels in throughput and ecosystem. The reason RocketMQ received a slightly higher rating is its better out-of-the-box match for core E-commerce workflows.
```

## HTTP

### **Sample code**

## curl

```
# The base_url varies by region. Modify it based on your actual region.
curl -X POST "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation" \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H "Content-Type: application/json" \
-H "X-DashScope-SSE: enable" \
-d '{
    "model": "qwen3.6-plus",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": "I need to choose a message queue for an E-commerce system that handles tens of millions of messages per day. Please provide a recommendation."
            },
            {
                "role": "assistant",
                "content": "Considering the core needs of an E-commerce scenario, I recommend Apache RocketMQ.",
                "reasoning_content": "The user needs a message queue for an E-commerce system with tens of millions of daily messages. RocketMQ natively supports transactional and delayed messages, making it more suitable for E-commerce scenarios. Kafka has extremely high throughput but requires custom compensation mechanisms."
            },
            {
                "role": "user",
                "content": "Why did you exclude Kafka in your comparison?"
            }
        ]
    },
    "parameters":{
        "enable_thinking": true,
        "preserve_thinking": true,
        "incremental_output": true,
        "result_format": "message"
    }
}'
```

### **Response**

```
id:1
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"user"},"role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":3466,"output_tokens":3,"input_tokens":3463,"output_tokens_details":{"reasoning_tokens":1}},"request_id":"example-request-001"}

id:2
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"follow-up"},"role":"assistant"},"finish_reason":"null"}]},"usage":{"total_tokens":3467,"output_tokens":4,"input_tokens":3463,"output_tokens_details":{"reasoning_tokens":2}},"request_id":"example-request-001"}

......

id:200
event:result
:HTTP_STATUS/200
data:{"output":{"choices":[{"message":{"content":"","reasoning_content":"","role":"assistant"},"finish_reason":"stop"}]},"usage":{"total_tokens":5850,"output_tokens":2387,"input_tokens":3463,"output_tokens_details":{"reasoning_tokens":1347}},"request_id":"example-request-001"}
```

### **Other features**

- [Multi-turn conversations](/help/en/model-studio/multi-round-conversation#9c315c95a8omt)
- [Tool calling](/help/en/model-studio/qwen-function-calling#fbf3d739f2q9f)
- [Web search](/help/en/model-studio/web-search#9b41940862qf3)

## **Billing**

- Thinking content is billed per output token.
- Some hybrid thinking models have different pricing for their thinking and non-thinking modes.

  > If a model in thinking mode fails to output a thinking process, it is billed at the price for non-thinking mode.

## **FAQ**

### **Q: How do I disable thinking mode?**

Whether you can disable thinking mode depends on the model type:

- For hybrid thinking models, such as qwen-plus and deepseek-v3.2-exp, set the enable_thinking parameter to false.
- For thinking-only models, such as qwen3-235b-a22b-thinking-2507 and deepseek-r1, the thinking mode cannot be disabled.

### **Q: Which models support non-streaming output?**

Deep thinking models increase response latency and the risk of timeouts for non-streaming output. Use streaming calls instead. If you require non-streaming output, use one of the following models.

## Qwen

- **Commercial edition**
  - **Qwen Max series**: qwen3-max-preview
  - **Qwen Plus series**: qwen3.6-plus, qwen3.6-plus-2026-04-02, qwen3.5-plus, qwen3.5-plus-2026-02-15, qwen-plus
  - **Qwen Flash series**: qwen3.5-flash, qwen3.5-flash-2026-02-23, qwen-flash, qwen-flash-2025-07-28
  - **Qwen Turbo series**: qwen-turbo

- **Open source edition**
  - qwen3.5-397b-a17b, qwen3.5-122b-a10b, qwen3.5-27b, qwen3.5-35b-a3b, qwen3-next-80b-a3b-thinking, qwen3-235b-a22b-thinking-2507, qwen3-30b-a3b-thinking-2507

## DeepSeek (Beijing region)

deepseek-v3.2, deepseek-v3.2-exp, deepseek-r1, deepseek-r1-0528, and the distilled deepseek-r1 model

## Kimi (Beijing region)

kimi-k2-thinking

### **Q:** [How do I purchase tokens after my free quota is used up?](/help/en/model-studio/new-free-quota)

Go to the [Expenses and Costs](https://usercenter2-intl.console.alibabacloud.com/billing/#/account/overview) center to top up your account. To call models, your account must not have overdue payments.

> After the free quota is used up, model calls are automatically billed per minute. To view spending details, go to **[Bill Details](https://usercenter2-intl.console.alibabacloud.com/finance/expense-report/expense-detail)**.

### **Q:** Can I upload images or documents as input**?**

These models support only text input. The Qwen3-VL and QVQ models support deep thinking on images.

### **Q: How do I view token** consumption **and the number of** calls**?**

**One hour after** you call a model, go to the Monitoring ([Singapore](https://modelstudio.console.alibabacloud.com/?tab=dashboard#/model-telemetry) or [Beijing](https://bailian.console.alibabacloud.com/?tab=model#/model-telemetry)) page. Set the query conditions, such as the time range and workspace. Then, in the **Models** area, find the target model and click **Monitor** in the **Actions** column to view the model's call statistics. For more information, see the [Monitoring](/help/en/model-studio/model-telemetry/) document.

> Data is updated hourly. During peak periods, there may be an hour-level latency.

![image](https://help-static-aliyun-doc.aliyuncs.com/assets/img/en-US/8821934571/p992753.png)

## API reference

For input and response parameters, see [Qwen](/help/en/model-studio/qwen-api-reference/).

## **Error codes**

If a call fails, see [Error messages](/help/en/model-studio/error-code).

/\* Reduce the top and bottom margin of blockquotes to prevent content from appearing too sparse. \*/ .unionContainer .markdown-body blockquote { margin: 4px 0; } .aliyun-docs-content table.qwen blockquote { border-left: none; /\* Remove the left border of blockquotes inside tables. \*/ padding-left: 5px; /\* Left padding \*/ margin: 4px 0; }
