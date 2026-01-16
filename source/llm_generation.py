from llama_cpp import Llama

# build prompt according to Qwen model chat template
def build_prompt(query, docs):
    system_msg = "<|im_start|>system\nYou are a helpful assistant that answers queries only using the relevant documents provided.\n<|im_end|>\n"

    docs_section = "\n".join(
        f"[{i+1}]\n{docs[i]}" for i in range(len(docs))
    )

    user_msg = (
        "<|im_start|>user\n"
        f"Query:\n{query}\n\nRelevant documents:\n{docs_section}\n\n"
        "Using the documents above, answer the query.\n"
        "<|im_end|>\n"
    )

    assistant_prefix = "<|im_start|>assistant\n"

    return system_msg + user_msg + assistant_prefix

#llama model for text generation
def generate(prompt): 
  stop_sequences = [".", "\n"]
  model_name = "./models/qwen2-1_5b-instruct-q5_k_m.gguf" 

  llm = Llama( 
    model_path=model_name, 
    n_ctx=32768, 
    embedding=False, 
    n_threads=4, 
    n_gpu_layers=-1, 
    logits_all=False,
    verbose=False,
    stop=stop_sequences) 
  output = llm( prompt=prompt, max_tokens=200, echo=False ) 
  return output['choices'][0]['text'].strip()