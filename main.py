from transformers import LlamaForCausalLM, LlamaTokenizer

class LLMModel():
    # Load the llama model in the /llm_model
    def __init__(self):
        #self.model_path = "/backend/MessageHandler/llm_model/adapter_model.safetensors"
        #self.tokenizer_path = "/backend/MessageHandler/llm_model/tokenizer.json"
        self.model_path = "R73/LlamaCancer"

        # Load the model
        self.model = LlamaForCausalLM.from_pretrained(self.model_path, use_safetensors=True)

    def get_message(self, user_input):
        # Encode the user input
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")

        # Generate the response
        response = self.model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode the response
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)

        return response_text

# Create the model
llm_model = LLMModel()
llm_model.get_message("What is the meaning of life?")