"""
Week 10: Introduction to Large Language Models
Code Examples and Implementations

This module demonstrates practical usage of LLMs, focusing on:
1. Basic model interaction
2. Different inference patterns
3. Attention visualization
4. Token processing
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class LLMDemo:
    def __init__(self, model_name="gpt2-medium"):
        """
        Initialize with a pre-trained model.
        Using GPT2-medium as a lightweight example.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_text(self, prompt, max_length=50):
        """
        Basic text generation with the model.
        
        Args:
            prompt (str): Input text to continue
            max_length (int): Maximum length of generated sequence
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def few_shot_example(self, task_description, examples, query):
        """
        Demonstrate few-shot learning through prompt engineering.
        
        Args:
            task_description (str): Description of the task
            examples (list): List of example input-output pairs
            query (str): New input to process
        """
        # Construct prompt with task description and examples
        prompt = f"{task_description}\n\n"
        for input_text, output_text in examples:
            prompt += f"Input: {input_text}\nOutput: {output_text}\n\n"
        prompt += f"Input: {query}\nOutput:"

        return self.generate_text(prompt)

    def visualize_attention(self, text):
        """
        Visualize attention patterns for a given input.
        
        Args:
            text (str): Input text to analyze
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(
                inputs["input_ids"],
                output_attentions=True
            )
        
        # Get attention weights from last layer
        attention = outputs.attentions[-1].squeeze(0)
        
        # Create attention heatmap
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            attention[0].numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        plt.title("Attention Pattern")
        plt.show()

def main():
    # Initialize demo
    llm_demo = LLMDemo()
    
    # Example 1: Basic generation
    prompt = "The future of artificial intelligence will"
    print("\nBasic Generation Example:")
    print(f"Prompt: {prompt}")
    print(f"Generated: {llm_demo.generate_text(prompt)}")
    
    # Example 2: Few-shot learning
    task = "Classify the sentiment of movie reviews as positive or negative."
    examples = [
        ("This movie was fantastic!", "positive"),
        ("I really hated this film.", "negative"),
        ("Great acting and plot!", "positive")
    ]
    query = "The movie was boring and too long."
    print("\nFew-shot Learning Example:")
    print(llm_demo.few_shot_example(task, examples, query))
    
    # Example 3: Attention visualization
    print("\nVisualizing Attention:")
    llm_demo.visualize_attention("The quick brown fox jumps over the lazy dog.")

if __name__ == "__main__":
    main() 