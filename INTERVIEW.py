from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import os

# Step 1: Setup Hugging Face token and define model name
os.environ["HF_TOKEN"] = "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"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def fine_tune_model(dataset_path):
    # Step 2: Load dataset
    dataset = load_dataset('json', data_files=dataset_path)

    # Step 3: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 4: Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['prompt'] + tokenizer.eos_token + examples['response'],
            truncation=True,
            padding='max_length',
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split dataset if validation set is missing
    if 'validation' not in tokenized_dataset:
        tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2)
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['test']
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation']

    # Step 5: Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Step 6: Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=2,
        report_to="none",  # Disable WandB or similar services if not needed
        fp16=True,  # Enable mixed-precision for faster training
    )

    # Step 7: Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Step 8: Train the model
    trainer.train()

    # Step 9: Save the fine-tuned model
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")

    print("Fine-tuning complete. Model saved to './fine_tuned_model'.")

# Step 10: Fine-tune the model
if __name__ == "__main__":
    dataset_path = "/content/questions_answers.json"  # Path to your JSON dataset
    fine_tune_model(dataset_path)
