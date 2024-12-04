import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk  # Load from local disk
import nltk
from nltk.tokenize import sent_tokenize
import os
import json
import csv
import time
import multiprocessing as mp
from setproctitle import setproctitle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to initialize the model and tokenizer for each process
def initialize_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

# Function to process each sentence using the pipeline
def detect_hate_in_sentence(sentence, model, tokenizer, device):
    # Predict hate speech for each sentence
    inputs = tokenizer(sentence,
                       return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=tokenizer.model_max_length,
                      ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class == 1 # 1 equals label 'hate'

# Function to process each chunk (i.e., year and text)
def process_chunk(chunk, model_name, device_count):
    year, text = chunk['year'], chunk['text']
    sentences = sent_tokenize(text)
    nr_sents = len(sentences)

     # Divide sentences among available GPUs
    sentences_per_gpu = [[] for _ in range(device_count)]
    for i, sentence in enumerate(sentences):
        sentences_per_gpu[i % device_count].append(sentence)

    # Queue to collect results from different GPU processes
    result_queue = mp.Queue()
    processes = []

    # Start a worker for each GPU
    for gpu_id in range(device_count):
        p = mp.Process(target=worker, args=(gpu_id, model_name, sentences_per_gpu[gpu_id], result_queue, year))
        p.start()
        processes.append(p)

    # Collect results from all GPUs
    hate_speech_sentences = []
    for _ in range(device_count):
        hate_speech_sentences.extend(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    return year, hate_speech_sentences, nr_sents


def worker(gpu_id, model_name, sentences, result_queue, year):
    # Set a custom process title for this worker
    setproctitle(f"fb_roberta_hate_eval_gpu_{gpu_id}_year_{year}")

    device = f'cuda:{gpu_id}'
    model, tokenizer = initialize_model_and_tokenizer(model_name, device)

    hate_speech_sentences = [s for s in sentences if detect_hate_in_sentence(s, model, tokenizer, device)]
    result_queue.put(hate_speech_sentences)

def update_csv(year, nr_sents, hate_sents, csv_file):
    # Check if the CSV file exists
    file_exists = os.path.exists(csv_file)

    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(["Year", "Total Sentences", "Hate Speech Sentences", "Hate Percentage"])
        # Write the row with the current year's data
        writer.writerow([year, nr_sents, hate_sents, (hate_sents/nr_sents)])

def main():
    start_time = time.time()
    # Access preloaded ressources from Docker Container
    nltk.data.path.append('/usr/share/nltk_data') # adjust to nltk location at your device
    dataset = load_from_disk("/app/pg_books_historic") # adjust to pg_books_historic location at your device
    # Set the start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    print(f"{torch.cuda.device_count()} GPUs are available", flush=True)

    # Prepare an output directory
    output_dir = "/app/output" # adjust to output location at your device
    os.makedirs(output_dir, exist_ok=True)

    # Prepare csv file for the summary
    csv_file = os.path.join(output_dir, "hate_speech_summary.csv")

    # Model & GPU infos
    model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
    device_count = torch.cuda.device_count()

    # Process each chunk sequentially
    chunks = len(dataset)
    for chunk_nr, chunk in enumerate(dataset):
        year = chunk['year']
        output_file = os.path.join(output_dir, f"results_{year}.jsonl")

        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"{chunk_nr+1}/{chunks}: Skipping year {year} as output file already exists.", flush=True)
            continue

        year, hate_speech_sentences, nr_sents = process_chunk(chunk, model_name, device_count)

        with open(output_file, "w") as f:
            for entry in hate_speech_sentences:
                f.write(json.dumps(entry) + "\n")

        # Update the CSV with total sentences and hate speech sentences for the year
        hate_sents_count = len(hate_speech_sentences)
        update_csv(year, nr_sents, hate_sents_count, csv_file)

        #Log
        current_time = time.time()
        current_runtime = current_time - start_time
        # Calculate minutes and seconds
        minutes = int(current_runtime // 60)
        seconds = round(current_runtime % 60)
        print(f"## {minutes}:{seconds:02d} - {chunk_nr+1}/{chunks} ## processed {year} chunk consisting of {nr_sents / 1000:.0f}K sentences", flush=True)


if __name__ == "__main__":
    main()

