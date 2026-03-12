###[ Two-Stage Process ]
# 1. Voice Enrollment: Convert a user's voice into a 'voice fingerprint' and store it.
# 2. Voice Identification: Compare a new voice with stored fingerprints to find the most similar person.

#  [ Step-1 Environment Setup ]
# pip install pyannote.audio sounddevice scipy numpy torch torchaudio

#  [ Step-2 Code Implementation (CLI Version) ]
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from pyannote.audio import Model, Inference
from pathlib import Path
import time

# --- Configuration ---
SAMPLE_RATE = 16000  # Sampling rate (Hz)
DURATION = 5        # Recording duration (seconds)
CHANNELS = 1        # Mono channel
HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # Your Hugging Face token

# Database file to store voice fingerprints (embeddings)
EMBEDDINGS_DB_PATH = Path("voice_db.npz")

# --- Load Model ---
print("Loading model, please wait...")
try:
    model = Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN)
    inference = Inference(model, window="whole")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if your Hugging Face token is correct and if you have agreed to the terms on the model page.")
    exit()

# --- Core Functions ---

def record_audio(file_path):
    """Records audio from the microphone for a specified duration and saves it to a file."""
    print(f"Starting recording for {DURATION} seconds... Please speak.")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    write(file_path, SAMPLE_RATE, recording)
    print(f"Recording complete! Saved to '{file_path}'.")

def get_embedding(file_path):
    """Extracts a voice fingerprint (embedding) from an audio file."""
    try:
        embedding = inference(file_path)
        # Flatten the embedding to a 1D vector
        return embedding.flatten()
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def enroll_user():
    """
    Enrolls a new user.
    Records 3 voice samples and saves the average of their embeddings for better accuracy.
    """
    user_name = input("Enter the name of the user to enroll: ")
    if not user_name:
        print("A name must be entered.")
        return

    embeddings_list = []
    for i in range(3):
        print(f"\n[{i+1}/3] Starting recording.")
        temp_audio_path = Path(f"temp_{user_name}_{i+1}.wav")
        record_audio(temp_audio_path)
        
        embedding = get_embedding(temp_audio_path)
        temp_audio_path.unlink() # Delete the temporary audio file immediately after use

        if embedding is None:
            print("Failed to extract embedding. Enrollment cancelled.")
            return
        
        embeddings_list.append(embedding)
        print(f"[{i+1}/3] Voice fingerprint extracted successfully.")
        if i < 2:
            time.sleep(1) # Brief pause before the next recording

    if len(embeddings_list) == 3:
        # Calculate the average of the 3 embeddings to create a final voice fingerprint
        average_embedding = np.mean(embeddings_list, axis=0)
        
        # Load existing database or create a new one
        if EMBEDDINGS_DB_PATH.exists():
            db = dict(np.load(EMBEDDINGS_DB_PATH, allow_pickle=True))
        else:
            db = {}
            
        db[user_name] = average_embedding
        np.savez(EMBEDDINGS_DB_PATH, **db)
        print(f"\nUser '{user_name}' has been successfully enrolled with the average of 3 recordings.")
    else:
        print("Enrollment failed as not all recordings could be processed.")

def identify_user():
    """Identifies a user from microphone input."""
    if not EMBEDDINGS_DB_PATH.exists() or not np.load(EMBEDDINGS_DB_PATH, allow_pickle=True).files:
        print("No users are enrolled. Please enroll a user first.")
        return

    db = dict(np.load(EMBEDDINGS_DB_PATH, allow_pickle=True))
    
    temp_audio_path = Path("temp_identification.wav")
    record_audio(temp_audio_path)
    
    current_embedding = get_embedding(temp_audio_path)
    temp_audio_path.unlink()

    if current_embedding is None:
        return

    max_similarity = -1
    identified_user = "Unknown"

    print("\n--- Identification Results ---")
    for name, saved_embedding in db.items():
        similarity = cosine_similarity(current_embedding, saved_embedding)
        print(f"Similarity with '{name}': {similarity:.4f}")
        if similarity > max_similarity:
            max_similarity = similarity
            identified_user = name

    threshold = 0.75
    if max_similarity >= threshold:
        print(f"\nFinal Result: You are '{identified_user}'! (Similarity: {max_similarity:.4f})")
    else:
        print(f"\nFinal Result: You are not a registered user. (Max Similarity: {max_similarity:.4f})")

def delete_user():
    """Deletes a registered user from the database."""
    if not EMBEDDINGS_DB_PATH.exists() or not np.load(EMBEDDINGS_DB_PATH, allow_pickle=True).files:
        print("The database is empty. There are no users to delete.")
        return

    db = dict(np.load(EMBEDDINGS_DB_PATH, allow_pickle=True))

    print("\n--- Currently Enrolled Users ---")
    for name in db.keys():
        print(f"- {name}")
    print("---------------------------------")

    user_name_to_delete = input("Enter the name of the user to delete: ")

    if user_name_to_delete in db:
        del db[user_name_to_delete]
        np.savez(EMBEDDINGS_DB_PATH, **db)
        print(f"User '{user_name_to_delete}' has been successfully deleted.")
    else:
        print(f"Error: User '{user_name_to_delete}' not found in the database.")

# --- Main Program Loop ---
def main():
    while True:
        print("\n--- Voice Recognition Program ---")
        print("1. Enroll User")
        print("2. Identify User")
        print("3. Delete User")
        print("4. Exit")
        choice = input("Enter the number for the desired action: ")

        if choice == '1':
            enroll_user()
        elif choice == '2':
            identify_user()
        elif choice == '3':
            delete_user()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid input. Please choose from 1, 2, 3, or 4.")
        
        time.sleep(1)

if __name__ == "__main__":
    main()