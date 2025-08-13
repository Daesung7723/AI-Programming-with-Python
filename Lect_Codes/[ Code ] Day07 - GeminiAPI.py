import tkinter as tk
from tkinter import scrolledtext, messagebox
import google.generativeai as genai
import threading
import os

# --- 1. Gemini API Setup ---
# Enter the API key issued from Google AI Studio here directly or load it from environment variables.
# For a local environment, it is recommended to get it from environment variables like this.
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"
# API_KEY = os.environ.get("GEMINI_API_KEY")

API_KEY = "Your API" # For testing, enter the API key here directly.

if not API_KEY:
    print("Error: API key is not set. Please enter the key in the API_KEY variable or set the environment variable.")
    exit()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- 2. Window Design and Tone & Manner Setup (Dark Blue Theme) ---
root = tk.Tk()
root.title("Gemini AI Chat")
root.geometry("800x600")
root.configure(bg="#0A192F")  # Dark blue background color

# Font and color settings
font_style = ("Malgun Gothic", 10) # Changed font to Malgun Gothic for Korean/English support
bg_color = "#0A192F"
fg_color = "#C0C8D6"
accent_color = "#64FFDA"
input_bg = "#112240"

# --- 3. UI Component Layout ---
# Text area to display chat history
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg=bg_color, fg=fg_color, font=font_style, borderwidth=0, highlightthickness=0)
chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_history.configure(state='disabled')  # Set to read-only

# Prompt input frame
input_frame = tk.Frame(root, bg=bg_color)
input_frame.pack(padx=10, pady=(0, 10), fill=tk.X)

# Prompt input text box
prompt_entry = tk.Entry(input_frame, bg=input_bg, fg=fg_color, font=font_style, borderwidth=1, relief="solid", insertbackground=accent_color)
prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
prompt_entry.bind("<Return>", lambda event: get_response())

# Send button
send_button = tk.Button(input_frame, text="Send", command=lambda: get_response(), bg=accent_color, fg=bg_color, font=font_style, relief="flat")
send_button.pack(side=tk.RIGHT, padx=(10, 0))

# Copy button
copy_button = tk.Button(root, text="Copy Response", command=lambda: copy_last_response(), bg=accent_color, fg=bg_color, font=font_style, relief="flat")
copy_button.pack(side=tk.BOTTOM, pady=(0, 10))

# --- 4. Function Definitions ---
last_response_text = ""

def get_response():
    global last_response_text
    user_prompt = prompt_entry.get()
    if not user_prompt:
        return

    # Add user prompt to chat history
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"You: {user_prompt}\n", 'user_tag')
    chat_history.configure(state='disabled')
    prompt_entry.delete(0, tk.END)

    # Call Gemini API (use threading to prevent UI from freezing while waiting for response)
    thread = threading.Thread(target=call_api, args=(user_prompt,))
    thread.start()

def call_api(user_prompt):
    global last_response_text
    try:
        response = model.generate_content(user_prompt)
        bot_response = response.text
        last_response_text = bot_response

        # Add bot response to chat history
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Gemini AI: {bot_response}\n\n", 'bot_tag')
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)  # Scroll to the very bottom

    except Exception as e:
        error_message = f"An error occurred: {e}"
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Gemini AI: {error_message}\n\n", 'error_tag')
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)

def copy_last_response():
    if last_response_text:
        root.clipboard_clear()
        root.clipboard_append(last_response_text)
        messagebox.showinfo("Copy Complete", "The response has been copied to the clipboard.")
    else:
        messagebox.showwarning("Copy Failed", "There is no response to copy.")

# --- 5. Text Color and Alignment Settings Using Tags ---
chat_history.tag_config('user_tag', foreground=accent_color, font=("Malgun Gothic", 10, "bold"))
chat_history.tag_config('bot_tag', foreground=fg_color, font=("Malgun Gothic", 10))
chat_history.tag_config('error_tag', foreground='red', font=("Malgun Gothic", 10))

# Start the main loop
root.mainloop()