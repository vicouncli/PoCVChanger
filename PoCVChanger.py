import numpy as np
import librosa
import sounddevice as sd
import scipy.signal
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
RATE = 44100
CHUNK = 2048

# Rutas seguras a recursos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(SCRIPT_DIR, "resource")
IMAGE_PATH = os.path.join(RESOURCE_DIR, "poc.png")
ICON_PATH = os.path.join(RESOURCE_DIR, "poc.ico")

# Parámetros de efectos (globales)
LOW_PASS_CUTOFF = 15000
HIGH_PASS_CUTOFF = 100
ECHO_DELAY = 0.03
NOISE_LEVEL = 0.0
BITCRUSH = 0
PITCH_SHIFT = -10
DISTORTION_LEVEL = 14
REVERB_DECAY = 0.0

stream = None
monitor_var = None

# ==============================
# FUNCIONES DE PROCESAMIENTO
# ==============================
def clean_audio(audio):
    return np.clip(np.nan_to_num(audio), -1.0, 1.0)

def bitcrush_audio(audio, bits):
    if bits <= 1:
        return audio
    scale = 2 ** bits
    return np.round(audio * scale) / scale

def process_audio(audio):
    global LOW_PASS_CUTOFF, HIGH_PASS_CUTOFF, ECHO_DELAY, NOISE_LEVEL, BITCRUSH
    global PITCH_SHIFT, DISTORTION_LEVEL, REVERB_DECAY

    audio = clean_audio(audio)

    # Pitch shift
    try:
        audio = librosa.effects.pitch_shift(audio, sr=RATE, n_steps=PITCH_SHIFT)
    except Exception as e:
        print(f"[WARN] Pitch shift falló: {e}")

    # Filtro pasa-altos
    if HIGH_PASS_CUTOFF > 20:
        sos_high = scipy.signal.butter(6, HIGH_PASS_CUTOFF, btype='high', fs=RATE, output='sos')
        audio = scipy.signal.sosfilt(sos_high, audio)
    # Filtro pasa-bajos
    if LOW_PASS_CUTOFF < (RATE // 2):
        sos_low = scipy.signal.butter(6, LOW_PASS_CUTOFF, btype='low', fs=RATE, output='sos')
        audio = scipy.signal.sosfilt(sos_low, audio)

    # Distorsión (saturación suave)
    audio = np.tanh(DISTORTION_LEVEL * audio)

    # Eco / Reverb simple
    echo = np.zeros_like(audio)
    delay = int(ECHO_DELAY * RATE)
    if delay > 0 and delay < len(audio):
        echo[delay:] = audio[:-delay] * REVERB_DECAY
    audio = audio + echo

    # Bitcrush (reducción de resolución)
    if BITCRUSH > 0:
        audio = bitcrush_audio(audio, BITCRUSH)

    # Ruido blanco
    if NOISE_LEVEL > 0:
        noise = np.random.normal(0, NOISE_LEVEL, audio.shape)
        audio = audio + noise

    return clean_audio(audio)

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    # Asegurar que el audio de entrada sea 1D
    input_audio = indata[:, 0] if indata.ndim > 1 else indata
    processed = process_audio(input_audio)
    # Salida
    if monitor_var.get():
        out_len = len(outdata)
        if len(processed) >= out_len:
            outdata[:, 0] = processed[:out_len]
        else:
            outdata[:len(processed), 0] = processed
            outdata[len(processed):, 0] = 0.0
    else:
        outdata[:, 0] = 0.0

# ==============================
# CONTROL DE STREAM DE AUDIO
# ==============================
def start_voice_changer():
    global stream
    if stream is not None:
        return
    input_name = input_device_var.get()
    output_name = output_device_var.get()
    input_index = get_device_index_by_name(input_name, kind='input')
    output_index = get_device_index_by_name(output_name, kind='output')

    if input_index is None or output_index is None:
        status_label.config(text="Device not found", fg="red")
        return

    try:
        stream = sd.Stream(
            samplerate=RATE,
            blocksize=CHUNK,
            dtype='float32',
            channels=1,
            callback=callback,
            device=(input_index, output_index)
        )
        stream.start()
        status_label.config(text="Voice Changer: ON", fg="red")
    except Exception as e:
        status_label.config(text=f"Error: {str(e)[:50]}", fg="red")
        print(f"[ERROR] Stream failed: {e}")

def stop_voice_changer():
    global stream
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
        status_label.config(text="Voice Changer: OFF", fg="white")

def get_device_index_by_name(name, kind='input'):
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[ERROR] No se pudieron obtener dispositivos: {e}")
        return None
    for i, dev in enumerate(devices):
        if dev['name'] == name:
            if kind == 'input' and dev['max_input_channels'] > 0:
                return i
            elif kind == 'output' and dev['max_output_channels'] > 0:
                return i
    return None

def get_input_output_devices():
    try:
        devices = sd.query_devices()
    except Exception:
        return [], []
    inputs = [dev['name'] for dev in devices if dev['max_input_channels'] > 0]
    outputs = [dev['name'] for dev in devices if dev['max_output_channels'] > 0]
    return inputs, outputs

# ==============================
# ACTUALIZADORES DE PARÁMETROS
# ==============================
def update_pitch(val):       global PITCH_SHIFT;       PITCH_SHIFT = float(val)
def update_distortion(val):  global DISTORTION_LEVEL;  DISTORTION_LEVEL = float(val)
def update_reverb(val):      global REVERB_DECAY;      REVERB_DECAY = float(val)
def update_lowpass(val):     global LOW_PASS_CUTOFF;   LOW_PASS_CUTOFF = float(val)
def update_highpass(val):    global HIGH_PASS_CUTOFF;  HIGH_PASS_CUTOFF = float(val)
def update_echo(val):        global ECHO_DELAY;        ECHO_DELAY = float(val)
def update_noise(val):       global NOISE_LEVEL;       NOISE_LEVEL = float(val)
def update_bitcrush(val):    global BITCRUSH;          BITCRUSH = int(float(val))

# ==============================
# INTERFAZ GRÁFICA
# ==============================
root = tk.Tk()
root.title("POC VOICE CHANGER")

# Cargar icono
if os.path.exists(ICON_PATH):
    try:
        root.iconbitmap(ICON_PATH)
    except Exception as e:
        print(f"[WARN] Icono no cargado: {e}")
else:
    print(f"[WARN] Icono no encontrado: {ICON_PATH}")

root.geometry("750x550")
root.resizable(False, False)
root.configure(bg="black")

monitor_var = tk.BooleanVar(value=False)

# Frame izquierdo: controles
settings_frame = tk.Frame(root, bg="black", width=250)
settings_frame.pack(side="left", fill="y")

slider_style = {"orient": "horizontal", "bg": "black", "fg": "white", "highlightthickness": 0}

controls = [
    ("Pitch", -12, 12, PITCH_SHIFT, update_pitch),
    ("Distortion", 1, 30, DISTORTION_LEVEL, update_distortion),
    ("Reverb", 0.0, 1.0, REVERB_DECAY, update_reverb, 0.01),
    ("Low-Pass", 500, RATE//2, LOW_PASS_CUTOFF, update_lowpass),
    ("High-Pass", 20, 5000, HIGH_PASS_CUTOFF, update_highpass),
    ("Echo Delay (s)", 0.0, 0.5, ECHO_DELAY, update_echo, 0.01),
    ("Noise Level", 0.0, 0.2, NOISE_LEVEL, update_noise, 0.01),
    ("Bitcrush (bits)", 0, 16, BITCRUSH, update_bitcrush, 1),
]

for label_text, from_, to_, init, cmd, *rest in controls:
    tk.Label(settings_frame, text=label_text, bg="black", fg="white").pack(pady=3)
    resolution = rest[0] if rest else 1
    slider = tk.Scale(
        settings_frame,
        from_=from_,
        to=to_,
        command=cmd,
        resolution=resolution,
        troughcolor="red",
        sliderrelief="flat",
        **slider_style
    )
    slider.set(init)
    slider.pack(pady=2)

# Frame derecho: logo, estado, dispositivos
main_frame = tk.Frame(root, bg="black")
main_frame.pack(side="left", fill="both", expand=True)

# Mostrar imagen del logo
image_tk = None
if os.path.exists(IMAGE_PATH):
    try:
        img = Image.open(IMAGE_PATH)
        img = img.resize((200, 200), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(main_frame, image=image_tk, bg="black")
        img_label.pack(pady=10)
    except Exception as e:
        print(f"[ERROR] Imagen no cargada: {e}")
        tk.Label(main_frame, text="WE ARE POC", font=("Arial", 16, "bold"), fg="white", bg="black").pack(pady=10)
else:
    print(f"[WARN] Imagen no encontrada: {IMAGE_PATH}")
    tk.Label(main_frame, text="WE ARE POC", font=("Arial", 16, "bold"), fg="white", bg="black").pack(pady=10)

# Estado
status_label = tk.Label(main_frame, text="Voice Changer: OFF", font=("Arial", 14), fg="white", bg="black")
status_label.pack(pady=10)

# Dispositivos de audio
input_devices, output_devices = get_input_output_devices()
input_device_var = tk.StringVar(value=input_devices[0] if input_devices else "")
output_device_var = tk.StringVar(value=output_devices[0] if output_devices else "")

tk.Label(main_frame, text="Input Device:", bg="black", fg="white").pack()
ttk.Combobox(main_frame, textvariable=input_device_var, values=input_devices).pack(pady=5)

tk.Label(main_frame, text="Output Device:", bg="black", fg="white").pack()
ttk.Combobox(main_frame, textvariable=output_device_var, values=output_devices).pack(pady=5)

# Monitor
tk.Checkbutton(main_frame, text="Monitor Audio", variable=monitor_var, bg="black", fg="white",
               selectcolor="black").pack(pady=5)

# Botones
tk.Button(main_frame, text="Start Voice Changer", command=start_voice_changer,
          fg="white", bg="red", relief="flat", height=2).pack(pady=10)
tk.Button(main_frame, text="Stop Voice Changer", command=stop_voice_changer,
          fg="white", bg="red", relief="flat").pack(pady=5)

# Ejecutar
root.mainloop()