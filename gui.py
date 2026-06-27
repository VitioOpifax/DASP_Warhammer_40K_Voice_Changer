import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import soundfile as sf
import sounddevice as sd
import json
import numpy as np
import scipy.signal as signal

# --- Matplotlib GUI Integration ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import engine  # Our DSP Engine

class ProfileEditor(tk.Toplevel):
    """A pop-up window to create and edit JSON profiles dynamically."""
    def __init__(self, parent_app, profile_file=None):
        super().__init__(parent_app.root)
        self.parent_app = parent_app
        self.title("Profile Editor" if profile_file else "New Profile")
        self.geometry("700x450")
        self.grab_set() # Lock interaction to this window until closed

        # Engine Data
        self.available_blocks = engine.get_available_blocks()
        self.pipeline = []
        self.current_filename = profile_file

        self.current_block_idx = None

        # Load existing data if editing
        initial_name = "New_Character"
        if profile_file:
            initial_name = Path(profile_file).stem
            with open(self.parent_app.profiles_dir / profile_file, 'r') as f:
                data = json.load(f)
                initial_name = data.get("profile_name", initial_name)
                self.pipeline = data.get("pipeline", [])

        self._build_editor_ui(initial_name)
        self.refresh_pipeline_list()

    def _build_editor_ui(self, initial_name):
        # Top: Profile Name
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="Profile Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(top_frame, width=30)
        self.name_entry.insert(0, initial_name)
        self.name_entry.pack(side=tk.LEFT, padx=10)

        # Middle: Lists and Parameters
        mid_frame = ttk.Frame(self, padding="10")
        mid_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Available Blocks
        left_frame = ttk.LabelFrame(mid_frame, text="Available DSP Blocks")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.avail_listbox = tk.Listbox(left_frame, height=10)
        for block in self.available_blocks.keys():
            self.avail_listbox.insert(tk.END, block)
        self.avail_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Button(left_frame, text="Add Block ➡", command=self.add_block).pack(pady=5)

        # Center: Current Pipeline
        center_frame = ttk.LabelFrame(mid_frame, text="Current Pipeline (Runs Top to Bottom)")
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.pipe_listbox = tk.Listbox(center_frame, height=10)
        self.pipe_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.pipe_listbox.bind('<<ListboxSelect>>', self.on_pipeline_select)

        btn_frame = ttk.Frame(center_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Move Up", command=lambda: self.move_block(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Move Down", command=lambda: self.move_block(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="❌ Remove", command=self.remove_block).pack(side=tk.LEFT, padx=2)

        # Right: Parameters Editor
        self.right_frame = ttk.LabelFrame(mid_frame, text="Block Parameters")
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.param_widgets = {} # Stores entry widgets to extract data later

        # Bottom: Save
        bottom_frame = ttk.Frame(self, padding="10")
        bottom_frame.pack(fill=tk.X)
        ttk.Button(bottom_frame, text="💾 Save Profile", command=self.save_profile).pack(side=tk.RIGHT)

    def refresh_pipeline_list(self):
        self.save_current_parameters()
        self.pipe_listbox.delete(0, tk.END)
        for i, block in enumerate(self.pipeline):
            self.pipe_listbox.insert(tk.END, f"{i+1}. {block['block_name']}")

        self.current_block_idx = None # Reset memory
        self.clear_parameters()

    def add_block(self):
        selection = self.avail_listbox.curselection()
        if not selection: return
        block_name = self.avail_listbox.get(selection[0])

        # Deep copy the default parameters so they don't link back to the engine
        new_params = {k: v for k, v in self.available_blocks[block_name].items()}

        self.pipeline.append({
            "block_name": block_name,
            "parameters": new_params
        })
        self.refresh_pipeline_list()

        # Select the newly added block automatically
        self.pipe_listbox.selection_set(tk.END)
        self.on_pipeline_select(None)

    def remove_block(self):
        selection = self.pipe_listbox.curselection()
        if not selection: return
        idx = selection[0]
        self.pipeline.pop(idx)
        self.current_block_idx = None # Clear memory before refreshing
        self.refresh_pipeline_list()

    def move_block(self, direction):
        selection = self.pipe_listbox.curselection()
        if not selection: return
        idx = selection[0]
        new_idx = idx + direction
        if 0 <= new_idx < len(self.pipeline):
            self.save_current_parameters() # Save data before moving!

            # Swap blocks
            self.pipeline[idx], self.pipeline[new_idx] = self.pipeline[new_idx], self.pipeline[idx]

            self.current_block_idx = None
            self.refresh_pipeline_list()

            # Re-select the moved block in its new position
            self.pipe_listbox.selection_set(new_idx)
            self.on_pipeline_select(None)

    def clear_parameters(self):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

    def on_pipeline_select(self, event):
        self.save_current_parameters() # Save whatever was there BEFORE drawing the new ones
        self.clear_parameters()

        selection = self.pipe_listbox.curselection()
        if not selection:
            self.current_block_idx = None
            return

        # Remember exactly which block we are drawing right now
        self.current_block_idx = selection[0]
        block_data = self.pipeline[self.current_block_idx]

        ttk.Label(self.right_frame, text=f"Editing: {block_data['block_name']}", font=("Arial", 9, "bold")).pack(pady=5)

        # Generate input fields for each parameter dynamically
        for param_name, param_val in block_data["parameters"].items():
            frame = ttk.Frame(self.right_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            ttk.Label(frame, text=param_name + ":", width=15).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=10)
            entry.insert(0, str(param_val))
            entry.pack(side=tk.LEFT)
            self.param_widgets[param_name] = entry

    def save_current_parameters(self):
        """Reads the entry boxes and saves them using the safe memory index."""
        if self.current_block_idx is None or not self.param_widgets:
            return

        for param_name, entry_widget in self.param_widgets.items():
            try:
                val = float(entry_widget.get())
                # Save into the specific block index we were tracking
                self.pipeline[self.current_block_idx]["parameters"][param_name] = val
            except ValueError:
                pass # Ignore invalid typing

    def save_profile(self):
        self.save_current_parameters()
        prof_name = self.name_entry.get().strip()
        if not prof_name:
            messagebox.showerror("Error", "Profile name cannot be empty.")
            return

        # Sanitize filename
        filename = prof_name.replace(" ", "_").lower() + ".json"
        filepath = self.parent_app.profiles_dir / filename

        data = {
            "profile_name": prof_name,
            "pipeline": self.pipeline
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

        self.parent_app.refresh_files()

        if filename in self.parent_app.profile_combo['values']:
            self.parent_app.profile_combo.set(filename)

        self.destroy() # Close the editor

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("40K Voice Changer - Blessed Machine Interface")

        # INCREASED WINDOW SIZE to fit the graphs
        self.root.geometry("1100x700")

        # --- 1. Define the Folders ---
        self.base_dir = Path(__file__).parent.resolve()
        self.input_dir = self.base_dir / "Input"
        self.profiles_dir = self.base_dir / "Profiles"
        self.output_dir = self.base_dir / "Output"

        self._ensure_directories()
        self._build_ui()
        self.refresh_files()

    def _ensure_directories(self):
        for directory in [self.input_dir, self.profiles_dir, self.output_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)

    def _build_ui(self):
        # --- SPLIT LAYOUT: Left for Controls, Right for Graphs ---
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT FRAME (Controls)
        control_frame = ttk.Frame(self.main_paned, width=450)
        self.main_paned.add(control_frame, weight=1)

        # RIGHT FRAME (Graphs)
        graph_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(graph_frame, weight=2)

        # ==========================================
        # LEFT SIDE: THE CONTROLS
        # ==========================================
        ttk.Label(control_frame, text="1. Select Input Audio (.wav):", font=("Arial", 10, "bold")).pack(anchor="w")
        input_frame = ttk.Frame(control_frame)
        input_frame.pack(fill=tk.X, pady=(0, 20))

        self.input_combo = ttk.Combobox(input_frame, state="readonly", width=35)
        self.input_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.input_combo.bind("<<ComboboxSelected>>", lambda e: self.update_output_list())

        ttk.Button(input_frame, text="▶ Play", command=self.play_input).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="2. Select Character Profile (.json):", font=("Arial", 10, "bold")).pack(anchor="w")
        profile_frame = ttk.Frame(control_frame)
        profile_frame.pack(fill=tk.X, pady=(0, 10))

        self.profile_combo = ttk.Combobox(profile_frame, state="readonly", width=30)
        self.profile_combo.pack(side=tk.LEFT, padx=(0, 5))

        # FIXED: Added command attributes to trigger the editor methods
        ttk.Button(profile_frame, text="➕ New", width=5, command=self.open_new_profile_editor).pack(side=tk.LEFT, padx=2)
        ttk.Button(profile_frame, text="✎ Edit", width=5, command=self.open_edit_profile_editor).pack(side=tk.LEFT, padx=2)

        self.process_btn = ttk.Button(control_frame, text="⚙ PROCESS AUDIO", command=self.process_audio)
        self.process_btn.pack(anchor="w", pady=(0, 20), fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=(0, 20))

        ttk.Label(control_frame, text="3. Generated Output Files:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.output_listbox = tk.Listbox(control_frame, height=8)
        self.output_listbox.pack(anchor="w", fill=tk.X, pady=(0, 10))

        # BINDING: Clicking an output file triggers the graphs!
        self.output_listbox.bind('<<ListboxSelect>>', self.on_output_select)

        bottom_frame = ttk.Frame(control_frame)
        bottom_frame.pack(fill=tk.X)
        ttk.Button(bottom_frame, text="▶ Play Selected", command=self.play_output).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(bottom_frame, text="⟳ Refresh", command=self.refresh_files).pack(side=tk.RIGHT)

        # ==========================================
        # RIGHT SIDE: THE AUSPEX (Matplotlib Graphs)
        # ==========================================
        # Create a Matplotlib figure with 2 subplots vertically
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.tight_layout(pad=4.0)

        # Setup initial empty graph states
        self.ax1.set_title("Time Domain Waveform", fontweight="bold")
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        self.ax2.set_title("Frequency Domain Spectrum", fontweight="bold")
        self.ax2.set_xlabel("Frequency [Hz]")
        self.ax2.set_ylabel("Magnitude [dB]")
        self.ax2.grid(True, linestyle='--', alpha=0.6)

        # Embed into Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    # --- Profile Editor Triggers ---
    def open_new_profile_editor(self):
        """Summons a blank Profile Editor."""
        ProfileEditor(self)

    def open_edit_profile_editor(self):
        """Summons the Profile Editor loaded with the currently selected JSON."""
        selected_profile = self.profile_combo.get()
        if selected_profile and not selected_profile.startswith("No .json"):
            ProfileEditor(self, selected_profile)

    # --- File Management Methods ---
    def refresh_files(self):
        input_files = [f.name for f in self.input_dir.glob('*.wav')]
        self.input_combo['values'] = input_files
        if input_files:
            if not self.input_combo.get() in input_files:
                self.input_combo.current(0)
        else:
            self.input_combo.set("No .wav files found...")

        profile_files = [f.name for f in self.profiles_dir.glob('*.json')]
        self.profile_combo['values'] = profile_files
        if profile_files:
            if not self.profile_combo.get() in profile_files:
                self.profile_combo.current(0)
        else:
            self.profile_combo.set("No .json files found...")

        self.update_output_list()

    def update_output_list(self):
        self.output_listbox.delete(0, tk.END)
        input_file = self.input_combo.get()
        if not input_file or input_file.startswith("No .wav"):
            return

        input_name = Path(input_file).stem
        specific_out_dir = self.output_dir / input_name

        if specific_out_dir.exists():
            for f in specific_out_dir.glob('*.wav'):
                self.output_listbox.insert(tk.END, f.name)

    # --- Audio Playback ---
    def _play_audio_file(self, filepath):
        try:
            data, fs = sf.read(filepath)
            sd.play(data, fs)
        except Exception as e:
            print(f"Machine Spirit Error during playback: {e}")

    def play_input(self):
        selected_file = self.input_combo.get()
        if selected_file and not selected_file.startswith("No .wav"):
            self._play_audio_file(self.input_dir / selected_file)

    def play_output(self):
        selection = self.output_listbox.curselection()
        if selection:
            input_name = Path(self.input_combo.get()).stem
            filepath = self.output_dir / input_name / self.output_listbox.get(selection[0])
            self._play_audio_file(filepath)

    # --- GRAPHING LOGIC (The Visualizer) ---
    def on_output_select(self, event):
        """Triggered when an output file is clicked. Updates the Matplotlib graphs."""
        selection = self.output_listbox.curselection()
        if not selection: return

        selected_output = self.output_listbox.get(selection[0])
        input_file = self.input_combo.get()

        if not input_file or input_file.startswith("No"): return

        in_filepath = self.input_dir / input_file
        out_filepath = self.output_dir / Path(input_file).stem / selected_output

        # Generate the visual comparison!
        self.draw_comparison_graphs(in_filepath, out_filepath)

    def draw_comparison_graphs(self, in_filepath, out_filepath):
        """Reads the audio arrays and draws the time and frequency domain plots."""
        try:
            # 1. Read Audio (Strictly using soundfile as per lectures)
            x_in, fs_in = sf.read(in_filepath)
            x_out, fs_out = sf.read(out_filepath)

            # Force mono for plotting
            if x_in.ndim > 1: x_in = x_in[:, 0]
            if x_out.ndim > 1: x_out = x_out[:, 0]

            # Clear old graphs
            self.ax1.clear()
            self.ax2.clear()

            # ---------------------------------------------
            # PLOT 1: Time Domain (Waveform)
            # ---------------------------------------------
            t_in = np.arange(len(x_in)) / fs_in
            t_out = np.arange(len(x_out)) / fs_out

            self.ax1.plot(t_in, x_in, label="Raw Input", color='blue', alpha=0.5)
            self.ax1.plot(t_out, x_out, label="Processed Output", color='orange', alpha=0.8)

            self.ax1.set_title("Time Domain Waveform", fontweight="bold")
            self.ax1.set_xlabel("Time [s]")
            self.ax1.set_ylabel("Amplitude")
            self.ax1.legend(loc="upper right")
            self.ax1.grid(True, linestyle='--', alpha=0.6)

            # ---------------------------------------------
            # PLOT 2: Frequency Domain
            # ---------------------------------------------
            freqs_in, psd_in = signal.welch(x_in, fs_in, nperseg=2048)
            freqs_out, psd_out = signal.welch(x_out, fs_out, nperseg=2048)

            # Convert to Decibels
            psd_in_db = 10 * np.log10(psd_in + 1e-10)
            psd_out_db = 10 * np.log10(psd_out + 1e-10)

            self.ax2.plot(freqs_in, psd_in_db, label="Raw Input", color='blue', alpha=0.5)
            self.ax2.plot(freqs_out, psd_out_db, label="Processed Output", color='orange', alpha=0.8)

            self.ax2.set_title("Frequency Domain Spectrum", fontweight="bold")
            self.ax2.set_xlabel("Frequency [Hz]")
            self.ax2.set_ylabel("Magnitude [dB]")
            self.ax2.set_xlim(0, 4000) # Focus on vocal range (0 to 4kHz)
            self.ax2.legend(loc="upper right")
            self.ax2.grid(True, linestyle='--', alpha=0.6)

            # Refresh the canvas!
            self.canvas.draw()

        except Exception as e:
            print(f"Auspex Failure: Could not draw graphs. {e}")

    # --- PROCESSING ---
    def process_audio(self):
        input_file = self.input_combo.get()
        profile_file = self.profile_combo.get()

        if not input_file or input_file.startswith("No .wav"): return
        if not profile_file or profile_file.startswith("No .json"): return

        input_name = Path(input_file).stem
        profile_name = Path(profile_file).stem

        in_filepath = self.input_dir / input_file
        specific_out_dir = self.output_dir / input_name
        specific_out_dir.mkdir(parents=True, exist_ok=True)

        out_filename = f"{input_name}_{profile_name}.wav"
        out_filepath = specific_out_dir / out_filename

        try:
            x, fs = sf.read(in_filepath)
            with open(self.profiles_dir / profile_file, 'r') as f:
                profile_data = json.load(f)

            # Send to Engine!
            processed_x = engine.process_audio(x, fs, profile_data)
            sf.write(out_filepath, processed_x, fs)

            self.update_output_list()

            # Automatically select the newly created file and draw its graphs!
            idx = self.output_listbox.get(0, tk.END).index(out_filename)
            self.output_listbox.selection_set(idx)
            self.on_output_select(None)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChangerApp(root)
    root.mainloop()
