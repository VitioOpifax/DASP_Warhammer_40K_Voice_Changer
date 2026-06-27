# Warhammer 40K Voice Changer

A modular Digital Audio Signal Processing (DASP) pipeline built in Python.  
This project transforms clean, uncompressed `.wav` speech recordings into stylized character voices inspired by the Warhammer 40,000 universe.

---

## 🧠 Architecture

The system is structured into two main components:

- `gui.py` → User interface
- `engine.py` → Core DSP processing pipeline logic

Each block is implemented as a modular processing stage, allowing flexible chaining and experimentation.

---

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```
---

## How to Use

- First select a input from the gui (make sure that there is a .wav file in the Input folder). You can play the file with the button near the selection menu.
- Select an already existing profile OR create a new one. After that press "PROCESS AUDIO" button to send the file through the pipeline.
- Your output will be automatically selected under the "Generated Output Files" tab. You can select other files and listen to them by pressing the "Play Selected" button.

INFO:
- The graphs on the side changes based on the selected output file.
- If you do manual editing of the files outside of the application you can press the "Refresh" button to update the application.
