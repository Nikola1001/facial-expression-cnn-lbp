"""Small local Tkinter image classification interface."""

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .infer import load_model, predict

DISPLAY_NAMES = {"angry": "Злость", "disgust": "Отвращение", "fear": "Страх",
                 "happy": "Радость", "sad": "Грусть", "surprise": "Удивление",
                 "neutral": "Нейтрально"}


def main():
    parser = argparse.ArgumentParser(description="Open image classification UI")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/fer/best.pt"))
    args = parser.parse_args()
    model, device = load_model(args.checkpoint)

    root = tk.Tk()
    root.title("Распознавание выражения лица")
    root.geometry("420x570")
    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Распознавание выражения лица", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Выберите фотографию с одним лицом анфас.").pack(anchor="w", pady=(4, 12))
    preview = ttk.Label(frame)
    preview.pack(pady=8)
    result = ttk.Label(frame, text="", font=("Segoe UI", 13, "bold"))
    result.pack(anchor="w", pady=(10, 6))
    detail = ttk.Label(frame, text="", justify="left")
    detail.pack(anchor="w")

    def open_image():
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        try:
            with Image.open(path) as source:
                image = source.copy()
            probabilities, crop = predict(image, model, device)
            shown = crop.copy()
            shown.thumbnail((320, 320))
            preview.image = ImageTk.PhotoImage(shown)
            preview.configure(image=preview.image)
            ranked = sorted(probabilities.items(), key=lambda item: -item[1])
            result.configure(text=f"{DISPLAY_NAMES[ranked[0][0]]} — {ranked[0][1]:.1%}")
            detail.configure(text="\n".join(f"{DISPLAY_NAMES[name]:15s} {score:.1%}"
                                             for name, score in ranked))
        except Exception as exc:
            messagebox.showerror("Не удалось обработать изображение", str(exc))

    ttk.Button(frame, text="Открыть изображение", command=open_image).pack(anchor="w")
    root.mainloop()


if __name__ == "__main__":
    main()
