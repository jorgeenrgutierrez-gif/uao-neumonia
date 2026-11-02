"""Vista Tkinter: botones, imágenes y resultado.

Regla de dependencia: este módulo SOLO conoce `integrator`. No importa
TensorFlow, Keras ni `load_model`.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showerror, showinfo

from PIL import Image, ImageTk

from src import integrator

PREVIEW_SIZE = (250, 250)


class App:
    def __init__(self, root: tk.Tk | None = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("815x560")
        self.root.resizable(False, False)
        bold = font.Font(weight="bold")

        ttk.Label(self.root, text="Imagen Radiográfica", font=bold).place(x=110, y=65)
        ttk.Label(self.root, text="Imagen con Heatmap", font=bold).place(x=545, y=65)
        ttk.Label(self.root, text="Resultado:", font=bold).place(x=500, y=350)
        ttk.Label(self.root, text="Cédula Paciente:", font=bold).place(x=65, y=350)
        ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold,
        ).place(x=122, y=25)
        ttk.Label(self.root, text="Probabilidad:", font=bold).place(x=500, y=400)

        self.patient_id = tk.StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.patient_id, width=10)
        box = {"bg": "white", "fg": "black", "relief": "solid", "borderwidth": 1}
        self.text_img1 = tk.Text(self.root, width=31, height=15, **box)
        self.text_img2 = tk.Text(self.root, width=31, height=15, **box)
        self.text2 = tk.Text(self.root, **box)
        self.text3 = tk.Text(self.root, **box)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(self.root, text="Cargar Imagen", command=self.load_img_file)
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(self.root, text="Guardar", command=self.save_results_csv)

        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)
        self.text1.focus_set()

        self.array = None
        self.original = None
        self.prediction = None
        self.img1 = None
        self.img2 = None
        self.report_id = 0

    def run(self) -> None:
        self.root.mainloop()

    def load_img_file(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Select image", filetypes=integrator.SUPPORTED_FILETYPES
        )
        if not filepath:
            return
        try:
            self.array, self.original = integrator.load_image(filepath)
        except ValueError as exc:
            showerror(title="Imagen", message=str(exc))
            return
        self.img1 = ImageTk.PhotoImage(self.original.resize(PREVIEW_SIZE, Image.LANCZOS))
        self.text_img1.delete("1.0", "end")
        self.text_img1.image_create("end", image=self.img1)
        self.button1["state"] = "enabled"

    def run_model(self) -> None:
        try:
            self.prediction = integrator.predict(self.array)
        except FileNotFoundError as exc:
            showerror(title="Modelo", message=str(exc))
            return
        heatmap = Image.fromarray(self.prediction.heatmap).resize(PREVIEW_SIZE, Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(heatmap)
        self.text_img2.delete("1.0", "end")
        self.text_img2.image_create("end", image=self.img2)
        self.text2.delete("1.0", "end")
        self.text3.delete("1.0", "end")
        self.text2.insert("end", self.prediction.label)
        self.text3.insert("end", f"{self.prediction.probability:.2f}%")

    def save_results_csv(self) -> None:
        if self.prediction is None:
            showinfo(title="Guardar", message="Primero ejecute una predicción.")
            return
        integrator.save_result_csv(
            self.text1.get(), self.prediction.label, self.prediction.probability
        )
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self) -> None:
        if self.prediction is None:
            showinfo(title="PDF", message="Primero ejecute una predicción.")
            return
        integrator.export_pdf(
            self.text1.get(), self.original, self.prediction, f"Reporte{self.report_id}.pdf"
        )
        self.report_id += 1
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self) -> None:
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            for widget in (self.text2, self.text3, self.text_img1, self.text_img2):
                widget.delete("1.0", "end")
            self.array = self.original = self.prediction = None
            self.button1["state"] = "disabled"
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main() -> int:
    App().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
